"""
    pip install -U ollama numpy
    pip install torch transformers
    ollama pull qwen3-embedding:0.6b
    ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M

    Put .txt / .md files in ./data
    A file may start with a line like ">> https://link"; that link is then the source of its chunks
    python leo.py
    --reindex (force rebuild of vector db)
"""

import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama
import yaml

# ---- Config ----------------------------------------------------------------
DOCS_DIR = Path("data")
INDEX_FILE = Path("index.json")
ROOT_DIR = Path(__file__).resolve().parent

EMBED_MODEL = "qwen3-embedding:0.6b"
# Qwen3-Embedding is instruction-aware: documents are embedded as-is, and every query gets
# a one-sentence task description in front of it (format: "Instruct: <task>\nQuery:<question>").
# Ollama does not add this for you. Keep the task in English and describe the real use case.
# If you switch embedding models, update these to match that model's card.
EMBED_TASK = "Given a student's question about campus services, people, or policies, retrieve the passages that answer it"
DOC_PREFIX = ""
QUERY_PREFIX = f"Instruct: {EMBED_TASK}\nQuery:"

MAX_ANSWER_TOKENS = 768   # cap on answer length; generation is the slow part on CPU
KEEP_ALIVE = "30m"        # keep models loaded in RAM between questions

CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 80        # characters repeated between neighboring chunks of one section
TOP_K_DENSE = 30          # chunks taken from dense (embedding) retrieval
TOP_K_SPARSE = 30         # chunks taken from sparse (BM25 keyword) retrieval
TOP_K_FUSED = 8           # chunks passed to the LLM (best by rerank score, or by RRF score if not reranking)
USE_RERANKING = True      # True: rerank every fused chunk; False: take the top TOP_K_FUSED by RRF score
RRF_K = 60                # reciprocal rank fusion constant: score = 1 / (RRF_K + rank)
BM25_K1 = 1.5             # BM25 term-frequency saturation
BM25_B = 0.75             # BM25 document-length normalization
NUM_CTX = 16384           # LLM context window (tokens); must fit the final chunks (~150 tokens each) plus the prompt
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"   # downloaded from Hugging Face on first use
RERANK_TASK = "Given a student's question about campus services, people, or policies, retrieve the passages that answer it"
RERANK_BATCH = 16         # chunks scored per forward pass; lower it if the GPU runs out of memory
RERANK_MAX_LENGTH = 1024  # max tokens per (instruction + question + chunk)
RERANK_DEVICE = "auto"     # "auto" uses a GPU if one is usable; or force "cuda" / "cpu"
RERANK_DTYPE = "auto"      # "auto" picks the best for your GPU; or "float32" / "float16" / "bfloat16"
EXTENSIONS = {".txt", ".md"}


# Load llm.yaml into a dictionary
with open(ROOT_DIR / "config" / "llm.yaml", 'r') as file:
    llm_config = yaml.safe_load(file)


# ---- Model calls (swap these functions when moving to AWS) ------------------
def _embed(texts: list[str]) -> np.ndarray:
    response = ollama.embed(model=EMBED_MODEL, input=texts, keep_alive=KEEP_ALIVE)
    return np.array(response["embeddings"], dtype=np.float32)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed document chunks (adds the document prefix)."""
    return _embed([DOC_PREFIX + t for t in texts])


def embed_query(query: str) -> np.ndarray:
    """Embed a user question (adds the query prefix)."""
    return _embed([QUERY_PREFIX + query])[0]

def _trim_start(pieces):
    """Strip leading whitespace from the start of the whole reply."""
    started = False
    for piece in pieces:
        if not started:
            piece = piece.lstrip()
            if not piece:
                continue
            started = True
        yield piece

def generate_stream(messages: list[dict]):
    """Yield the LLM's reply piece by piece so text appears as it is produced."""
    stream = ollama.chat(
        model=llm_config['model'],
        messages=messages,
        stream=True,
        keep_alive=KEEP_ALIVE,
        think=False,
        options={
            "num_predict": llm_config['max_answer_tokens'],
            "temperature": llm_config['temperature'],
            "num_ctx": NUM_CTX,
            "think": False
            },
        
    )
    yield from _trim_start(chunk["message"]["content"] for chunk in stream)


# ---- Ingestion --------------------------------------------------------------
# A file may start with ">> <link>"; that link then becomes the source of every chunk in the file.
LINK_LINE = re.compile(r"^\s*>>\s*(\S.*?)\s*$")
MD_LINK = re.compile(r"^\[[^\]]*\]\(([^)\s]+)[^)]*\)$")


def extract_link(text: str) -> tuple[str | None, str]:
    """If the first line is ">> <link>", return (link, text without that line); else (None, text)."""
    text = text.lstrip("\ufeff")  # drop a byte-order mark if the editor added one
    first, _, rest = text.partition("\n")
    match = LINK_LINE.match(first)
    if not match:
        return None, text
    link = match.group(1)
    md = MD_LINK.match(link)
    if md:
        link = md.group(1)                   # [label](https://...) -> https://...
    elif link[0] in "[<" and link[-1] in "]>":
        link = link[1:-1].strip()            # [https://...] or <https://...> -> https://...
    return (link, rest) if link else (None, text)


def file_creation_date(path: Path) -> str:
    """Creation date of a file as YYYY-MM-DD.

    Uses the real creation time where the OS provides it (Windows, macOS). Elsewhere (typically
    Linux) Python cannot read it, so this falls back to the last metadata-change time.
    """
    st = path.stat()
    timestamp = getattr(st, "st_birthtime", None) or st.st_ctime
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


# Abbreviations that end in a period but do not end a sentence.
ABBREVIATIONS = {"a.m.", "p.m.", "dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "ave.",
                 "blvd.", "rm.", "dept.", "e.g.", "i.e.", "vs."}
# A sentence break (. ! ? then whitespace then a capital/digit/quote) or any line break.
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def split_units(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentences/lines, keeping the original whitespace."""
    spans, start = [], 0
    for m in BOUNDARY.finditer(text):
        piece = text[start:m.start()]
        words = piece.split()
        # Don't split after an abbreviation such as "p.m." or "Dr." (a line break still splits).
        if "\n" not in m.group() and words and words[-1].lower() in ABBREVIATIONS:
            continue
        if piece.strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))

    # Break any single run of text longer than a chunk at a word boundary.
    out = []
    for s, e in spans:
        while e - s > CHUNK_SIZE:
            cut = text.rfind(" ", s, s + CHUNK_SIZE)
            cut = cut if cut > s else s + CHUNK_SIZE
            out.append((s, cut))
            s = cut
        out.append((s, e))
    return out


def pack_chunks(body: str) -> list[str]:
    """Pack whole sentences/lines into chunks of up to CHUNK_SIZE characters.

    Neighboring chunks repeat up to CHUNK_OVERLAP characters, in whole sentences.
    """
    units = split_units(body)
    chunks, i = [], 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and units[j + 1][1] - units[i][0] <= CHUNK_SIZE:
            j += 1
        chunks.append(body[units[i][0]:units[j][1]].strip())
        if j + 1 >= len(units):
            break
        nxt = j + 1  # step back over trailing units that fit in the overlap
        while nxt - 1 > i and units[j][1] - units[nxt - 1][0] <= CHUNK_OVERLAP:
            nxt -= 1
        i = nxt
    return chunks


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Split a document into (heading, body) sections.

    A heading is a markdown '#' line or a short line ending in '?'. Markdown headings
    nest, so a section can be labeled 'Housing > Move-out > Deadlines'.
    """
    sections, trail, leaf, body = [], {}, "", []

    def label() -> str:
        return " > ".join([trail[k] for k in sorted(trail)] + ([leaf] if leaf else []))

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((label(), content))
        body.clear()

    for line in text.splitlines():
        stripped = line.strip()
        md = MD_HEADING.match(stripped)
        if md:
            flush()
            level = len(md.group(1))
            trail = {k: v for k, v in trail.items() if k < level}
            trail[level], leaf = md.group(2), ""
        elif 0 < len(stripped) <= 120 and stripped.endswith("?"):
            flush()
            leaf = stripped
        else:
            body.append(line)
    flush()
    return sections


def chunk_document(rel_path: Path, text: str, source: str, file_creation_date: str) -> list[dict]:
    """Chunk one file and pair every chunk with its metadata.

    Each record holds: source, file_creation_date, header (the section heading, "" if none),
    content (the bare chunk, shown to the LLM) and text (folder > file > section, then the chunk;
    this is what gets embedded, keyword-searched and reranked, so it still makes sense on its own).
    """
    where = [p.replace("-", " ").replace("_", " ") for p in rel_path.with_suffix("").parts]
    records = []
    for header, body in parse_sections(text):
        context = " > ".join(where + ([header] if header else []))
        for chunk in pack_chunks(body):
            records.append({"source": source, "file_creation_date": file_creation_date,
                            "header": header, "content": chunk, "text": f"{context}\n{chunk}"})
    return records


def index_meta() -> dict:
    """Settings that change the vectors; a mismatch means the index is stale."""
    return {
        "embed_model": EMBED_MODEL,
        "doc_prefix": DOC_PREFIX,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunker": "sections-v2",  # bump when chunking logic changes
    }


def build_index() -> dict:
    """Read every doc, chunk it, embed the chunks, and save to disk."""
    records = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        link, text = extract_link(text)  # ">> link" first line, if any, is removed from the text
        records.extend(chunk_document(path.relative_to(DOCS_DIR), text,
                                      source=link or path.name, file_creation_date=file_creation_date(path)))

    if not records:
        sys.exit(f"No {', '.join(EXTENSIONS)} files found in ./{DOCS_DIR}")

    print(f"Embedding {len(records)} chunks...")
    vectors = []
    for i in range(0, len(records), 32):  # batch to keep requests small
        batch = [r["text"] for r in records[i:i + 32]]
        vectors.append(embed_documents(batch))
    matrix = np.vstack(vectors)

    index = {"meta": index_meta(), "records": records, "embeddings": matrix.tolist()}
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    return index


def load_index(force_rebuild: bool = False) -> tuple[list[dict], np.ndarray]:
    index = None
    if INDEX_FILE.exists() and not force_rebuild:
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if index.get("meta") != index_meta():
            print("Index was built with different settings; rebuilding.")
            index = None
    if index is None:
        index = build_index()
    matrix = np.array(index["embeddings"], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)  # normalize for cosine similarity
    return index["records"], matrix


# ---- Retrieval + answering --------------------------------------------------
# Common words that carry no keyword signal for BM25.
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
             "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "what",
             "when", "where", "which", "who", "will", "with"}
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric words with stopwords removed (no stemming)."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 keyword scoring. The corpus is small, so no external library is needed."""

    def __init__(self, texts: list[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in docs], dtype=np.float32)
        self.avg_len = max(float(self.lengths.mean()), 1.0) if docs else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(chunk index, count)]
        for i, d in enumerate(docs):
            for term, tf in d.items():
                self.postings.setdefault(term, []).append((i, tf))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        """Return one BM25 score per chunk (0 for chunks sharing no term with the query)."""
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, []):
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores


_bm25_cache: dict[int, BM25] = {}


def get_bm25(records: list[dict]) -> BM25:
    """Build the BM25 index once per loaded set of records (it is fast and not saved to disk)."""
    key = id(records)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25([r["text"] for r in records])
    return _bm25_cache[key]


# ---- Reranking (Qwen3-Reranker-0.6B) -----------------------------------------
# Ollama has no rerank endpoint, so the reranker runs locally through transformers.
# It reads "<instruction> + query + chunk" and answers "yes" or "no"; the score is the
# probability of "yes" (0 to 1). Prompt format is from the model card.
RERANK_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
                 "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                 "<|im_end|>\n<|im_start|>user\n")
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_reranker = None


def _gpu_usable(torch) -> bool:
    """True if a CUDA GPU exists AND this PyTorch build ships kernels that can run on it.

    Newer PyTorch CUDA builds dropped older GPUs (e.g. Pascal / GTX 10-series), and on those
    torch.cuda.is_available() is still True but every GPU call fails.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    capability = major * 10 + minor
    for arch in torch.cuda.get_arch_list():  # e.g. "sm_60", "sm_90", "compute_120"
        kind, _, digits = arch.partition("_")
        digits = "".join(ch for ch in digits if ch.isdigit())
        if not digits:
            continue
        version = int(digits)
        # Compiled code runs on the same GPU generation (equal or newer minor version);
        # PTX ("compute_XX") also runs on any newer GPU.
        if (kind == "sm" and version // 10 == major and version <= capability) or \
           (kind == "compute" and version <= capability):
            return True
    return False


def _pick_device_and_dtype(torch):
    """Choose where and how to run the reranker (RERANK_DEVICE / RERANK_DTYPE can override)."""
    if RERANK_DEVICE == "cpu":
        device = "cpu"
    else:
        usable = _gpu_usable(torch)
        if RERANK_DEVICE == "cuda" and not usable:
            raise RuntimeError("RERANK_DEVICE is 'cuda' but no GPU this PyTorch build supports was found. "
                               "Install a PyTorch build that matches your GPU, or set RERANK_DEVICE = 'auto'.")
        if not usable and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"Note: {name} is not supported by this PyTorch build, so the reranker will run on the CPU. "
                  "Install a PyTorch build that still includes your GPU's architecture.")
        elif not usable and torch.version.cuda is None:
            print("Note: this PyTorch is a CPU-only build, so the reranker will run on the CPU. "
                  "Install a CUDA build of PyTorch to use a GPU.")
        device = "cuda" if usable else "cpu"

    if RERANK_DTYPE != "auto":
        dtype = getattr(torch, RERANK_DTYPE)
    elif device == "cuda":
        major = torch.cuda.get_device_capability(0)[0]
        # bfloat16 on Ampere and newer; float16 on Volta/Turing; float32 on Pascal, whose float16 is very slow.
        dtype = torch.bfloat16 if major >= 8 else torch.float16 if major == 7 else torch.float32
    else:
        dtype = torch.float32
    return device, dtype


def get_reranker():
    """Load the reranker once; the first question after startup pays the load time."""
    global _reranker
    if _reranker is None:
        import os
        import torch
        # transformers 5 loads model weights on a thread pool, which hard-crashes (no error
        # message) on some Windows setups. This makes it load on the main thread instead.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device, dtype = _pick_device_and_dtype(torch)
        print(f"Loading reranker {RERANK_MODEL} on {device} ({str(dtype).replace('torch.', '')}), first question only...")
        tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL).to(device=device, dtype=dtype).eval()
        _reranker = (tokenizer, model,
                     tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no"))
    return _reranker


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Return one relevance score (probability of "yes", 0 to 1) per text for this query."""
    import torch
    tokenizer, model, yes_id, no_id = get_reranker()
    prefix = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
    scores = []
    for start in range(0, len(texts), RERANK_BATCH):
        pairs = [f"<Instruct>: {RERANK_TASK}\n<Query>: {query}\n<Document>: {t}"
                 for t in texts[start:start + RERANK_BATCH]]
        inputs = tokenizer(pairs, padding=False, truncation="longest_first", return_attention_mask=False,
                           max_length=RERANK_MAX_LENGTH - len(prefix) - len(suffix))
        for j, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][j] = prefix + ids + suffix
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=RERANK_MAX_LENGTH)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
        pair_logits = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
        scores += torch.nn.functional.log_softmax(pair_logits, dim=1)[:, 1].exp().tolist()
    return scores


def retrieve(query: str, records: list[dict], matrix: np.ndarray,
             k_dense: int = TOP_K_DENSE, k_sparse: int = TOP_K_SPARSE,
             k_final: int = TOP_K_FUSED) -> list[dict]:
    """Hybrid retrieval, optionally followed by reranking.

    Dense (embeddings) and sparse (BM25) results are merged with reciprocal rank fusion (RRF).
    If USE_RERANKING, every fused chunk is reranked and the best k_final are returned;
    otherwise the k_final chunks with the highest RRF score are returned.
    """
    # Dense: cosine similarity between the query embedding and every chunk embedding.
    q = embed_query(query)
    q /= np.linalg.norm(q)
    dense_top = [int(i) for i in np.argsort(matrix @ q)[::-1][:k_dense]]

    # Sparse: BM25 keyword scores. Chunks sharing no query term (score 0) are left out.
    sparse_scores = get_bm25(records).scores(query)
    sparse_top = [int(i) for i in np.argsort(sparse_scores)[::-1][:k_sparse] if sparse_scores[i] > 0]

    # Reciprocal rank fusion: each list adds 1 / (RRF_K + rank) to a chunk's score.
    dense_rank = {i: r for r, i in enumerate(dense_top, start=1)}
    sparse_rank = {i: r for r, i in enumerate(sparse_top, start=1)}
    fused = {}
    for ranks in (dense_rank, sparse_rank):
        for i, r in ranks.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)

    # Every fused chunk, best RRF first (so reranking ties keep their RRF order).
    candidates = sorted(fused, key=fused.get, reverse=True)
    if USE_RERANKING:
        relevance = rerank_scores(query, [records[i]["text"] for i in candidates])
        order = sorted(range(len(candidates)), key=lambda n: relevance[n], reverse=True)[:k_final]
    else:
        relevance = [None] * len(candidates)  # no rerank score without reranking
        order = list(range(len(candidates)))[:k_final]  # candidates are already in RRF order

    return [{**records[candidates[n]], "rrf_score": fused[candidates[n]], "rerank_score": relevance[n],
             "dense_rank": dense_rank.get(candidates[n]), "sparse_rank": sparse_rank.get(candidates[n])}
            for n in order]


def format_document(h: dict) -> str:
    """Wrap one chunk and its metadata in XML tags for the LLM. No header tag if there is no header."""
    parts = [f"<source>{h['source']}</source>",
             f"<file_creation_date>{h['file_creation_date']}</file_creation_date>"]
    if h.get("header"):
        parts.append(f"<header>{h['header']}</header>")
    parts.append(f"<context>{h['content']}</context>")
    return "<document>\n" + "\n".join(parts) + "\n</document>"


def build_messages(query: str, hits: list[dict]) -> list[dict]:
    context = "\n\n".join(format_document(h) for h in hits)

    if not context:
        context = "(No reference documents retrieved.)"

    user_content = f"""\
    <retrieved_context>
    {context}
    </retrieved_context>

    <student_question>
    {query}
    </student_question>
    """
    

    return [
        {"role": "system", "content": llm_config['system_prompt']},
        {"role": "user", "content": user_content}
    ]


# ---- CLI --------------------------------------------------------------------
def main():
    records, matrix = load_index(force_rebuild="--reindex" in sys.argv)
    print(f"Loaded {len(records)} chunks.")
    while True:
        query = input("\nInput >> ").strip()
        if not query:
            break

        # Retrieval is fast, so show what was found right away...
        hits = retrieve(query, records, matrix)
        print("\nMost relevant passages:")
        print("(Note: RRF scores are multiplied by 100 for readability.)")
        for h in hits:
            preview = " ".join(h["text"].split())[:140]
            rerank = f"rerank {h['rerank_score']:.2f} | " if h["rerank_score"] is not None else ""
            print(f"  [{h['source']}] ({rerank}RRF {h['rrf_score'] * 100:.2f}) {preview}...")

        # ...then stream the generated answer as it is produced.
        print("\nOutput >> ", end="", flush=True)
        for piece in generate_stream(build_messages(query, hits)):
            print(piece, end="", flush=True)
        print()
        print("** Answer Complete **")

if __name__ == "__main__":
    main()