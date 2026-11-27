#!/usr/bin/env python3
"""
Single-page text extractor.

Fetches only the URL(s) given (no links are followed) and saves each page's
visible text to a .txt file named after the URL path. The first line of each
file is the TEXT_FLAG below (omitted if empty). Each <h3> becomes a "# heading"
and the text after it stays in that section until the next <h3>.

<a class="btn-link" href="URL"><span>Text</span></a> becomes "Text (URL)".

Any element whose class or id appears in the lists below is skipped, along with
everything inside it.

Usage:
    python crawler.py https://example.com/about
    python crawler.py https://example.com/a https://example.com/b -o output

Requires:
    pip install requests beautifulsoup4
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

# The announcements url where information is extracted from
URL = "https://fhu-announcements.netlify.app/"

# Written as the entire first line of every saved file (nothing is written if empty).
TEXT_FLAG = f"The following information is from TJ Kirk's official announcements, extracted from the link {URL}"

# Returned by get_page_text() when a page can't be fetched.
FAILURE_FLAG = f"Failed to fetch announcements, they can be found manually by visiting {URL}"

# Links with this class are written as "Text (absolute link)".
LINK_CLASS = "btn-link"

# Elements with these classes / ids are excluded from the extracted text.
EXCLUDE_CLASSES = ["mobile-nav", "breadcrumbs", "sidebar"]
EXCLUDE_IDS = []

# Private marker used internally to flag h3 headers while extracting.
H3_MARK = "\ue000"

# Tags that never contain page text worth keeping.
EXCLUDE_TAGS = ["script", "style", "noscript", "template", "svg", "iframe",
                "nav", "footer"]


def url_to_filename(url):
    """https://site.com/docs/intro -> docs_intro.txt ('/' -> index.txt)."""
    parsed = urlparse(url)
    name = re.sub(r"\.(html?|php|aspx?|jsp)$", "", parsed.path.strip("/"),
                  flags=re.IGNORECASE).replace("/", "_") or "index"
    if parsed.query:
        name += "_" + parsed.query.replace("=", "-").replace("&", "_")
    return (re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_.") or "index")[:150] + ".txt"


def extract_text(html, base_url=""):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(EXCLUDE_TAGS):
        tag.decompose()
    for cls in EXCLUDE_CLASSES:
        for tag in soup.find_all(class_=cls):
            tag.decompose()
    for id_ in EXCLUDE_IDS:
        for tag in soup.find_all(id=id_):
            tag.decompose()

    # Turn <a class="btn-link" href=...>Text</a> into "Text (link)"
    for a in soup.find_all("a", class_=LINK_CLASS):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = (a.get("href") or "").strip()
        link = urljoin(base_url, href) if href else ""
        a.replace_with(NavigableString(f"{text} ({link})" if link else text))

    # Mark each h3 with a sentinel so it can be turned into a section header
    for h3 in soup.find_all("h3"):
        title = " ".join(h3.get_text(" ", strip=True).split())
        h3.replace_with(NavigableString(f"\n{H3_MARK}{title}\n"))

    soup.smooth()  # merge adjacent text so inline links stay within their sentence

    output = []
    for line in soup.get_text(separator="\n").splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if not line:
            continue
        if line.startswith(H3_MARK):
            if output:
                output.append("")  # blank line between sections
            output.append("# " + line[len(H3_MARK):].strip())
        else:
            output.append(line)
    return "\n".join(output)


# Extract announcement data from URL
def get_latest_announcement_content():
    """
    Fetch URL and return its extracted text as a string.

    Applies the same extraction as the command-line tool (exclusion lists,
    h3 section headers, btn-link conversion). TEXT_FLAG is not included.
    Returns FAILURE_FLAG if the page can't be fetched.
    """
    try:
        resp = requests.get(URL, timeout=15, headers={"User-Agent": "SimpleTextCrawler/1.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return FAILURE_FLAG
    return extract_text(resp.text, resp.url)


def main():
    parser = argparse.ArgumentParser(description="Save the text of one or more pages.")
    parser.add_argument("urls", nargs="+", help="Page URL(s) to extract")
    parser.add_argument("-o", "--output", default="output",
                        help="Output directory (default: output)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for url in args.urls:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "SimpleTextCrawler/1.0"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[error] {url} ({exc})")
            continue

        filename = url_to_filename(resp.url)
        body = extract_text(resp.text, resp.url)
        content = f"{TEXT_FLAG}\n\n{body}\n" if TEXT_FLAG else f"{body}\n"
        (out / filename).write_text(content, encoding="utf-8")
        print(f"[saved] {resp.url} -> {filename}")


if __name__ == "__main__":
    main()