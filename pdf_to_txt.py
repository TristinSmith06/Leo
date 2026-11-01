from pypdf import PdfReader

# Define file paths
pdf_path = "fhu-handbook.pdf"
text_path = "fhu-student-handbook-2025-to-2026.txt"

# Initialize the reader
reader = PdfReader(pdf_path)

# Extract and save text page by page
with open(text_path, "w", encoding="utf-8") as f:
    for page in reader.pages:
        text = page.extract_text()
        if text:
            f.write(text + "\n")

print(f"Successfully converted {pdf_path} to {text_path}")