import json
import fitz

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "documents" / "notes.pdf"

OUTPUT_PATH = BASE_DIR / "documents" / "page_chunks.json"


def chunk_text(text, chunk_size=100, overlap=20):

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def extract_page_chunks(pdf_path):

    document = fitz.open(pdf_path)

    all_chunks = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        if not text.strip():
            continue

        page_chunks = chunk_text(text)

        for chunk in page_chunks:

            all_chunks.append({
                "text": chunk,
                "page": page_number,
                "source": pdf_path.name
            })

    document.close()

    return all_chunks


chunks = extract_page_chunks(PDF_PATH)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"Created {len(chunks)} page-aware chunks.")
print(f"Saved to: {OUTPUT_PATH}")