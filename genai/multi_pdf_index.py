import os
os.environ["USE_TF"] = "0"

import json
import fitz
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIGURATION
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "documents" / "pdfs"

OUTPUT_JSON = BASE_DIR / "documents" / "multi_chunks.json"

OUTPUT_INDEX = BASE_DIR / "documents" / "multi.index"


# -----------------------------
# TEXT CHUNKING
# -----------------------------

def chunk_text(text, chunk_size=100, overlap=20):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


# -----------------------------
# EXTRACT ALL PDFS
# -----------------------------

def extract_all_pdfs():

    all_chunks = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        document = fitz.open(pdf_path)

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


# -----------------------------
# SAVE CHUNKS
# -----------------------------

chunks = extract_all_pdfs()

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"\nTotal chunks: {len(chunks)}")


# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    chunk["text"]
    for chunk in chunks
]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# -----------------------------
# BUILD FAISS INDEX
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    str(OUTPUT_INDEX)
)


print("\nMulti-PDF indexing completed!")
print("Total documents:", len(list(PDF_DIR.glob("*.pdf"))))
print("Total chunks:", index.ntotal)
print("Embedding dimension:", dimension)

print("\nSaved files:")
print(OUTPUT_JSON)
print(OUTPUT_INDEX)