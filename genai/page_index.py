import json
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

# Load page-aware chunks
with open(
    DOCUMENTS_DIR / "page_chunks.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)

# Extract text
texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(texts)} chunks.")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Build FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# Save index
faiss.write_index(
    index,
    str(DOCUMENTS_DIR / "page.index")
)

print("\nFAISS index created successfully!")
print("Total vectors:", index.ntotal)
print("Embedding dimension:", dimension)