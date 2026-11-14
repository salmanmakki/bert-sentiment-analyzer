import os
os.environ["USE_TF"] = "0"

import json
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(r"C:\deep-learning\documents")

CHUNKS_PATH = BASE_DIR / "multi_chunks.json"
DB_PATH = BASE_DIR / "chroma_db"


# Load chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Total chunks:", len(chunks))


# Load embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path=str(DB_PATH)
)


# Create collection
collection = client.get_or_create_collection(
    name="study_documents",
    metadata={"hnsw:space": "cosine"}
)


# Generate and store embeddings
print("\nGenerating embeddings and storing documents...")

batch_size = 100

for start in range(0, len(chunks), batch_size):

    batch = chunks[start:start + batch_size]

    ids = [
        f"chunk_{i}"
        for i in range(start, start + len(batch))
    ]

    documents = [
        chunk["text"]
        for chunk in batch
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": int(chunk["page"])
        }
        for chunk in batch
    ]

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Processed {min(start + batch_size, len(chunks))}/{len(chunks)}")


print("\nChromaDB indexing completed!")

print("Total stored documents:", collection.count())

print("Database location:", DB_PATH)