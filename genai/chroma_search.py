import os
os.environ["USE_TF"] = "0"

import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer


# Database path
DB_PATH = Path(r"C:\deep-learning\documents\chroma_db")


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(DB_PATH)
)


# Load collection
collection = client.get_collection(
    name="study_documents"
)

print("Total documents:", collection.count())


# Load embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Search function
def search_documents(query):

    # Convert question into embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    print("\nSearch Results:")
    print("=" * 50)

    for i in range(len(results["ids"][0])):

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nResult {i + 1}")

        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Distance:", round(distance, 4))

        print("\nContent:")
        print(document)

        print("-" * 50)


# Interactive search
while True:

    query = input("\nAsk something (type exit to stop): ")

    if query.lower() in ["exit", "quit"]:
        break

    if query.strip():
        search_documents(query)