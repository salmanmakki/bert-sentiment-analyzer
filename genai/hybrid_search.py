import json
import faiss
import numpy as np

from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "documents" / "chunks.json"
INDEX_PATH = BASE_DIR / "documents" / "pdf.index"

# Load chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Load FAISS index
index = faiss.read_index(str(INDEX_PATH))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare BM25
tokenized_chunks = [
    chunk.lower().split() for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


def hybrid_search(query, k=5):

    # Semantic search
    query_embedding = model.encode([query]).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, semantic_ids = index.search(query_embedding, k)

    semantic_results = semantic_ids[0].tolist()

    # Keyword search
    keyword_scores = bm25.get_scores(query.lower().split())

    keyword_ids = np.argsort(keyword_scores)[::-1][:k].tolist()

    # Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, idx in enumerate(semantic_results):
        if idx >= 0:
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    for rank, idx in enumerate(keyword_ids):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    # Sort combined results
    final_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {
            "score": score,
            "content": chunks[idx]
        }
        for idx, score in final_results[:k]
    ]


# Test
question = input("Ask something about your PDF: ")

results = hybrid_search(question)

for i, result in enumerate(results, 1):

    print(f"\nResult {i}")
    print("RRF Score:", round(result["score"], 5))
    print("Content:", result["content"])