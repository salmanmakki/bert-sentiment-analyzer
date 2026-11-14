import os
os.environ["USE_TF"] = "0"

import json
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"


# -----------------------------
# LOAD PAGE-AWARE CHUNKS
# -----------------------------

with open(
    DOCUMENTS_DIR / "page_chunks.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


texts = [chunk["text"] for chunk in chunks]


# -----------------------------
# LOAD FAISS INDEX
# -----------------------------

index = faiss.read_index(
    str(DOCUMENTS_DIR / "page.index")
)

assert index.ntotal == len(chunks), "Index and chunks do not match!"


# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loading reranking model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -----------------------------
# BM25
# -----------------------------

tokenized_chunks = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_chunks)


# -----------------------------
# HYBRID SEARCH
# -----------------------------

def hybrid_search(query, top_k=20):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    _, semantic_indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    semantic_results = [
        int(idx)
        for idx in semantic_indices[0]
        if idx != -1
    ]

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    keyword_results = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    # Reciprocal Rank Fusion

    rrf_scores = {}

    for rank, idx in enumerate(semantic_results):

        rrf_scores[idx] = (
            rrf_scores.get(idx, 0)
            + 1 / (60 + rank + 1)
        )

    for rank, idx in enumerate(keyword_results):

        rrf_scores[idx] = (
            rrf_scores.get(idx, 0)
            + 1 / (60 + rank + 1)
        )

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    return ranked_indices


# -----------------------------
# CROSS-ENCODER RERANKING
# -----------------------------

def rerank_results(query, candidate_indices):

    pairs = [
        [query, texts[idx]]
        for idx in candidate_indices
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidate_indices, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results[:5]


# -----------------------------
# TEST SEARCH
# -----------------------------

if __name__ == "__main__":

    while True:

        query = input("\nAsk something about your PDF: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        candidates = hybrid_search(query)

        results = rerank_results(
            query,
            candidates
        )

        print("\nTop results:\n")

        for rank, (idx, score) in enumerate(results, start=1):

            chunk = chunks[idx]

            print(f"\nResult {rank}")
            print(f"Relevance Score: {score:.4f}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Content: {chunk['text'][:300]}")

            print("-" * 60)