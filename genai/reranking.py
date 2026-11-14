import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent

# Load document chunks
with open(BASE_DIR / "documents" / "chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Load FAISS index
index = faiss.read_index(str(BASE_DIR / "documents" / "pdf.index"))

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Prepare BM25
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Reciprocal Rank Fusion
def hybrid_search(query, top_k=20):

    # Semantic search
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    semantic_results = [
        int(idx) for idx in indices[0] if idx != -1
    ]

    # Keyword search
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
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    for rank, idx in enumerate(keyword_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    return sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )


# User query
query = input("Ask something about your PDF: ")

# Retrieve candidates
candidate_indices = hybrid_search(query, top_k=20)

# Prepare question-document pairs
pairs = [
    [query, chunks[idx]]
    for idx in candidate_indices
]

# Rerank candidates
rerank_scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(candidate_indices, rerank_scores),
    key=lambda item: item[1],
    reverse=True
)

# Display best results
print("\nTop results after reranking:\n")

for rank, (idx, score) in enumerate(ranked_results[:5], start=1):

    print(f"\nResult {rank}")
    print(f"Relevance Score: {score:.4f}")
    print(f"Content: {chunks[idx]}")