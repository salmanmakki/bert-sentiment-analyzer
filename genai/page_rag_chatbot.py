import os
os.environ["USE_TF"] = "0"

import json
import faiss

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# -----------------------------
# CONFIGURATION
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"

load_dotenv(Path(__file__).resolve().parent / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


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

assert index.ntotal == len(chunks)


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

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    keyword_results = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

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

    return sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )


# -----------------------------
# RERANKING
# -----------------------------

def rerank_results(query, candidate_indices):

    pairs = [
        [query, texts[idx]]
        for idx in candidate_indices
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_indices, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:5]


# -----------------------------
# GEMINI ANSWER GENERATION
# -----------------------------

def generate_answer(query, results):

    context_parts = []

    for i, (idx, score) in enumerate(results, start=1):

        chunk = chunks[idx]

        context_parts.append(
            f"""
[Source {i}]
Document: {chunk['source']}
Page: {chunk['page']}
Content: {chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a PDF question-answering assistant.

Answer the question using ONLY the provided document context.

RULES:
1. Do not invent information.
2. If information is unavailable, say so.
3. Cite sources using [Source 1], [Source 2], etc.
4. Never invent page numbers.
5. Keep the answer clear and informative.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# -----------------------------
# MAIN CHAT LOOP
# -----------------------------

print("\nPage-Aware PDF RAG Chatbot Ready!")
print("Type 'exit' to stop.\n")

while True:

    query = input("You: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not query:
        continue

    try:

        candidates = hybrid_search(query)

        results = rerank_results(
            query,
            candidates
        )

        answer = generate_answer(
            query,
            results
        )

        print("\nAI:", answer)

        print("\nVerified Sources:")

        for i, (idx, score) in enumerate(results, start=1):

            chunk = chunks[idx]

            print(
                f"[Source {i}] "
                f"{chunk['source']} | "
                f"Page {chunk['page']} | "
                f"Score: {score:.4f}"
            )

        print("\n" + "=" * 60)

    except Exception as error:

        print("\nError:", error)
        print("Please try again.")