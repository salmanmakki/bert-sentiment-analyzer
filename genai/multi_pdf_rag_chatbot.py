import os
os.environ["USE_TF"] = "0"

import json
import faiss
import numpy as np

from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv


# --------------------------------
# PATHS
# --------------------------------

BASE_DIR = Path(r"C:\deep-learning\documents")

CHUNKS_PATH = BASE_DIR / "multi_chunks.json"
INDEX_PATH = BASE_DIR / "multi.index"


# --------------------------------
# LOAD GEMINI API
# --------------------------------

load_dotenv(Path(__file__).parent / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


# --------------------------------
# LOAD DOCUMENTS
# --------------------------------

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]

print("Total chunks loaded:", len(chunks))


# --------------------------------
# LOAD FAISS INDEX
# --------------------------------

index = faiss.read_index(str(INDEX_PATH))

if index.ntotal != len(chunks):
    raise ValueError("FAISS index and chunks count do not match!")

print("FAISS index loaded:", index.ntotal)


# --------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-Encoder for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --------------------------------
# BUILD BM25 INDEX
# --------------------------------

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)


# --------------------------------
# HYBRID SEARCH + RRF
# --------------------------------

def hybrid_search(query):

    # Semantic search
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    semantic_scores, semantic_ids = index.search(
        query_embedding, 10
    )

    # Keyword search
    tokenized_query = query.lower().split()

    keyword_scores = bm25.get_scores(tokenized_query)

    keyword_ids = np.argsort(keyword_scores)[::-1][:10]

    # Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, chunk_id in enumerate(semantic_ids[0]):
        if chunk_id != -1:
            rrf_scores[int(chunk_id)] = (
                rrf_scores.get(int(chunk_id), 0)
                + 1 / (60 + rank + 1)
            )

    for rank, chunk_id in enumerate(keyword_ids):
        rrf_scores[int(chunk_id)] = (
            rrf_scores.get(int(chunk_id), 0)
            + 1 / (60 + rank + 1)
        )

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    return ranked_ids[:10]


# --------------------------------
# CROSS-ENCODER RERANKING
# --------------------------------

def rerank_results(query, candidate_ids):

    pairs = [
        [query, texts[chunk_id]]
        for chunk_id in candidate_ids
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidate_ids, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        chunks[chunk_id]
        for chunk_id, score in ranked_results[:5]
    ]


# --------------------------------
# GEMINI ANSWER GENERATION
# --------------------------------

def generate_answer(query, retrieved_chunks):

    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context += f"""
[Source {i}]
Document: {chunk['source']}
Page: {chunk['page']}

Content:
{chunk['text']}

"""

    prompt = f"""
You are an AI assistant that answers questions using the provided documents.

Rules:
1. Answer using only the provided context.
2. Do not invent facts.
3. If the answer is not present in the context, clearly say so.
4. Cite the supporting source using [Source 1], [Source 2], etc.
5. Give a clear and helpful answer.

Context:
{context}

Question:
{query}

Answer:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as error:

        print("\nGemini API Error:", error)

        return "I couldn't generate an answer. Please try again."


# --------------------------------
# CHAT LOOP
# --------------------------------

print("\n================================")
print("       MULTI-PDF RAG CHATBOT")
print("================================")

print("Loaded documents:")

for source in sorted(set(chunk["source"] for chunk in chunks)):
    print("-", source)

print("\nAsk questions about your PDFs.")
print("Type 'exit' to stop.\n")


while True:

    query = input("\nYou: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not query:
        continue

    # Step 1: Hybrid retrieval
    candidate_ids = hybrid_search(query)

    # Step 2: Reranking
    retrieved_chunks = rerank_results(
        query,
        candidate_ids
    )

    # Step 3: Generate answer
    answer = generate_answer(
        query,
        retrieved_chunks
    )

    print("\nAssistant:")
    print(answer)

    # Step 4: Display sources
    print("\nSources used:")

    for i, chunk in enumerate(retrieved_chunks, start=1):

        print(
            f"[Source {i}] "
            f"{chunk['source']} | "
            f"Page {chunk['page']}"
        )