import os
import json
import faiss

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


MEMORY_FILE = Path(__file__).resolve().parent / "chat_history.json"


def load_memory():

    if MEMORY_FILE.exists():

        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return []

def save_memory(history):

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )

# -----------------------------
# PATHS AND CONFIGURATION
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

load_dotenv(Path(__file__).resolve().parent / ".env")

# Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------

with open(DOCUMENTS_DIR / "chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

index = faiss.read_index(str(DOCUMENTS_DIR / "pdf.index"))


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
# BM25 INITIALIZATION
# -----------------------------

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
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
        top_k
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
            rrf_scores.get(idx, 0) +
            1 / (60 + rank + 1)
        )

    for rank, idx in enumerate(keyword_results):
        rrf_scores[idx] = (
            rrf_scores.get(idx, 0) +
            1 / (60 + rank + 1)
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
        [query, chunks[idx]]
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
# GEMINI ANSWER GENERATION
# -----------------------------
def generate_answer(query, retrieved_results, history):

    history_text = "\n".join(
    [
        f"User: {item['question']}\nAI: {item['answer']}"
        for item in history[-5:]
    ]
)

    context = "\n\n".join(
        [
            f"[Source {i}]\n{chunks[idx]}"
            for i, (idx, score) in enumerate(
                retrieved_results,
                start=1
            )
        ]
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent facts.
2. If the answer is not available in the context, say:
   "I couldn't find this information in the provided PDF."
3. Keep the answer clear and informative.
4. Mention the source numbers supporting your answer.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as error:
        print(f"\nGemini API Error: {error}")
        return "Sorry, I couldn't generate an answer. Please try again."

        # -----------------------------
# QUERY REWRITING
# -----------------------------

def rewrite_query(query, history):

    # No history: use original question
    if not history:
        return query

    history_text = "\n".join(
        [
            f"User: {item['question']}\nAI: {item['answer']}"
            for item in history[-5:]
        ]
    )

    prompt = f"""
You are a query rewriting assistant.

Rewrite the latest question as a standalone question
using the conversation history.

Rules:
1. Resolve references such as he, she, it, and they.
2. Preserve the user's original meaning.
3. Do not answer the question.
4. If already clear, return it unchanged.
5. Return only the rewritten question.

Conversation history:
{history_text}

Latest question:
{query}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as error:
        print(f"Query rewriting failed: {error}")
        return query

# -----------------------------
# MAIN CHAT LOOP WITH MEMORY
# -----------------------------

conversation_history = load_memory()

print("\nPDF RAG Chatbot is ready!")
print("Conversation memory is enabled.")
print("Type 'exit' to stop.\n")

while True:

    query = input("You: ").strip()

    # Exit chatbot
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not query:
        continue

    try:
        # Step 1: Rewrite query using conversation history
        standalone_query = rewrite_query(
            query,
            conversation_history
        )

        print("\nSearch query:", standalone_query)

        # Step 2: Hybrid Search
        candidates = hybrid_search(
            standalone_query,
            top_k=20
        )

        # Step 3: Reranking
        results = rerank_results(
            standalone_query,
            candidates
        )

        # Step 4: Generate answer using Gemini
        answer = generate_answer(
            standalone_query,
            results,
            conversation_history
        )

        # Step 5: Display answer
        print("\nAI:", answer)

        # Step 6: Save conversation history
        conversation_history.append({
            "question": query,
            "answer": answer
        })
        save_memory(conversation_history)

        # Keep only the latest 5 conversations
        conversation_history = conversation_history[-5:]

        # Step 7: Display sources
        print("\nSources used:")

        for i, (idx, score) in enumerate(results, start=1):

            print(f"\nSource {i} | Score: {score:.4f}")
            print(chunks[idx][:300])

        print("\n" + "=" * 60)

    except Exception as error:

        print("\nError:", error)
        print("Please try asking your question again.")