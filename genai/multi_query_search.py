import os
os.environ["USE_TF"] = "0"

import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ChromaDB
db = chromadb.PersistentClient(
    path=r"C:\deep-learning\documents\chroma_db"
)

collection = db.get_collection("study_documents")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_queries(question):

    prompt = f"""
Generate 3 alternative search queries for this question.

Return only the queries, one per line.
Do not number them.

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    if not response.text:
        return [question]

    alternatives = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]

    return [question] + alternatives[:3]


def multi_query_search(question):

    queries = generate_queries(question)

    print("\nGenerated queries:")
    for query in queries:
        print("-", query)

    unique_results = {}

    for query in queries:

        embedding = model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

        for i, chunk_id in enumerate(results["ids"][0]):

            unique_results.setdefault(
                chunk_id,
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
            )

    return list(unique_results.values())


while True:

    question = input("\nAsk a question (exit to stop): ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        continue

    results = multi_query_search(question)

    print("\nRetrieved chunks:")

    for i, result in enumerate(results[:5], 1):

        print(f"\nResult {i}")
        print("Source:", result["metadata"]["source"])
        print("Page:", result["metadata"]["page"])
        print(result["text"][:500])