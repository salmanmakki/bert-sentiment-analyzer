import os
os.environ["USE_TF"] = "0"

import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


# --------------------------------
# CONFIGURATION
# --------------------------------

DB_PATH = Path(r"C:\deep-learning\documents\chroma_db")

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# --------------------------------
# GEMINI CLIENT
# --------------------------------

client = genai.Client(
    api_key=api_key,
    http_options={"api_version": "v1beta"}
)


# --------------------------------
# CHROMADB CONNECTION
# --------------------------------

chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = chroma_client.get_collection(
    name="study_documents"
)

print("Total document chunks:", collection.count())


# --------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# RETRIEVAL
# --------------------------------

def retrieve_documents(query):

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):

        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# --------------------------------
# GENERATE ANSWER
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
You are an AI study assistant.

Answer the user's question using only the provided document context.

Rules:
1. Do not invent information.
2. If the answer is not available in the context, say so.
3. Cite supporting information using [Source 1], [Source 2], etc.
4. Explain concepts clearly and simply.
5. Do not cite a source unless it supports the statement.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return "Gemini returned an empty response."

    except Exception as error:

        print("\nGemini API Error:", error)

        return "Sorry, I couldn't generate an answer. Please try again."


# --------------------------------
# CHAT LOOP
# --------------------------------

print("\n================================")
print("       CHROMADB RAG CHATBOT")
print("================================")

print("\nLoaded documents:")

# Show available PDF names
all_data = collection.get(
    limit=collection.count(),
    include=["metadatas"]
)

sources = sorted({
    item["source"]
    for item in all_data["metadatas"]
})

for source in sources:
    print("-", source)

print("\nAsk questions about your PDFs.")
print("Type 'exit' to stop.")


while True:

    query = input("\nYou: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not query:
        continue

    # Retrieve relevant chunks
    retrieved_chunks = retrieve_documents(query)

    if not retrieved_chunks:
        print("\nNo relevant documents found.")
        continue

    # Generate answer
    answer = generate_answer(
        query,
        retrieved_chunks
    )

    print("\nAssistant:")
    print(answer)

    # Display retrieved sources
    print("\nRetrieved sources:")

    for i, chunk in enumerate(retrieved_chunks, start=1):

        print(
            f"[Source {i}] "
            f"{chunk['source']} | "
            f"Page {chunk['page']} | "
            f"Distance: {chunk['distance']:.4f}"
        )