import json
import os

from google import genai
from dotenv import load_dotenv

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={
        "api_version": "v1beta"
    }
)

# Load memory
with open(BASE_DIR / "memory.json", "r") as file:
    memory = json.load(file)

# Convert memory into instructions
memory_text = json.dumps(memory)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config={
        "system_instruction": f"""
        You are a helpful AI assistant.

        Here is some information about the user:
        {memory_text}

        Use this information when relevant.
        """
    }
)

print("Memory Chatbot Started!")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = chat.send_message(question)

    print("\nAI:", response.text)