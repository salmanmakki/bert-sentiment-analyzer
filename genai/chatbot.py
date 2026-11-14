from google import genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# Initialize Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Create chat session
chat = client.chats.create(
    model="gemini-2.5-flash",
    config={
        "system_instruction": """
        You are an experienced Python tutor.
        Explain concepts in simple English.
        Provide practical Python examples.
        Keep explanations concise.
        """
    }
)

print("Streaming Gemini Chatbot Started!")
print("Type 'exit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:

        print("\nAI: ", end="", flush=True)

        # Generate streaming response
        for chunk in chat.send_message_stream(user_input):

            if chunk.text:
                print(chunk.text, end="", flush=True)

        print("\n")

    except Exception as e:
        print("Error:", e)