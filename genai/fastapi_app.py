import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai


# =====================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")


# =====================================
# 2. GEMINI CLIENT
# =====================================

client = genai.Client(
    api_key=api_key,
    http_options={"api_version": "v1beta"}
)

MODEL_NAME = "gemini-2.5-flash"


# =====================================
# 3. FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Gemini AI Backend",
    description="AI chatbot API powered by Gemini",
    version="1.0.0"
)


# =====================================
# 4. CORS CONFIGURATION
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# 5. REQUEST AND RESPONSE MODELS
# =====================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# =====================================
# 6. HOME ENDPOINT
# =====================================

@app.get("/")
def home():
    return {
        "message": "Gemini AI Backend is running!"
    }


# =====================================
# 7. NORMAL CHAT ENDPOINT
# =====================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=request.message
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return {
            "reply": response.text
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Gemini API error:", error)

        raise HTTPException(
            status_code=502,
            detail="Unable to generate an AI response."
        )


# =====================================
# 8. STREAMING CHAT ENDPOINT
# =====================================

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    def generate():

        try:

            response = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=request.message
            )

            for chunk in response:

                if chunk.text:
                    yield chunk.text

        except Exception as error:

            print("Gemini streaming error:", error)

            yield "\n[Error: Unable to generate response.]"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )