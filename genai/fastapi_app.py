import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import certifi
import jwt
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google import genai
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel
from pwdlib import PasswordHash
from pymongo import MongoClient
from pymongo.errors import PyMongoError, DuplicateKeyError


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
MODEL_NAME = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not MONGO_URI:
    raise ValueError("MONGODB_URI not found in .env")

if not JWT_SECRET_KEY or len(JWT_SECRET_KEY) < 32:
    raise ValueError("JWT_SECRET_KEY must contain at least 32 characters.")


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1beta"},
)


# ============================================================
# 3. MONGODB CLIENT
# ============================================================

mongo_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=10000,
)

database = mongo_client["gemini_chatbot"]
conversations_collection = database["conversations"]
users_collection = database["users"]


# ============================================================
# 4. PASSWORD HASHING AND JWT AUTHENTICATION
# ============================================================

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        user_id = payload.get("sub")
        if not isinstance(user_id, str) or not user_id:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception

    try:
        user = users_collection.find_one({"user_id": user_id})
    except PyMongoError as error:
        print("User lookup error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable.",
        )

    if not user:
        raise credentials_exception

    return user


# ============================================================
# 5. FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mongo_client.admin.command("ping")
        users_collection.create_index("email", unique=True)
        conversations_collection.create_index(
            "conversation_id",
            unique=True,
        )
        conversations_collection.create_index(
            [("user_id", 1), ("created_at", -1)]
        )
        print("MongoDB connected successfully.")
        yield
    except Exception as error:
        print("Application startup error:", repr(error))
        raise
    finally:
        mongo_client.close()
        print("MongoDB connection closed.")


app = FastAPI(
    title="Gemini AI Backend",
    description="Gemini chatbot with MongoDB memory and JWT authentication",
    version="4.0.0",
    lifespan=lifespan,
)


# ============================================================
# 6. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://gemini-ai-chat-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 7. REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class ConversationCreate(BaseModel):
    title: str = "New Chat"


class MessageRequest(BaseModel):
    message: str


class UserRegister(BaseModel):
    name: str
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def get_collection():
    try:
        mongo_client.admin.command("ping")
        return conversations_collection
    except PyMongoError as error:
        print("MongoDB error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable.",
        )


def serialize_conversation(conversation: dict) -> dict:
    # Return a copy so serialization does not mutate the MongoDB result.
    result = dict(conversation)
    result.pop("_id", None)

    if isinstance(result.get("created_at"), datetime):
        result["created_at"] = result["created_at"].isoformat()

    serialized_messages = []
    for message in result.get("messages", []):
        item = dict(message)
        if isinstance(item.get("created_at"), datetime):
            item["created_at"] = item["created_at"].isoformat()
        serialized_messages.append(item)

    result["messages"] = serialized_messages
    return result


def generate_title(message: str) -> str:
    title = " ".join(message.strip().split())
    if len(title) > 40:
        title = title[:37] + "..."
    return title or "New Chat"


# ============================================================
# 9. HOME AND DATABASE CHECK
# ============================================================

@app.get("/")
def home():
    return {"message": "Gemini AI Backend is running!"}


@app.get("/db-check")
def db_check():
    get_collection()
    return {"database": "Connected successfully"}


# ============================================================
# 10. USER REGISTRATION
# ============================================================

@app.post("/auth/register")
def register_user(data: UserRegister):
    name = data.name.strip()
    email = data.email.strip().lower()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name cannot be empty.",
        )

    if len(data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters.",
        )

    if len(data.password) > 128:
        raise HTTPException(
            status_code=400,
            detail="Password is too long.",
        )

    try:
        validate_email(email)
    except EmailNotValidError:
        raise HTTPException(
            status_code=400,
            detail="Invalid email address.",
        )

    user = {
        "user_id": str(uuid4()),
        "name": name,
        "email": email,
        "password": password_hash.hash(data.password),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        users_collection.insert_one(user)
        return {
            "message": "Account created successfully.",
            "user_id": user["user_id"],
            "name": name,
            "email": email,
        }
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail="Email is already registered.",
        )
    except PyMongoError as error:
        print("Registration error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Unable to create account.",
        )


# ============================================================
# 11. USER LOGIN
# ============================================================

@app.post("/auth/login", response_model=TokenResponse)
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    email = form_data.username.strip().lower()

    try:
        user = users_collection.find_one({"email": email})
    except PyMongoError as error:
        print("Login database error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable.",
        )

    if not user or not password_hash.verify(
        form_data.password,
        user["password"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": user["user_id"]})
    return {"access_token": token, "token_type": "bearer"}


# ============================================================
# 12. GET CURRENT USER
# ============================================================

@app.get("/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["user_id"],
        "name": current_user["name"],
        "email": current_user["email"],
    }


# ============================================================
# 13. NORMAL GEMINI CHAT (PUBLIC / STATELESS)
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=request.message,
        )
        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )
        return {"reply": response.text}
    except HTTPException:
        raise
    except Exception as error:
        print("Gemini error:", repr(error))
        raise HTTPException(
            status_code=502,
            detail="Unable to generate AI response.",
        )


# ============================================================
# 14. STREAMING CHAT (PUBLIC / STATELESS)
# ============================================================

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    def generate():
        try:
            response = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=request.message,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as error:
            print("Streaming error:", repr(error))
            yield "\n[Error: Unable to generate response.]"

    return StreamingResponse(generate(), media_type="text/plain")


# ============================================================
# 15. CREATE CONVERSATION (AUTHENTICATED)
# ============================================================

@app.post("/conversations")
def create_conversation(
    data: ConversationCreate | None = None,
    current_user: dict = Depends(get_current_user),
):
    collection = get_collection()
    conversation_id = str(uuid4())
    requested_title = data.title.strip() if data else "New Chat"

    conversation = {
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"],
        "title": requested_title or "New Chat",
        "messages": [],
        "created_at": datetime.now(timezone.utc),
    }

    try:
        collection.insert_one(conversation)
        return serialize_conversation(conversation)
    except PyMongoError as error:
        print("Create conversation error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Unable to create conversation.",
        )


# ============================================================
# 16. SEND MESSAGE WITH MEMORY (OWNER ONLY)
# ============================================================

@app.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: str,
    request: MessageRequest,
    current_user: dict = Depends(get_current_user),
):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    collection = get_collection()
    owner_filter = {
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"],
    }

    try:
        conversation = collection.find_one(owner_filter)

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        previous_messages = conversation.get("messages", [])
        chat_history = []

        for message in previous_messages:
            role = message.get("role")
            if role not in ["user", "assistant"]:
                continue

            chat_history.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": message.get("content", "")}],
            })

        chat_history.append({
            "role": "user",
            "parts": [{"text": request.message}],
        })

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=chat_history,
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )

        ai_reply = response.text
        now = datetime.now(timezone.utc)

        messages_to_save = [
            {
                "role": "user",
                "content": request.message,
                "created_at": now,
            },
            {
                "role": "assistant",
                "content": ai_reply,
                "created_at": datetime.now(timezone.utc),
            },
        ]

        update_fields = {}
        if not previous_messages:
            update_fields["title"] = generate_title(request.message)

        result = collection.update_one(
            owner_filter,
            {
                "$push": {
                    "messages": {"$each": messages_to_save},
                },
                "$set": update_fields,
            },
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Conversation no longer exists.",
            )

        return {
            "conversation_id": conversation_id,
            "user_message": request.message,
            "assistant_reply": ai_reply,
            "memory_enabled": True,
        }

    except HTTPException:
        raise
    except PyMongoError as error:
        print("Message database error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable.",
        )
    except Exception as error:
        print("Message processing error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Unable to process conversation.",
        )


# ============================================================
# 17. GET ONE CONVERSATION (OWNER ONLY)
# ============================================================

@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    collection = get_collection()

    try:
        conversation = collection.find_one(
            {
                "conversation_id": conversation_id,
                "user_id": current_user["user_id"],
            },
            {"_id": 0},
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        return serialize_conversation(conversation)
    except HTTPException:
        raise
    except PyMongoError as error:
        print("Conversation retrieval error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve conversation.",
        )


# ============================================================
# 18. GET ALL USER CONVERSATIONS (OWNER ONLY)
# ============================================================

@app.get("/conversations")
def get_all_conversations(
    current_user: dict = Depends(get_current_user),
):
    collection = get_collection()

    try:
        conversations = list(
            collection.find(
                {"user_id": current_user["user_id"]},
                {
                    "_id": 0,
                    "conversation_id": 1,
                    "title": 1,
                    "created_at": 1,
                },
            ).sort("created_at", -1)
        )

        for conversation in conversations:
            if isinstance(conversation.get("created_at"), datetime):
                conversation["created_at"] = (
                    conversation["created_at"].isoformat()
                )

        return {
            "total": len(conversations),
            "conversations": conversations,
        }
    except PyMongoError as error:
        print("Get conversations error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve conversations.",
        )


# ============================================================
# 19. DELETE CONVERSATION (OWNER ONLY)
# ============================================================

@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    collection = get_collection()

    try:
        result = collection.delete_one({
            "conversation_id": conversation_id,
            "user_id": current_user["user_id"],
        })

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        return {
            "message": "Conversation deleted successfully.",
            "conversation_id": conversation_id,
        }
    except HTTPException:
        raise
    except PyMongoError as error:
        print("Delete conversation error:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Unable to delete conversation.",
        )


# ============================================================
# 20. SERVER RUNNER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )