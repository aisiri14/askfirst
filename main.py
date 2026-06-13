from dotenv import load_dotenv
load_dotenv()
"""
main.py — FastAPI backend for AskFirst chat app.

LLM priority:  Groq (primary)  →  Gemini (fallback on any error / rate-limit)
Memory:        Universal — past messages from all persistent threads are
               injected as context into the system prompt.
Temp chats:    Handled entirely in-memory; nothing is written to SQLite.
"""

import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (
    init_db, get_db,
    create_thread, get_all_threads, get_thread, delete_thread, rename_thread,
    add_message, get_thread_messages, get_global_memory_summary,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("askfirst")

# ── LLM clients ──────────────────────────────────────────────────────────────
try:
    from groq import Groq
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
except Exception as e:
    log.warning(f"Groq client init failed: {e}")
    _groq_client = None
    GROQ_AVAILABLE = False

try:
    from google import genai as google_genai
    _gemini_key = os.getenv("GEMINI_API_KEY", "")
    GEMINI_AVAILABLE = bool(_gemini_key)
except ImportError:
    try:
        import google.generativeai as _legacy_genai  # fallback to legacy SDK
        _gemini_key = os.getenv("GEMINI_API_KEY", "")
        if _gemini_key:
            _legacy_genai.configure(api_key=_gemini_key)
        GEMINI_AVAILABLE = bool(_gemini_key)
        google_genai = None
    except Exception as e:
        log.warning(f"Gemini client init failed: {e}")
        GEMINI_AVAILABLE = False
        google_genai = None
except Exception as e:
    log.warning(f"Gemini client init failed: {e}")
    GEMINI_AVAILABLE = False
    google_genai = None

GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama3-8b-8192")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("Database initialised.")
    log.info(f"Groq available: {GROQ_AVAILABLE}  |  Gemini available: {GEMINI_AVAILABLE}")
    yield

app = FastAPI(title="AskFirst Chat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
#  LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict], max_tokens: int = 1024) -> tuple[str, str]:
    """Call Groq. Returns (response_text, model_label)."""
    if not GROQ_AVAILABLE or not _groq_client:
        raise RuntimeError("Groq not configured")
    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip(), f"groq/{GROQ_MODEL}"


def _call_gemini(messages: list[dict], max_tokens: int = 1024) -> tuple[str, str]:
    """
    Call Gemini. Supports both new google-genai SDK and legacy google-generativeai.
    Returns (response_text, model_label).
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not configured")

    # Build a single merged prompt (compatible with all Gemini SDKs)
    parts = []
    for m in messages:
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m["role"], m["role"].title())
        parts.append(f"[{label}]: {m['content']}")
    full_prompt = "\n\n".join(parts)

    # Try new google-genai SDK first
    if google_genai is not None:
        try:
            client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=full_prompt)
            return resp.text.strip(), f"gemini/{GEMINI_MODEL}"
        except Exception as e:
            log.warning(f"New Gemini SDK call failed, trying legacy: {e}")

    # Fallback: legacy google-generativeai
    try:
        import google.generativeai as _genai
        _genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        model = _genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(full_prompt)
        return resp.text.strip(), f"gemini/{GEMINI_MODEL}"
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

def call_llm(
    messages: list[dict],
    max_tokens: int = 1024,
    retries: int = 2,
) -> tuple[str, str]:
    """
    Try Groq first; on any failure (rate-limit, network, etc.) fall back to Gemini.
    Returns (text, model_label).
    """
    errors = []

    # ── Groq attempt ──
    if GROQ_AVAILABLE:
        for attempt in range(retries):
            try:
                return _call_groq(messages, max_tokens)
            except Exception as e:
                err_str = str(e).lower()
                errors.append(f"Groq attempt {attempt+1}: {e}")
                log.warning(f"Groq error (attempt {attempt+1}): {e}")
                # Rate-limit: short wait before retry
                if "rate_limit" in err_str or "429" in err_str:
                    time.sleep(2 ** attempt)
                else:
                    break  # Non-retriable error — fall through to Gemini immediately

    # ── Gemini fallback ──
    if GEMINI_AVAILABLE:
        for attempt in range(retries):
            try:
                result = _call_gemini(messages, max_tokens)
                log.info("Fell back to Gemini successfully.")
                return result
            except Exception as e:
                errors.append(f"Gemini attempt {attempt+1}: {e}")
                log.warning(f"Gemini error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

    raise RuntimeError(
        "All LLM providers failed.\n" + "\n".join(errors)
        + "\n\nPlease check your API keys and quotas."
    )


def build_messages(
    thread_history: list[dict],
    global_memory: list[dict],
    user_message: str,
) -> list[dict]:
    """
    Construct the message array to send to the LLM.
    Structure:
      1. System prompt (persona + universal memory context)
      2. Current thread history (excluding the new message)
      3. New user message
    """
    memory_text = ""
    if global_memory:
        lines = []
        for m in global_memory:
            role_label = "User" if m["role"] == "user" else "Assistant"
            snippet = m["content"][:300].replace("\n", " ")
            lines.append(f"  [{role_label}]: {snippet}")
        memory_text = (
            "\n\n--- UNIVERSAL MEMORY (from past threads) ---\n"
            + "\n".join(lines)
            + "\n--- END MEMORY ---"
        )

    system_content = (
        "You are a helpful, knowledgeable AI assistant built by AskFirst. "
        "You maintain continuity across conversations — you have access to a "
        "memory of past interactions with this user."
        + memory_text
        + "\n\nUse this memory naturally when relevant. "
        "Be concise, clear, and friendly."
    )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(thread_history)
    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class ThreadCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)

class ThreadRename(BaseModel):
    title: str = Field(..., max_length=200)

class ChatRequest(BaseModel):
    thread_id: Optional[int] = None          # None → temporary chat
    message: str = Field(..., min_length=1)
    is_temporary: bool = False
    temp_history: list[dict] = Field(default_factory=list)  # for temp chats

class ChatResponse(BaseModel):
    reply: str
    model_used: str
    thread_id: Optional[int]
    is_temporary: bool
    message_id: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Routes — Threads
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/threads")
def list_threads(db: Session = Depends(get_db)):
    threads = get_all_threads(db)
    return [t.to_dict() for t in threads]


@app.post("/threads", status_code=201)
def create_new_thread(body: ThreadCreate, db: Session = Depends(get_db)):
    thread = create_thread(db, title=body.title)
    return thread.to_dict()


@app.get("/threads/{thread_id}")
def get_thread_detail(thread_id: int, db: Session = Depends(get_db)):
    thread = get_thread(db, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread.to_dict()


@app.patch("/threads/{thread_id}/rename")
def rename_thread_endpoint(thread_id: int, body: ThreadRename, db: Session = Depends(get_db)):
    thread = rename_thread(db, thread_id, body.title)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread.to_dict()


@app.delete("/threads/{thread_id}")
def delete_thread_endpoint(thread_id: int, db: Session = Depends(get_db)):
    success = delete_thread(db, thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"detail": "Thread and all its messages deleted successfully."}


@app.get("/threads/{thread_id}/messages")
def get_messages(thread_id: int, db: Session = Depends(get_db)):
    thread = get_thread(db, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    msgs = get_thread_messages(db, thread_id)
    return [m.to_dict() for m in msgs]


# ─────────────────────────────────────────────────────────────────────────────
#  Routes — Chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint.

    Temporary chat (is_temporary=True):
      - `temp_history` carries the in-memory conversation so far
      - Nothing is written to SQLite
      - Global memory is NOT injected (truly ephemeral)

    Persistent chat (is_temporary=False):
      - Auto-creates a thread if thread_id is None
      - Saves user + assistant messages
      - Injects global memory from all other threads
    """
    if body.is_temporary:
        # ── Temporary path ────────────────────────────────────────────────
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in body.temp_history
            if m.get("role") in ("user", "assistant")
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. This is a temporary chat — "
                    "your conversation is private and will not be saved."
                ),
            },
            *history,
            {"role": "user", "content": body.message},
        ]
        try:
            reply, model_used = call_llm(messages)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        return ChatResponse(
            reply=reply,
            model_used=model_used,
            thread_id=None,
            is_temporary=True,
        )

    # ── Persistent path ───────────────────────────────────────────────────
    # 1. Resolve / create thread
    if body.thread_id:
        thread = get_thread(db, body.thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        thread_id = thread.id
    else:
        # Auto-create a thread titled after the first message
        short_title = body.message[:60] + ("…" if len(body.message) > 60 else "")
        thread = create_thread(db, title=short_title)
        thread_id = thread.id

    # 2. Fetch this thread's history
    thread_msgs = get_thread_messages(db, thread_id)
    thread_history = [{"role": m.role, "content": m.content} for m in thread_msgs]

    # 3. Fetch global memory (other threads)
    global_memory = get_global_memory_summary(db, exclude_thread_id=thread_id, limit=40)

    # 4. Build prompt and call LLM
    messages = build_messages(thread_history, global_memory, body.message)
    try:
        reply, model_used = call_llm(messages)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 5. Persist both turns
    add_message(db, thread_id, "user", body.message)
    saved_msg = add_message(db, thread_id, "assistant", reply, model_used=model_used)

    # 6. Auto-rename thread from first user message if still default
    if thread.title in ("New Chat", "") and not thread_msgs:
        rename_thread(db, thread_id, body.message[:60] + ("…" if len(body.message) > 60 else ""))

    return ChatResponse(
        reply=reply,
        model_used=model_used,
        thread_id=thread_id,
        is_temporary=False,
        message_id=saved_msg.id,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_available":   GROQ_AVAILABLE,
        "gemini_available": GEMINI_AVAILABLE,
        "groq_model":       GROQ_MODEL,
        "gemini_model":     GEMINI_MODEL,
    }
