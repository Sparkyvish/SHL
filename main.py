"""
main.py  –  FastAPI service for the SHL Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply": str, "recommendations": list, "end_of_conversation": bool}

Startup: loads the FAISS index and embedding model into memory.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import chat as agent_chat
from retriever import _load_resources  # preloads model + index at startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty.")
        return v


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def must_end_with_user(cls, msgs: List[Message]) -> List[Message]:
        if msgs[-1].role != "user":
            raise ValueError("The last message must have role='user'.")
        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


# ── Lifespan: warm up retriever on startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up retriever (loading model + FAISS index) …")
    try:
        _load_resources()
        log.info("Retriever ready.")
    except Exception as exc:
        log.error("Retriever failed to load: %s", exc)
        # Don't crash – /health will still return ok; /chat will fail gracefully
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please retry."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Readiness check")
def health() -> Dict[str, str]:
    """Returns 200 OK when the service is up."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational assessment recommendation",
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint.

    The caller sends the FULL conversation history on every request.
    The service returns the next agent reply and, when appropriate, a
    structured shortlist of 1–10 SHL assessment recommendations.
    """
    messages = [m.model_dump() for m in request.messages]

    # Hard guard: honour the 8-turn cap
    if len(messages) > 8:
        return ChatResponse(
            reply=(
                "This conversation has reached the maximum length. "
                "Please start a new session if you need further help."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    try:
        result = agent_chat(messages)
    except Exception as exc:
        log.exception("agent_chat raised: %s", exc)
        raise HTTPException(status_code=500, detail="Agent error. Please retry.")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )
