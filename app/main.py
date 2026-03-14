"""
SupportGenie FastAPI application.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables from project .env if present.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from app.agent import chat

app = FastAPI(title="SupportGenie", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatTurn]] = None


class SourceDoc(BaseModel):
    id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    ticket: Optional[Dict[str, Any]] = None
    sources: List[SourceDoc]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page chat UI."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a user message and return the agent's response."""
    try:
        history = (
            [{"role": t.role, "content": t.content} for t in request.history]
            if request.history
            else []
        )
        result = chat(request.message, history=history)
        return ChatResponse(
            answer=result["answer"],
            citations=result["citations"],
            ticket=result["ticket"],
            sources=[SourceDoc(**s) for s in result["sources"]],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "SupportGenie"}
