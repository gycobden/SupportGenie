"""
Agent logic for SupportGenie.

Orchestrates RAG retrieval, tool calling, and LLM response generation.

Supported LLM providers (in priority order):
  1. OpenAI (or any OpenAI-compatible endpoint):
       OPENAI_API_KEY=sk-...
       OPENAI_BASE_URL=https://api.openai.com/v1   (optional override)
       OPENAI_MODEL=gpt-4o-mini                    (optional override)
  2. Google Gemini via its OpenAI-compatible endpoint:
       GEMINI_API_KEY=AIza...
       GEMINI_MODEL=gemini-2.0-flash               (optional override)

If neither key is set, the agent falls back to a rule-based RAG-only mode
that requires no external API and still demonstrates retrieval and ticket creation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from app.rag import format_context, retrieve
from app.tools import create_ticket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SupportGenie, an AI support assistant.
- Retrieve answers from the knowledge base and cite document IDs in [faq_XX] format.
- If the user says "open ticket", "create ticket", "report issue", or similar, call the tool `create_ticket` with title, severity, and summary.
- Be concise, professional, and avoid hallucinations.
- If the answer is not in the knowledge base, say: "That information isn't available in the knowledge base."
- Always include citations like [faq_01] when referencing knowledge base articles."""

# OpenAI tool schema
_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a structured support ticket when a user reports an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title describing the issue.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Severity level of the issue.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Detailed description of the issue.",
                    },
                },
                "required": ["title", "severity", "summary"],
            },
        },
    }
]


_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def _resolve_provider() -> tuple[str, str, str] | None:
    """
    Determine which LLM provider to use based on available environment variables.

    Returns (api_key, base_url, model) for the first configured provider, or
    None if no provider is configured.

    Priority:
      1. OPENAI_API_KEY  — OpenAI or any OpenAI-compatible endpoint.
      2. GEMINI_API_KEY  — Google Gemini via its OpenAI-compatible REST endpoint.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return openai_key, base_url, model

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)
        return gemini_key, _GEMINI_BASE_URL, model

    return None


def _get_openai_client():
    """Lazily create an OpenAI client for the resolved provider."""
    from openai import OpenAI  # type: ignore

    provider = _resolve_provider()
    if provider is None:
        raise RuntimeError("No LLM API key is configured.")
    api_key, base_url, _model = provider
    return OpenAI(api_key=api_key, base_url=base_url)


def _llm_chat(messages: List[Dict[str, Any]], use_tools: bool = True) -> Dict[str, Any]:
    """
    Call the LLM and return a dict with keys:
      - 'content': str  (the text reply, may be None if a tool was called)
      - 'tool_call': dict | None  (name + arguments if a tool was invoked)
    """
    provider = _resolve_provider()
    if provider is None:
        raise RuntimeError("No LLM API key is configured.")
    _api_key, _base_url, model = provider
    client = _get_openai_client()

    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if use_tools:
        kwargs["tools"] = _TOOLS_SCHEMA
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    tool_call = None
    if message.tool_calls:
        tc = message.tool_calls[0]
        tool_call = {
            "name": tc.function.name,
            "arguments": json.loads(tc.function.arguments),
        }

    return {"content": message.content, "tool_call": tool_call}


def _is_ticket_request(text: str) -> bool:
    """Heuristic: detect ticket-creation intent in the user message."""
    lower = text.lower()
    # Action keywords that indicate intent to create/submit something
    action_patterns = [
        r"\bopen\b",
        r"\bcreate\b",
        r"\bfile\b",
        r"\bsubmit\b",
        r"\blog\b",
        r"\braise\b",
        r"\breport\b",
    ]
    # Object keywords — a ticket, bug, or issue is being referenced
    object_patterns = [
        r"\bticket\b",
        r"\bbug\b",
        r"\bissue\b",
    ]
    has_action = any(re.search(p, lower) for p in action_patterns)
    has_object = any(re.search(p, lower) for p in object_patterns)
    return has_action and has_object


def chat(user_message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """
    Main entry point for the agent.

    Args:
        user_message: The latest user message.
        history: Optional list of prior {role, content} turns.

    Returns:
        A dict with:
          - 'answer': str  — The assistant's reply text.
          - 'citations': list[str]  — Document IDs cited.
          - 'ticket': dict | None  — Ticket data if one was created.
          - 'sources': list[dict]  — Retrieved KB snippets.
    """
    history = history or []

    # 1. RAG retrieval
    retrieved = retrieve(user_message, top_k=3)
    context = format_context(retrieved)
    sources = [
        {"id": doc["id"], "title": doc["title"], "score": round(score, 3)}
        for doc, score in retrieved
    ]

    # 2. Build message list
    rag_system = (
        SYSTEM_PROMPT
        + "\n\nRelevant knowledge base articles:\n"
        + context
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": rag_system}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # 3. Call LLM
    provider = _resolve_provider()
    ticket = None
    answer = ""

    if provider is not None:
        try:
            result = _llm_chat(messages, use_tools=True)

            if result["tool_call"] and result["tool_call"]["name"] == "create_ticket":
                args = result["tool_call"]["arguments"]
                ticket = create_ticket(
                    title=args.get("title", "Support Issue"),
                    severity=args.get("severity", "medium"),
                    summary=args.get("summary", user_message),
                )
                # Second LLM call to get the final textual response
                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "content": json.dumps(ticket),
                }
                messages.append(
                    {
                        "role": "assistant",
                        "content": result["content"],
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "create_ticket",
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }
                )
                messages.append(tool_result_msg)
                follow_up = _llm_chat(messages, use_tools=False)
                answer = follow_up["content"] or ""
            else:
                answer = result["content"] or ""

        except Exception as exc:
            logger.warning("LLM call failed: %s — falling back to RAG-only reply.", exc)
            answer, ticket = _fallback_reply(user_message, retrieved)
    else:
        # No API key — use built-in fallback
        answer, ticket = _fallback_reply(user_message, retrieved)

    # Extract citations from answer text; deduplicate while preserving insertion order
    seen: set[str] = set()
    citations: list[str] = []
    for c in re.findall(r"\[faq_\d+\]", answer):
        if c not in seen:
            seen.add(c)
            citations.append(c)

    return {
        "answer": answer,
        "citations": citations,
        "ticket": ticket,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Fallback: rule-based response when no LLM API key is available
# ---------------------------------------------------------------------------

def _fallback_reply(
    user_message: str,
    retrieved: list,
) -> tuple[str, dict | None]:
    """
    Generate a response without an LLM using the retrieved KB snippets directly.
    Also handles ticket creation via heuristics.
    """
    ticket = None

    if _is_ticket_request(user_message):
        # Extract severity hint from message
        severity = "medium"
        for level in ("critical", "high", "medium", "low"):
            if level in user_message.lower():
                severity = level
                break

        # Build a title from the most relevant KB entry
        top_doc = retrieved[0][0] if retrieved else {"title": "Support Issue"}
        ticket = create_ticket(
            title=f"Issue: {top_doc['title']}",
            severity=severity,
            summary=user_message,
        )
        top_ids = [doc["id"] for doc, _ in retrieved[:3]]
        citations_str = " ".join(f"[{i}]" for i in top_ids)
        answer = (
            f"I've created a support ticket for you. "
            f"Here is a summary of relevant knowledge base articles: {citations_str}.\n\n"
            f"Ticket details are shown below."
        )
        return answer, ticket

    # Compose answer from retrieved snippets
    if not retrieved or retrieved[0][1] < 0.05:
        return "That information isn't available in the knowledge base.", None

    lines = []
    for doc, score in retrieved:
        if score < 0.03:
            break
        lines.append(f"**{doc['title']}** [{doc['id']}]\n{doc['content']}")

    answer = "\n\n".join(lines)
    return answer, None
