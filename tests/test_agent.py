"""
Tests for SupportGenie.

These tests cover:
- Knowledge base loading
- RAG retrieval (cosine similarity ranking)
- Ticket tool creation
- Agent fallback mode (no LLM key required)
- FastAPI endpoints
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Tool tests ──────────────────────────────────────────────────────────────

class TestCreateTicket:
    def test_returns_dict_with_required_fields(self):
        from app.tools import create_ticket

        ticket = create_ticket(
            title="Login broken",
            severity="high",
            summary="Users cannot log in via SSO.",
        )
        assert "ticket_id" in ticket
        assert ticket["title"] == "Login broken"
        assert ticket["severity"] == "high"
        assert ticket["summary"] == "Users cannot log in via SSO."

    def test_ticket_id_format(self):
        from app.tools import create_ticket

        ticket = create_ticket("Test", "low", "desc")
        assert ticket["ticket_id"].startswith("T-")
        assert len(ticket["ticket_id"]) == 8  # T- + 6 hex chars

    def test_unique_ticket_ids(self):
        from app.tools import create_ticket

        ids = {create_ticket("T", "low", "d")["ticket_id"] for _ in range(20)}
        assert len(ids) == 20  # all unique


# ── Knowledge base tests ────────────────────────────────────────────────────

class TestLoadKB:
    def test_loads_all_entries(self):
        from app.rag import load_kb

        docs = load_kb()
        assert len(docs) >= 15  # 15 entries (faq_01 through faq_15)

    def test_document_schema(self):
        from app.rag import load_kb

        docs = load_kb()
        for doc in docs:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc
            assert doc["id"].startswith("faq_")

    def test_known_entry_exists(self):
        from app.rag import load_kb

        docs = load_kb()
        ids = {d["id"] for d in docs}
        for required_id in ("faq_01", "faq_02", "faq_03", "faq_04", "faq_05", "faq_06", "faq_07"):
            assert required_id in ids


# ── RAG retrieval tests ─────────────────────────────────────────────────────

class TestRetrieval:
    def test_returns_top_k_results(self):
        from app.rag import retrieve

        results = retrieve("password reset", top_k=3)
        assert len(results) == 3

    def test_each_result_has_doc_and_score(self):
        from app.rag import retrieve

        results = retrieve("SSO login issues", top_k=2)
        for doc, score in results:
            assert isinstance(doc, dict)
            assert "id" in doc
            assert isinstance(score, float)

    def test_scores_are_descending(self):
        from app.rag import retrieve

        results = retrieve("billing subscription pricing", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_password_query_returns_faq01_in_top3(self):
        from app.rag import retrieve

        results = retrieve("how do I reset my password if I lost my phone", top_k=3)
        top_ids = [doc["id"] for doc, _ in results]
        assert "faq_01" in top_ids

    def test_sso_query_returns_faq04_in_top3(self):
        from app.rag import retrieve

        results = retrieve("SSO not working for new employees", top_k=3)
        top_ids = [doc["id"] for doc, _ in results]
        assert "faq_04" in top_ids

    def test_billing_query_returns_faq03_in_top3(self):
        from app.rag import retrieve

        results = retrieve("what are your pricing options", top_k=3)
        top_ids = [doc["id"] for doc, _ in results]
        assert "faq_03" in top_ids

    def test_format_context_contains_ids(self):
        from app.rag import format_context, retrieve

        results = retrieve("API rate limits", top_k=3)
        context = format_context(results)
        for doc, _ in results:
            assert doc["id"] in context


# ── Agent fallback tests (no LLM key needed) ────────────────────────────────

class TestAgentFallback:
    def setup_method(self):
        """Ensure no API key is set so fallback path is used."""
        os.environ.pop("OPENAI_API_KEY", None)

    def test_chat_returns_required_keys(self):
        from app.agent import chat

        result = chat("How do I reset my password?")
        assert "answer" in result
        assert "citations" in result
        assert "ticket" in result
        assert "sources" in result

    def test_password_question_returns_answer(self):
        from app.agent import chat

        result = chat("How do I reset my password if I lost my phone?")
        assert len(result["answer"]) > 0

    def test_ticket_creation_on_request(self):
        from app.agent import chat

        result = chat("Open a ticket for SSO issue with severity high")
        assert result["ticket"] is not None
        ticket = result["ticket"]
        assert ticket["ticket_id"].startswith("T-")
        assert ticket["severity"] == "high"

    def test_ticket_has_correct_severity(self):
        from app.agent import chat

        for sev in ("low", "medium", "high", "critical"):
            result = chat(f"Report an issue with {sev} severity")
            assert result["ticket"] is not None
            assert result["ticket"]["severity"] == sev

    def test_sources_returned_with_answer(self):
        from app.agent import chat

        result = chat("What are the API rate limits?")
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0

    def test_chat_with_history(self):
        from app.agent import chat

        history = [
            {"role": "user", "content": "What are pricing plans?"},
            {"role": "assistant", "content": "Basic $10, Pro $25 [faq_03]."},
        ]
        result = chat("Can you open a ticket for a billing issue?", history=history)
        assert result["ticket"] is not None


# ── Ticket intent detection tests ───────────────────────────────────────────

class TestTicketIntentDetection:
    def test_open_ticket_detected(self):
        from app.agent import _is_ticket_request

        assert _is_ticket_request("open a ticket for this issue") is True

    def test_report_issue_detected(self):
        from app.agent import _is_ticket_request

        assert _is_ticket_request("I need to report an issue") is True

    def test_create_ticket_detected(self):
        from app.agent import _is_ticket_request

        assert _is_ticket_request("create a ticket please") is True

    def test_regular_question_not_detected(self):
        from app.agent import _is_ticket_request

        assert _is_ticket_request("How do I reset my password?") is False

    def test_pricing_question_not_detected(self):
        from app.agent import _is_ticket_request

        assert _is_ticket_request("What are your pricing options?") is False


# ── FastAPI endpoint tests ───────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


class TestAPIEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "SupportGenie"

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "SupportGenie" in response.text

    def test_chat_endpoint_basic(self, client):
        response = client.post(
            "/chat",
            json={"message": "How do I reset my password?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_chat_endpoint_with_history(self, client):
        response = client.post(
            "/chat",
            json={
                "message": "Tell me more about SSO",
                "history": [
                    {"role": "user", "content": "What is SSO?"},
                    {"role": "assistant", "content": "SSO stands for Single Sign-On [faq_04]."},
                ],
            },
        )
        assert response.status_code == 200

    def test_chat_ticket_creation(self, client):
        response = client.post(
            "/chat",
            json={"message": "Open a ticket for the SSO issue, severity high"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ticket"] is not None
        assert data["ticket"]["ticket_id"].startswith("T-")

    def test_sources_have_required_fields(self, client):
        response = client.post(
            "/chat",
            json={"message": "API rate limits"},
        )
        assert response.status_code == 200
        data = response.json()
        for source in data["sources"]:
            assert "id" in source
            assert "title" in source
            assert "score" in source
