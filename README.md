# SupportGenie

**SupportGenie** is a demo-ready, AI-powered product support agent that:

- **Answers support questions** from a structured knowledge base using RAG (Retrieval-Augmented Generation).
- **Cites sources** — every answer includes `[faq_XX]` references to the knowledge base entries used.
- **Creates structured support tickets** when users report issues (via tool use).
- **Clean chat UI** — a modern, dark-mode single-page interface served directly from the app.

---

## Architecture

```
SupportGenie/
├── app/
│   ├── main.py          # FastAPI application and REST endpoints
│   ├── agent.py         # Agent logic: RAG + tool calling + LLM integration
│   ├── rag.py           # RAG engine (TF-IDF by default; upgrades to dense embeddings if available)
│   ├── tools.py         # create_ticket tool
│   └── templates/
│       └── index.html   # Single-page chat UI
├── data/
│   └── kb_seed/
│       └── support_kb.json   # 15-entry knowledge base
├── tests/
│   └── test_agent.py    # 30 unit + integration tests
├── requirements.txt
└── README.md
```

### RAG Engine

- **Default (no internet needed):** TF-IDF + cosine similarity via `scikit-learn`. Works offline.
- **Optional upgrade:** Set `SENTENCE_TRANSFORMERS_MODEL` and pre-cache the model for dense semantic embeddings.
- Retrieves top-3 most relevant KB snippets per query.

### LLM Integration

The agent supports **OpenAI**, **Google Gemini**, and any other OpenAI-compatible API:

| Provider | Environment Variables |
|----------|-----------------------|
| OpenAI   | `OPENAI_API_KEY` (default model: `gpt-4o-mini`) |
| Gemini   | `GEMINI_API_KEY` (default model: `gemini-2.0-flash`) |
| Groq     | `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://api.groq.com/openai/v1` |
| Ollama   | `OPENAI_API_KEY=ollama` + `OPENAI_BASE_URL=http://localhost:11434/v1` |
| Any other OpenAI-compatible provider | `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL` |

Provider selection priority: `OPENAI_API_KEY` > `GEMINI_API_KEY` > built-in RAG-only fallback.

If no API key is set, the agent uses a built-in **RAG-only fallback** that still demonstrates full retrieval and ticket creation.

---

## Setup

### Prerequisites

- Python 3.10+

### Install

```bash
# Clone the repo
git clone https://github.com/gycobden/SupportGenie.git
cd SupportGenie

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure (optional)

Copy `.env.example` or create a `.env` file:

```env
# Option 1: OpenAI
OPENAI_API_KEY=sk-...

# Option 2: Google Gemini
# GEMINI_API_KEY=AIza...

# Option 3: Groq (OpenAI-compatible)
# OPENAI_API_KEY=gsk_...
# OPENAI_BASE_URL=https://api.groq.com/openai/v1
# OPENAI_MODEL=llama3-8b-8192

# Option 4: Override model for any provider
# OPENAI_MODEL=gpt-4o
# GEMINI_MODEL=gemini-1.5-pro
```

Provider priority: `OPENAI_API_KEY` > `GEMINI_API_KEY` > offline fallback.

The app loads `.env` automatically via `python-dotenv`.

### Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## API

| Method | Path     | Description                  |
|--------|----------|------------------------------|
| `GET`  | `/`      | Serve the chat UI            |
| `POST` | `/chat`  | Send a message to the agent  |
| `GET`  | `/health`| Health check                 |

### Example `/chat` request

```json
POST /chat
{
  "message": "How do I reset my password if I lost my phone?",
  "history": []
}
```

### Example `/chat` response

```json
{
  "answer": "You can reset your password from **Settings > Security**. If 2FA is enabled, a backup code or registered device is required [faq_01].",
  "citations": ["[faq_01]"],
  "ticket": null,
  "sources": [
    { "id": "faq_01", "title": "Password Reset", "score": 0.842 },
    { "id": "faq_11", "title": "Two-Factor Authentication Setup", "score": 0.501 },
    { "id": "faq_04", "title": "SSO Login Issues", "score": 0.312 }
  ]
}
```

---

## Tests

```bash
pytest tests/ -v
```

30 tests covering: tool creation, KB loading, RAG retrieval accuracy, agent fallback mode, intent detection, and API endpoints.

---

## What I Focused On

- **Robustness without network access** — TF-IDF retrieval works offline so the demo runs in any environment; sentence-transformers upgrades automatically when a cached model is present.
- **Full tool-use pipeline** — ticket creation works both via LLM function calling and via a heuristic fallback, so the feature is always demonstrated.
- **Clean, impressive UI** — dark-mode chat interface with animated typing indicator, citation badges, ticket cards, and a live ticket sidebar panel.
- **Multi-provider LLM support** — the same `OPENAI_API_KEY` / `OPENAI_BASE_URL` pattern works with OpenAI, Groq, Ollama, and any other compatible provider.

## What I'd Improve With One More Hour

- **Streaming responses** — stream LLM tokens to the chat UI for a more responsive feel.
- **Conversation memory** — persist history server-side with Redis so sessions survive page reloads.
- **Richer KB** — add a live ingestion endpoint so admins can upload new articles without redeploying.
- **Confidence threshold** — hide sources below a minimum relevance score to avoid showing misleading snippets.
