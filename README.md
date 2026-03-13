# SupportGenie

**SupportGenie** is a demo-ready, AI-powered product support agent that:

- **Answers support questions** from a structured knowledge base using RAG (Retrieval-Augmented Generation).
- **Cites sources** — every answer includes `[faq_XX]` references to the knowledge base entries used.
- **Creates structured support tickets** when users report issues (via tool use).
- **Clean chat UI** — a modern, dark-mode single-page interface served directly from the app.

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