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