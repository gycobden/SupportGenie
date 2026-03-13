"""
RAG (Retrieval-Augmented Generation) engine for SupportGenie.

Loads the knowledge base, builds a TF-IDF index for retrieval, and optionally
upgrades to dense sentence-transformers embeddings when a locally-cached model
is available.  The TF-IDF path requires no network access and works out of the
box.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_KB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "kb_seed", "support_kb.json"
)

# Lazy-loaded globals
_documents: List[dict] | None = None
_embeddings: np.ndarray | None = None  # normalised row vectors
_vectorizer = None   # TfidfVectorizer (fallback)
_dense_model = None  # SentenceTransformer (optional upgrade)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _try_load_dense_model():
    """
    Attempt to load a sentence-transformers model from the local cache.
    Returns the model on success, or None if it is not available.
    """
    model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
    try:
        import huggingface_hub  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        # Raise immediately if the model is not in the local cache
        huggingface_hub.snapshot_download(
            f"sentence-transformers/{model_name}",
            local_files_only=True,
        )
        model = SentenceTransformer(model_name)
        logger.info("Loaded dense embedding model: %s", model_name)
        return model
    except Exception:
        logger.info(
            "Dense model not available locally — using TF-IDF retrieval instead."
        )
        return None


def _build_tfidf_index(documents: List[dict]) -> Tuple[np.ndarray, object]:
    """Build and return (normalised_matrix, vectorizer) using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    texts = [f"{doc['title']} {doc['content']}" for doc in documents]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    return matrix, vec


def _build_dense_index(documents: List[dict], model) -> np.ndarray:
    """Build and return a normalised dense embedding matrix."""
    texts = [f"{doc['title']}: {doc['content']}" for doc in documents]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / np.where(norms == 0, 1, norms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_kb(kb_path: str = _KB_PATH) -> List[dict]:
    """Load the knowledge base from a JSON file."""
    with open(kb_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_index():
    global _documents, _embeddings, _vectorizer, _dense_model
    if _documents is not None:
        return

    _documents = load_kb()

    dense = _try_load_dense_model()
    if dense is not None:
        _dense_model = dense
        _embeddings = _build_dense_index(_documents, _dense_model)
    else:
        _embeddings, _vectorizer = _build_tfidf_index(_documents)


def retrieve(query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
    """
    Retrieve the top-k most relevant documents for *query*.

    Returns a list of (document, score) tuples sorted by descending relevance.
    """
    _ensure_index()

    if _dense_model is not None:
        q_vec = _dense_model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]
    else:
        q_vec = _vectorizer.transform([query]).toarray()[0].astype(np.float32)

    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm

    scores = _embeddings @ q_vec  # cosine similarity
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(_documents[i], float(scores[i])) for i in top_indices]


def format_context(results: List[Tuple[dict, float]]) -> str:
    """Format retrieved documents into a context string for the LLM prompt."""
    lines = []
    for doc, _score in results:
        lines.append(f"[{doc['id']}] {doc['title']}\n{doc['content']}")
    return "\n\n".join(lines)


def reset_index() -> None:
    """Reset the in-memory index (useful for testing with a custom KB path)."""
    global _documents, _embeddings, _vectorizer, _dense_model
    _documents = None
    _embeddings = None
    _vectorizer = None
    _dense_model = None
