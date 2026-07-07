"""Minimal OpenAI-compatible embedding server using deterministic local embeddings.

No external model download required — embeddings are derived from character
n-gram hashing, producing 384-dimension vectors suitable for basic retrieval.

Usage:
  .venv\\Scripts\\python.exe -m src.utils.embedding_server --port 11435

Then configure LightRAG with:
  EMBEDDING_BINDING=openai
  EMBEDDING_BINDING_HOST=http://127.0.0.1:11435
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Any

# ---------------------------------------------------------------------------
# Deterministic embedding (no external model)
# ---------------------------------------------------------------------------

EMBED_DIM = 384


def _text_hash_vector(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Derive a deterministic embedding vector from text character n-grams."""
    vec = [0.0] * dim
    # Unigrams (char codes)
    for ch in text.encode("utf-8"):
        idx = ch % dim
        vec[idx] += 1.0
    # Bigrams
    for i in range(len(text) - 1):
        bg = text[i:i + 2].encode("utf-8")
        h = int(hashlib.md5(bg).hexdigest()[:8], 16)
        vec[h % dim] += 1.5
    # Trigrams
    for i in range(len(text) - 2):
        tg = text[i:i + 3].encode("utf-8")
        h = int(hashlib.md5(tg).hexdigest()[:8], 16)
        vec[h % dim] += 2.0

    # L2-normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def build_embedding(texts: list[str]) -> list[list[float]]:
    return [_text_hash_vector(t) for t in texts]


def handle_request(environ, start_response):
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    # Health check
    if path in ("/health", "/v1/health") and method == "GET":
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps({"status": "ok"}).encode()]

    # Embeddings endpoint
    if path in ("/v1/embeddings", "/embeddings") and method == "POST":
        try:
            length = int(environ.get("CONTENT_LENGTH", 0))
            body = json.loads(environ["wsgi.input"].read(length))
        except Exception:
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps({"error": "invalid request body"}).encode()]

        texts = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        model_name = body.get("model", "local-embedding")

        try:
            vectors = build_embedding(texts)
        except Exception as exc:
            start_response("500 Internal Server Error", [("Content-Type", "application/json")])
            return [json.dumps({"error": str(exc)}).encode()]

        result = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ],
            "model": model_name,
            "usage": {"prompt_tokens": sum(len(t) for t in texts), "total_tokens": sum(len(t) for t in texts)},
        }
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps(result, ensure_ascii=False).encode()]

    # Model list (LightRAG checks this)
    if path in ("/v1/models", "/models") and method == "GET":
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps({
            "object": "list",
            "data": [{"id": "local-embedding", "object": "model"}],
        }).encode()]

    start_response("404 Not Found", [("Content-Type", "application/json")])
    return [json.dumps({"error": "not found"}).encode()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible embedding server")
    parser.add_argument("--port", type=int, default=11435, help="Port to listen on")
    args = parser.parse_args()

    from wsgiref.simple_server import make_server

    port = args.port
    print(f"[embed-server] Starting on http://127.0.0.1:{port}")
    print(f"[embed-server] Model: {os.getenv('LOCAL_EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')}")
    print(f"[embed-server] Pre-warming model...")

    # Warm up
    build_embedding(["warmup"])

    print(f"[embed-server] Ready. (Ctrl+C to stop)")
    server = make_server("0.0.0.0", port, handle_request)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[embed-server] Shutting down.")


if __name__ == "__main__":
    main()
