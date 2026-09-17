"""Endpoint harness: run LangGraph tests against any OpenAI-compatible endpoint.

The tester supplies an already-running endpoint (a cloud provider, or a local
server they started themselves — ollama, vllm, sglang, etc.). This harness only
reads the URL/credentials and calls it; it never starts or stops a server.

Environment variables
---------------------
  LLM_BASE_URL    OpenAI-compatible base URL   (required)
                  e.g. https://api.openai.com/v1  or  http://localhost:11434/v1
  LLM_API_KEY     credential                   (default: "dummy")
  LLM_MODEL       model id/name to request     (default: gpt-3.5-turbo)
  LLM_MAX_TOKENS  generation cap               (default: 32)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class Endpoint:
    """A resolved, ready-to-call OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    max_tokens: int = 32

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Call POST {base_url}/chat/completions and return the reply text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=120) as client:
            r = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


def _wait_until_ready(base_url: str, api_key: str, timeout: float) -> None:
    """Poll the endpoint's /models until it answers or timeout elapses."""
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + timeout
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, headers=headers, timeout=5)
            if r.status_code < 500:
                return
        except Exception as e:  # noqa: BLE001 - server not reachable yet
            last_err = e
        time.sleep(1.0)
    raise TimeoutError(
        f"endpoint {base_url} not reachable after {timeout}s (last error: {last_err})"
    )


def resolve_endpoint() -> Endpoint:
    """Build an Endpoint from LLM_* env vars and verify it is reachable."""
    base_url = os.environ["LLM_BASE_URL"]  # required
    api_key = os.getenv("LLM_API_KEY", "dummy")
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "32"))
    _wait_until_ready(base_url, api_key, timeout=15)
    return Endpoint(base_url, api_key, model, max_tokens)
