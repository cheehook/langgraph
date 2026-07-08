"""Run a LangGraph graph against any OpenAI-compatible endpoint.

The tester supplies the endpoint at run time via environment variables; the
harness just calls it (it does not start or stop any server).

  # OpenAI (or any OpenAI-compatible cloud provider)
  LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=sk-... LLM_MODEL=gpt-4o-mini \
      pytest tests/test_endpoint_graph.py -v

  # A local server the tester already started (ollama / vllm / sglang ...)
  LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2 \
      pytest tests/test_endpoint_graph.py -v

With no LLM_BASE_URL set, these tests are skipped (default suite stays offline).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState

from tests.endpoint_harness import Endpoint, resolve_endpoint

pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_BASE_URL"),
    reason="no LLM endpoint configured (set LLM_BASE_URL to an OpenAI-compatible URL)",
)


class _EndpointChatModel(BaseChatModel):
    """Minimal BaseChatModel that forwards to an OpenAI-compatible Endpoint."""

    endpoint: Endpoint

    model_config = {"arbitrary_types_allowed": True}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        payload = [
            {"role": role_map.get(m.type, "user"), "content": str(m.content)}
            for m in messages
        ]
        text = self.endpoint.chat(payload)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    @property
    def _llm_type(self) -> str:
        return "endpoint-chat"


@pytest.fixture(scope="session")
def endpoint() -> Endpoint:
    """Resolve the OpenAI-compatible endpoint from LLM_* env vars."""
    return resolve_endpoint()


def _build_graph(model: BaseChatModel):
    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    return (
        StateGraph(MessagesState)
        .add_node("call_model", call_model)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )


def test_direct_endpoint_call(endpoint: Endpoint) -> None:
    """The endpoint itself answers (sanity check, bypasses the graph)."""
    reply = endpoint.chat([{"role": "user", "content": "Reply with the word: ready"}])
    assert isinstance(reply, str) and len(reply) > 0
    print(f"\n[endpoint] {endpoint.base_url} ({endpoint.model}) -> {reply!r}")


def test_graph_invoke_against_endpoint(endpoint: Endpoint) -> None:
    """A LangGraph graph whose node calls the real endpoint."""
    graph = _build_graph(_EndpointChatModel(endpoint=endpoint))
    out = graph.invoke({"messages": "Name one primary color."})
    last = out["messages"][-1]
    assert isinstance(last, AIMessage)
    assert isinstance(last.content, str) and len(last.content) > 0
    print(f"\n[graph] answer -> {last.content!r}")
