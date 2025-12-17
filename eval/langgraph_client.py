"""
Minimal HTTP client for talking to the LangGraph server for evaluation.

This intentionally mirrors the HTTP patterns used by the Streamlit UI, but
returns a structured dictionary suitable for automated checks.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_SERVER_URL = os.getenv("LANGGRAPH_SERVER_URL", "http://localhost:2024")
DEFAULT_GRAPH_NAME = os.getenv("LANGGRAPH_GRAPH_NAME", "data_analysis_agent")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LangGraphResult:
    """Normalized result returned from the LangGraph server."""

    answer_text: str
    structured: Dict[str, Any]
    artifacts: List[str]
    raw: Dict[str, Any]
    latency_seconds: float


def _extract_content_text(content: Any) -> str:
    """Extract text from content, handling both string and list formats."""
    if not content:
        return ""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or str(item)
                if text:
                    text_parts.append(str(text))
        return " ".join(text_parts)

    return str(content)


def _try_extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract the first top-level JSON object from the given text.

    This is intentionally simple but robust enough for eval-mode prompts that
    ask the model to return a small JSON object, possibly wrapped in a code
    block.
    """
    if not text:
        return None

    # Fast path: pure JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            return None

    # Scan for the first {...} block
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : idx + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _get_or_create_assistant(server_url: str, graph_name: str) -> str:
    """
    Get or create an assistant for the given graph.

    For evaluation we deliberately ignore any pre-configured assistant ID and
    always work off the graph name. This avoids issues where a hard-coded
    assistant ID (e.g. from local dev) is not known to the ephemeral LangGraph
    server in CI.
    """
    # Try to fetch an existing assistant by graph_id (mirrors Streamlit UI logic)
    try:
        response = requests.get(
            f"{server_url}/assistants/{graph_name}",
            timeout=5,
        )
        if response.status_code == 200:
            assistant = response.json()
            return assistant.get("assistant_id") or graph_name
    except Exception:
        pass

    # If none exists, create one
    try:
        response = requests.post(
            f"{server_url}/assistants",
            json={"graph_id": graph_name},
            timeout=5,
        )
        if response.status_code in (200, 201):
            assistant = response.json()
            return assistant.get("assistant_id") or graph_name
    except Exception:
        pass

    # Fallback: use the graph name (LangGraph will error clearly if invalid)
    return graph_name


def _get_or_create_thread(server_url: str, thread_id: str) -> str:
    """Get or create a thread in LangGraph Server."""
    try:
        response = requests.get(
            f"{server_url}/threads/{thread_id}",
            timeout=5,
        )
        if response.status_code == 200:
            return thread_id
    except Exception:
        pass

    try:
        response = requests.post(
            f"{server_url}/threads",
            json={"thread_id": thread_id},
            timeout=5,
        )
        if response.status_code in (200, 201):
            return thread_id
    except Exception as exc:
        raise RuntimeError(f"Failed to create thread: {exc}") from exc

    return thread_id


def _create_run(
    server_url: str,
    thread_id: str,
    input_data: dict,
    assistant_id: str,
) -> dict:
    """Create a run in LangGraph Server."""
    # Import here to avoid circular imports
    from src.langgraph_server.graph import get_recursion_limit

    recursion_limit = get_recursion_limit()

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": recursion_limit,
    }

    response = requests.post(
        f"{server_url}/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": input_data,
            "config": config,
        },
        timeout=30,
    )

    if response.status_code in (200, 201):
        return response.json()

    error_detail: Any = response.text
    try:
        error_detail = response.json()
    except Exception:
        pass
    raise RuntimeError(f"Failed to create run: {response.status_code} - {error_detail}")


def _stream_final_answer(server_url: str, thread_id: str, run_id: str) -> str:
    """Stream events from a run and return the final assistant text."""
    response = requests.get(
        f"{server_url}/threads/{thread_id}/runs/{run_id}/stream",
        stream=True,
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to stream events: {response.status_code} - {response.text}"
        )

    accumulated_content = ""
    current_event_type: Optional[str] = None

    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode("utf-8")

        if line_str.startswith("event: "):
            current_event_type = line_str[7:].strip()
        elif line_str.startswith("data: "):
            data_str = line_str[6:]
            try:
                event_data = json.loads(data_str)

                if current_event_type == "values" and "messages" in event_data:
                    for msg in event_data["messages"]:
                        if msg.get("type") == "ai":
                            raw_content = msg.get("content", "")
                            content = _extract_content_text(raw_content)
                            if content and content.strip():
                                accumulated_content = content

                current_event_type = None
            except json.JSONDecodeError:
                # Ignore malformed SSE chunks
                continue

    return accumulated_content.strip()


def _find_recent_artifacts(since_timestamp: float) -> list[str]:
    """
    Find any plot artifacts created after the given timestamp.

    By convention, analysis plots are written to the top-level ``img/`` folder
    with filenames like ``plot_*.png``.
    """
    img_dir = PROJECT_ROOT / "img"
    if not img_dir.exists():
        return []

    artifacts: list[str] = []
    for plot_file in sorted(
        img_dir.glob("plot_*.png"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if plot_file.stat().st_mtime >= since_timestamp:
            artifacts.append(str(plot_file.relative_to(PROJECT_ROOT)))
    return artifacts


def query_langgraph(
    query: str,
    *,
    server_url: str | None = None,
    graph_name: str | None = None,
    assistant_id: str | None = None,
    thread_id: str | None = None,
) -> LangGraphResult:
    """
    Send a single query to the LangGraph server and return a normalized result.

    This function is synchronous and designed specifically for evaluation.
    """
    server_url = server_url or DEFAULT_SERVER_URL
    graph_name = graph_name or DEFAULT_GRAPH_NAME
    # For evaluation, always resolve assistant from the current graph name
    # to avoid relying on any pre-existing assistant IDs.
    resolved_assistant_id = _get_or_create_assistant(server_url, graph_name)

    # Use a deterministic thread ID per eval run unless caller overrides
    thread_id = thread_id or str(uuid.uuid4())

    start_time = time.time()
    query_start_time = start_time

    # Ensure thread exists
    effective_thread_id = _get_or_create_thread(server_url, thread_id)

    # Create run
    run_data = _create_run(
        server_url,
        effective_thread_id,
        {"messages": [{"role": "human", "content": query}]},
        resolved_assistant_id,
    )
    run_id = run_data.get("run_id") or run_data.get("id")
    if not run_id:
        raise RuntimeError(f"Failed to get run ID from response: {run_data}")

    # Stream final answer
    answer_text = _stream_final_answer(server_url, effective_thread_id, run_id)
    latency_seconds = time.time() - start_time

    # Attempt to extract structured JSON from the answer (if present)
    structured: Dict[str, Any] = {}
    maybe_json = _try_extract_json_object(answer_text)
    if maybe_json is not None:
        structured = maybe_json

    # Detect artifacts created during this query
    artifacts = _find_recent_artifacts(query_start_time)

    return LangGraphResult(
        answer_text=answer_text,
        structured=structured,
        artifacts=artifacts,
        raw={"run": run_data},
        latency_seconds=latency_seconds,
    )


__all__ = ["LangGraphResult", "query_langgraph"]
