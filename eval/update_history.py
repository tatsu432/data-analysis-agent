"""
Utilities for maintaining a long‑term history of evaluation runs.

This script reads the most recent ``eval_report.json`` (single run)
and appends a compact summary into ``eval_history.json`` so that
we can analyse trends over time (e.g. in visualisations).

Usage (from project root):

    uv run python -m eval.update_history \
        --report eval_report.json \
        --history eval_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _summarise_run(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary from a full eval report."""
    metadata = report.get("metadata", {})
    case_results: List[Dict[str, Any]] = report.get("case_results", [])

    summary_cases: List[Dict[str, Any]] = []
    for c in case_results:
        summary_cases.append(
            {
                "id": c.get("id"),
                "score": c.get("score"),
                "latency_seconds": c.get("latency_seconds"),
                "passed": c.get("passed"),
                "critical": c.get("critical"),
            }
        )

    return {
        "timestamp": metadata.get("timestamp"),
        "git_sha": metadata.get("git_sha"),
        "fail_under": metadata.get("fail_under"),
        "cases_dir": metadata.get("cases_dir"),
        "overall_score": report.get("overall_score"),
        "pass": report.get("pass"),
        "num_cases": len(case_results),
        "cases": summary_cases,
    }


def append_to_history(report_path: Path, history_path: Path) -> None:
    """Append a single run (from report_path) into history_path."""
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    report = _load_json(report_path)
    run_summary = _summarise_run(report)

    if history_path.exists():
        history = _load_json(history_path)
        # Backwards / defensive: tolerate both list-of-runs and {"runs": [...]}.
        if isinstance(history, dict):
            runs: List[Dict[str, Any]] = history.setdefault("runs", [])
        else:
            runs = history
            history = {"runs": runs}
    else:
        history = {"runs": []}
        runs = history["runs"]

    runs.append(run_summary)
    _save_json(history_path, history)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append the latest eval report into eval history."
    )
    parser.add_argument(
        "--report",
        type=str,
        default="eval_report.json",
        help="Path to single-run eval report JSON (default: eval_report.json).",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="eval_history.json",
        help="Path to cumulative history JSON (default: eval_history.json).",
    )

    args = parser.parse_args(argv)

    report_path = (PROJECT_ROOT / args.report).resolve()
    history_path = (PROJECT_ROOT / args.history).resolve()

    append_to_history(report_path, history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
