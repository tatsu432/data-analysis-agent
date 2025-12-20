"""
Quick visualisation utilities for evaluation history.

This script reads ``eval_history.json`` (produced by ``eval.update_history``)
and generates simple plots to help you see whether the agent is improving
over time.

Usage (from project root):

    uv run python -m eval.plot_eval_history \
        --history eval_history.json \
        --out img/eval_history.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.style.use("ggplot")


@dataclass
class RunSummary:
    timestamp: datetime
    git_sha: Optional[str]
    overall_score: float
    passed: bool


def _load_history(history_path: Path) -> List[RunSummary]:
    with history_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    runs_raw: List[Dict[str, Any]]
    if isinstance(raw, dict):
        runs_raw = raw.get("runs", [])
    else:
        runs_raw = raw

    runs: List[RunSummary] = []
    for r in runs_raw:
        ts_str = r.get("timestamp")
        if not ts_str:
            # Skip entries without timestamp; they are hard to order on a time axis.
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            continue

        overall_score = float(r.get("overall_score", 0.0))
        passed = bool(r.get("pass", False))
        runs.append(
            RunSummary(
                timestamp=ts,
                git_sha=r.get("git_sha"),
                overall_score=overall_score,
                passed=passed,
            )
        )

    # Sort chronologically just in case entries were appended out of order.
    runs.sort(key=lambda r: r.timestamp)
    return runs


def plot_overall_score(runs: List[RunSummary], out_path: Path) -> None:
    if not runs:
        raise ValueError("No runs found in history; nothing to plot.")

    xs = [r.timestamp for r in runs]
    ys = [r.overall_score for r in runs]
    colors = ["tab:green" if r.passed else "tab:red" for r in runs]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, color="tab:blue", linewidth=1.5, label="overall_score")
    plt.scatter(xs, ys, c=colors, s=40, zorder=3, label="runs")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Run timestamp (UTC)")
    plt.ylabel("Overall score")
    plt.title("Evaluation overall score over time")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot evaluation history over time.")
    parser.add_argument(
        "--history",
        type=str,
        default="eval_history.json",
        help="Path to cumulative history JSON (default: eval_history.json).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="img/eval_history.png",
        help="Output image path for the plot (default: img/eval_history.png).",
    )

    args = parser.parse_args(argv)
    history_path = (PROJECT_ROOT / args.history).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}. "
            "Run eval.update_history at least once to create it."
        )

    runs = _load_history(history_path)
    plot_overall_score(runs, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
