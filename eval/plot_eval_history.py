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
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval.plot_eval_history")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.style.use("ggplot")


@dataclass
class RunSummary:
    timestamp: datetime
    git_sha: Optional[str]
    overall_score: float
    passed: bool
    cases: List[Dict[str, Any]]


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
        cases = r.get("cases", [])
        runs.append(
            RunSummary(
                timestamp=ts,
                git_sha=r.get("git_sha"),
                overall_score=overall_score,
                passed=passed,
                cases=cases,
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


def plot_metrics(runs: List[RunSummary], out_path: Path) -> None:
    """Plot individual metric scores over time."""
    if not runs:
        raise ValueError("No runs found in history; nothing to plot.")

    # Collect all unique metric names across all runs
    metric_names: set[str] = set()
    for run in runs:
        for case in run.cases:
            metrics = case.get("metrics", [])
            for metric in metrics:
                metric_names.add(metric.get("metric_name", "unknown"))

    if not metric_names:
        # No metrics to plot
        return

    # Organize data by metric
    metric_data: Dict[str, List[tuple[datetime, float]]] = {
        name: [] for name in sorted(metric_names)
    }

    for run in runs:
        for case in run.cases:
            metrics = case.get("metrics", [])
            for metric in metrics:
                metric_name = metric.get("metric_name", "unknown")
                # Use overall_score (0-100) instead of score (0-1)
                score = float(metric.get("overall_score", metric.get("score", 0.0)))
                if metric_name in metric_data:
                    metric_data[metric_name].append((run.timestamp, score))

    # Create subplots for each metric
    num_metrics = len(metric_data)
    if num_metrics == 0:
        return

    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]

    for idx, (metric_name, data_points) in enumerate(sorted(metric_data.items())):
        if not data_points:
            continue

        ax = axes[idx]
        data_points.sort(key=lambda x: x[0])  # Sort by timestamp
        xs = [dp[0] for dp in data_points]
        ys = [dp[1] for dp in data_points]

        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=metric_name)
        ax.set_ylabel("Score (0-100)")
        ax.set_title(f"Metric: {metric_name}")
        ax.set_ylim(0.0, 105.0)  # 0-100 scale for metric scores
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()

    axes[-1].set_xlabel("Run timestamp (UTC)")
    plt.tight_layout()

    # Save to a separate file for metrics
    metrics_out_path = out_path.parent / f"{out_path.stem}_metrics{out_path.suffix}"
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(metrics_out_path)
    plt.close()


def plot_case_scores(runs: List[RunSummary], out_path: Path) -> None:
    """Plot individual case scores over time."""
    if not runs:
        raise ValueError("No runs found in history; nothing to plot.")

    # Collect all unique case IDs
    case_ids: set[str] = set()
    for run in runs:
        for case in run.cases:
            case_ids.add(case.get("id", "unknown"))

    if not case_ids:
        return

    # Organize data by case
    case_data: Dict[str, List[tuple[datetime, float]]] = {
        case_id: [] for case_id in sorted(case_ids)
    }

    for run in runs:
        for case in run.cases:
            case_id = case.get("id", "unknown")
            score = float(case.get("score", 0.0))
            if case_id in case_data:
                case_data[case_id].append((run.timestamp, score))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for case_id, data_points in sorted(case_data.items()):
        if not data_points:
            continue
        data_points.sort(key=lambda x: x[0])
        xs = [dp[0] for dp in data_points]
        ys = [dp[1] for dp in data_points]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=case_id)

    ax.set_xlabel("Run timestamp (UTC)")
    ax.set_ylabel("Case Score")
    ax.set_title("Individual Case Scores Over Time")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save to a separate file for cases
    cases_out_path = out_path.parent / f"{out_path.stem}_cases{out_path.suffix}"
    cases_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cases_out_path)
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

    # Plot metrics if available
    try:
        plot_metrics(runs, out_path)
        logger.info("Metrics plot generated")
    except Exception as e:
        logger.warning(f"Could not generate metrics plot: {e}")

    # Plot individual case scores
    try:
        plot_case_scores(runs, out_path)
        logger.info("Case scores plot generated")
    except Exception as e:
        logger.warning(f"Could not generate case scores plot: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
