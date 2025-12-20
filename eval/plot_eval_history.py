"""
Quick visualisation utilities for evaluation history.

This script reads ``eval_history.json`` (produced by ``eval.update_history``)
and generates comprehensive plots to help you see whether the agent is improving
over time.

Usage (from project root):

    uv run python -m eval.plot_eval_history \
        --history eval_history.json \
        --out eval/img/eval_history.png
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


def plot_comprehensive_evaluation(runs: List[RunSummary], out_path: Path) -> None:
    """
    Generate a comprehensive figure with subplots showing:
    1. Overall score over time
    2. Each metric's trend over time
    3. Individual case scores over time
    """
    if not runs:
        raise ValueError("No runs found in history; nothing to plot.")

    # Collect all unique metric names across all runs
    metric_names: set[str] = set()
    case_ids: set[str] = set()
    for run in runs:
        for case in run.cases:
            case_ids.add(case.get("id", "unknown"))
            metrics = case.get("metrics", [])
            for metric in metrics:
                metric_names.add(metric.get("metric_name", "unknown"))

    # Calculate number of subplots: 1 (overall) + metrics + 1 (cases if exists)
    num_subplots = 1  # Overall score
    if metric_names:
        num_subplots += len(metric_names)
    if case_ids:
        num_subplots += 1  # Cases subplot
    if num_subplots < 1:
        num_subplots = 1  # At least overall score

    # Determine common x-axis range from all runs
    all_timestamps = [r.timestamp for r in runs]
    x_min = min(all_timestamps) if all_timestamps else None
    x_max = max(all_timestamps) if all_timestamps else None

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4 * num_subplots))
    gs = fig.add_gridspec(num_subplots, 1, hspace=0.3)

    subplot_idx = 0

    # 1. Overall score subplot
    ax_overall = fig.add_subplot(gs[subplot_idx, 0])
    xs = [r.timestamp for r in runs]
    ys = [r.overall_score for r in runs]
    colors = ["tab:green" if r.passed else "tab:red" for r in runs]

    ax_overall.plot(
        xs,
        ys,
        color="tab:blue",
        linewidth=2,
        label="Overall Score",
        marker="o",
        markersize=6,
    )
    ax_overall.scatter(
        xs, ys, c=colors, s=60, zorder=3, alpha=0.7, edgecolors="black", linewidth=1
    )
    ax_overall.set_ylim(0.0, 1.05)
    if x_min and x_max:
        ax_overall.set_xlim(x_min, x_max)
    ax_overall.set_ylabel("Overall Score (0-1)", fontsize=11, fontweight="bold")
    ax_overall.set_title(
        "Overall Evaluation Score Over Time", fontsize=12, fontweight="bold"
    )
    ax_overall.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_overall.legend(loc="best", fontsize=10)
    # Hide x-axis tick labels for this subplot (will show on last one)
    ax_overall.tick_params(axis="x", labelbottom=False)
    subplot_idx += 1

    # 2. Individual metrics subplots
    if metric_names:
        # Organize data by metric
        metric_data: Dict[str, List[tuple[datetime, float]]] = {
            name: [] for name in sorted(metric_names)
        }

        for run in runs:
            for case in run.cases:
                metrics = case.get("metrics", [])
                for metric in metrics:
                    metric_name = metric.get("metric_name", "unknown")
                    score = float(metric.get("overall_score", metric.get("score", 0.0)))
                    if metric_name in metric_data:
                        metric_data[metric_name].append((run.timestamp, score))

        for metric_name in sorted(metric_names):
            if metric_name not in metric_data or not metric_data[metric_name]:
                continue

            ax_metric = fig.add_subplot(gs[subplot_idx, 0], sharex=ax_overall)
            data_points = metric_data[metric_name]
            data_points.sort(key=lambda x: x[0])
            xs_metric = [dp[0] for dp in data_points]
            ys_metric = [dp[1] for dp in data_points]

            ax_metric.plot(
                xs_metric,
                ys_metric,
                marker="o",
                linewidth=2,
                markersize=5,
                label=metric_name,
                color="tab:orange",
            )
            ax_metric.set_ylabel("Score (0-100)", fontsize=11, fontweight="bold")
            ax_metric.set_title(
                f"Metric: {metric_name}", fontsize=12, fontweight="bold"
            )
            ax_metric.set_ylim(0.0, 105.0)
            if x_min and x_max:
                ax_metric.set_xlim(x_min, x_max)
            ax_metric.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax_metric.legend(loc="best", fontsize=10)
            # Hide x-axis tick labels for intermediate subplots (will show on last one)
            ax_metric.tick_params(axis="x", labelbottom=False)
            subplot_idx += 1

    # 3. Individual case scores subplot
    if case_ids:
        ax_cases = fig.add_subplot(gs[subplot_idx, 0], sharex=ax_overall)

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

        # Plot each case with different colors
        colors_list = plt.cm.tab10(range(len(case_data)))
        for idx, (case_id, data_points) in enumerate(sorted(case_data.items())):
            if not data_points:
                continue
            data_points.sort(key=lambda x: x[0])
            xs_case = [dp[0] for dp in data_points]
            ys_case = [dp[1] for dp in data_points]
            ax_cases.plot(
                xs_case,
                ys_case,
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=case_id,
                color=colors_list[idx % len(colors_list)],
            )

        ax_cases.set_xlabel("Run timestamp (UTC)", fontsize=11, fontweight="bold")
        ax_cases.set_ylabel("Case Score (0-1)", fontsize=11, fontweight="bold")
        ax_cases.set_title(
            "Individual Case Scores Over Time", fontsize=12, fontweight="bold"
        )
        ax_cases.set_ylim(0.0, 1.05)
        if x_min and x_max:
            ax_cases.set_xlim(x_min, x_max)
        ax_cases.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_cases.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        # Show x-axis tick labels on the last subplot
        ax_cases.tick_params(axis="x", labelbottom=True)

    # X-axis label is already set on the last subplot (cases), so we don't need to set it again

    plt.suptitle(
        "Comprehensive Evaluation History", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comprehensive evaluation plot saved to {out_path}")


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
        default="eval/img/eval_history.png",
        help="Output image path for the plot (default: eval/img/eval_history.png).",
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

    # Generate comprehensive plot with all subplots
    try:
        plot_comprehensive_evaluation(runs, out_path)
        logger.info("Comprehensive evaluation plot generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate comprehensive plot: {e}", exc_info=True)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
