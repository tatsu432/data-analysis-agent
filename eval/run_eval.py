"""
CLI entry point for running LangGraph-based evaluations.

Usage:

    python -m eval.run_eval \
        --cases eval/cases \
        --out eval_report.json \
        --fail-under 0.85
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import yaml

from .langgraph_client import LangGraphResult, query_langgraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    type: str
    passed: bool
    message: str


@dataclass
class CaseResult:
    id: str
    score: float
    latency_seconds: float
    passed: bool
    critical: bool
    error: Optional[str]
    checks: List[CheckResult]


def _load_cases(cases_dir: Path, max_cases: Optional[int] = None) -> List[dict]:
    """Load YAML evaluation cases from a directory."""
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    cases: list[dict] = []
    for path in sorted(list(cases_dir.glob("*.yml")) + list(cases_dir.glob("*.yaml"))):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Case file must contain a mapping: {path}")
        data.setdefault("file", str(path.relative_to(PROJECT_ROOT)))
        cases.append(data)
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases


def _get_git_sha() -> Optional[str]:
    """Return current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _json_path_lookup(obj: Any, path: str) -> Any:
    """
    Very small JSONPath-like lookup supporting dot-separated paths and
    integer list indices, e.g. ``metrics.score`` or ``items.0.value``.
    """
    parts = [p for p in path.split(".") if p]
    current: Any = obj
    for part in parts:
        if isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                raise KeyError(f"Expected integer index for list, got '{part}'")
            try:
                current = current[idx]
            except IndexError as exc:
                raise KeyError(f"List index out of range: {part}") from exc
        elif isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key not found in object: '{part}'")
            current = current[part]
        else:
            raise KeyError(f"Cannot descend into non-container at '{part}'")
    return current


def _run_check(
    check: dict,
    result: LangGraphResult,
) -> CheckResult:
    """Run a single check against the LangGraphResult."""
    check_type = check.get("type")

    if check_type == "contains":
        expected = check.get("text", "")
        passed = expected in result.answer_text
        msg = (
            f"answer contains '{expected}'"
            if passed
            else f"answer missing '{expected}'"
        )
        return CheckResult(type=check_type, passed=passed, message=msg)

    if check_type == "regex":
        pattern = check.get("pattern", "")
        try:
            regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
            passed = bool(regex.search(result.answer_text))
            msg = (
                f"answer matches regex '{pattern}'"
                if passed
                else f"answer does not match regex '{pattern}'"
            )
        except re.error as exc:
            passed = False
            msg = f"invalid regex '{pattern}': {exc}"
        return CheckResult(type=check_type, passed=passed, message=msg)

    if check_type == "json_path_numeric_range":
        path = check.get("path")
        min_val = check.get("min")
        max_val = check.get("max")
        try:
            value = _json_path_lookup(result.structured, path)
            numeric_value = float(value)
            in_range = (min_val is None or numeric_value >= float(min_val)) and (
                max_val is None or numeric_value <= float(max_val)
            )
            passed = in_range
            msg = (
                f"{path}={numeric_value} within [{min_val}, {max_val}]"
                if passed
                else f"{path}={numeric_value} outside [{min_val}, {max_val}]"
            )
        except Exception as exc:
            passed = False
            msg = f"failed to evaluate json_path_numeric_range: {exc}"
        return CheckResult(type=check_type, passed=passed, message=msg)

    if check_type == "artifact_exists":
        glob_pattern = check.get("glob")
        matched: list[str] = []
        # Check artifacts reported by the client first
        for artifact in result.artifacts:
            if Path(PROJECT_ROOT, artifact).match(glob_pattern):
                matched.append(artifact)

        # Fall back to filesystem glob if nothing was reported
        if not matched:
            for path in PROJECT_ROOT.glob(glob_pattern):
                matched.append(str(path.relative_to(PROJECT_ROOT)))

        passed = len(matched) > 0
        msg = (
            f"found artifacts matching '{glob_pattern}': {matched}"
            if passed
            else f"no artifacts matching '{glob_pattern}'"
        )
        return CheckResult(type=check_type, passed=passed, message=msg)

    return CheckResult(
        type=str(check_type),
        passed=False,
        message=f"unknown check type: {check_type}",
    )


def run_case(case: dict) -> CaseResult:
    """Run a single evaluation case."""
    case_id = case.get("id") or "<missing-id>"
    checks = case.get("checks") or []
    critical = bool(case.get("critical", False))

    try:
        result = query_langgraph(case["query"])
    except Exception as exc:
        return CaseResult(
            id=case_id,
            score=0.0,
            latency_seconds=0.0,
            passed=False,
            critical=critical,
            error=str(exc),
            checks=[],
        )

    check_results: list[CheckResult] = []
    for check in checks:
        check_results.append(_run_check(check, result))

    if check_results:
        passed_checks = sum(1 for c in check_results if c.passed)
        score = passed_checks / len(check_results)
    else:
        # If no checks are defined, treat as neutral (score 1.0) but log a warning.
        score = 1.0
        check_results.append(
            CheckResult(
                type="meta",
                passed=True,
                message="no checks defined; case treated as passing by default",
            )
        )

    return CaseResult(
        id=case_id,
        score=score,
        latency_seconds=result.latency_seconds,
        passed=score >= 1.0,
        critical=critical,
        error=None,
        checks=check_results,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run LangGraph evaluation suite.")
    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Directory containing YAML evaluation case files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write JSON evaluation report.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.85,
        help="Fail if overall score is below this threshold.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of cases to run.",
    )

    args = parser.parse_args(argv)

    cases_dir = (PROJECT_ROOT / args.cases).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()

    raw_cases = _load_cases(cases_dir, max_cases=args.max_cases)

    case_results: list[CaseResult] = []
    for case in raw_cases:
        case_results.append(run_case(case))

    scores = [c.score for c in case_results]
    overall_score = mean(scores) if scores else 0.0

    any_critical_fail = any(c.critical and not c.passed for c in case_results)
    passed_threshold = overall_score >= float(args.fail_under)
    overall_pass = passed_threshold and not any_critical_fail

    report: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),
            "fail_under": args.fail_under,
            "cases_dir": str(cases_dir.relative_to(PROJECT_ROOT)),
        },
        "overall_score": overall_score,
        "pass": overall_pass,
        "case_results": [
            {
                **{
                    "id": c.id,
                    "score": c.score,
                    "latency_seconds": c.latency_seconds,
                    "passed": c.passed,
                    "critical": c.critical,
                    "error": c.error,
                },
                "checks": [asdict(ch) for ch in c.checks],
            }
            for c in case_results
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Exit code logic
    if not overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
