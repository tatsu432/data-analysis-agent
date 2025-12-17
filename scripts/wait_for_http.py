"""
Simple helper script to poll an HTTP endpoint until it is ready.

Usage:

    python scripts/wait_for_http.py http://localhost:8082/health \
        --timeout 60 --interval 2
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import requests


def wait_for_http(url: str, timeout: int, interval: float) -> bool:
    """Poll the given URL until it returns a status code < 500 or timeout."""
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            response = requests.get(url, timeout=5)
            status = response.status_code
            if 200 <= status < 500:
                print(f"[wait_for_http] Ready: {url} (status={status})")
                return True
            print(f"[wait_for_http] Attempt {attempt}: status={status}, retrying...")
        except Exception as exc:
            print(f"[wait_for_http] Attempt {attempt}: error={exc}, retrying...")

        time.sleep(interval)

    print(f"[wait_for_http] Timeout after {timeout}s waiting for {url}")
    return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wait for an HTTP endpoint to be ready."
    )
    parser.add_argument("url", type=str, help="URL to poll.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Maximum time to wait in seconds.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between polls.",
    )

    args = parser.parse_args(argv)

    ok = wait_for_http(args.url, timeout=args.timeout, interval=args.interval)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
