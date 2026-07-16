#!/usr/bin/env python3
"""Run MalEval metric groups with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def rq_report_type(flag: str) -> str:
    return "meta_behavior" if flag == "meta" else "behavior"


def rq_flag(flag: str) -> str:
    return "context" if flag == "meta" else flag


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MalEval metric groups for one model/setting.")
    parser.add_argument("--model", required=True, help="Model name, e.g., gpt, deepseek, qwen.")
    parser.add_argument(
        "--flag",
        required=True,
        choices=["context", "no_context", "meta"],
        help="Evaluation setting: Table 3 context, Table 5 no_context, or Table 6 meta.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["malradar", "new"],
        choices=["malradar", "new", "benign"],
        help="Splits for judge-based EAS/RQ metrics. Paper values use malradar and new.",
    )
    parser.add_argument("--judge-model", default="gpt-5", help="Judge model used by EAS.")
    parser.add_argument("--rq-judge-model", default="gpt-5", help="Judge model used by report-quality generation.")
    parser.add_argument("--skip-eas", action="store_true", help="Skip EAS judge-model calls.")
    parser.add_argument("--include-rq", action="store_true", help="Also run report-quality generation.")
    args = parser.parse_args()

    py = sys.executable
    run([py, "src/metrics/aec_b_f1.py", "--model", args.model, "--flag", args.flag])
    run([py, "src/metrics/fpcr_tpmr_f1c.py", "--model", args.model, "--flag", args.flag])

    if not args.skip_eas:
        for split in args.splits:
            run(
                [
                    py,
                    "src/metrics/eas.py",
                    "--model",
                    args.model,
                    "--flag",
                    args.flag,
                    "--split",
                    split,
                    "--judge-model",
                    args.judge_model,
                ]
            )

    if args.include_rq:
        for split in args.splits:
            if split == "benign":
                continue
            run(
                [
                    py,
                    "src/metrics/rq_generation.py",
                    "--model",
                    args.model,
                    "--report-type",
                    rq_report_type(args.flag),
                    "--flag",
                    rq_flag(args.flag),
                    "--mode",
                    "normal",
                    "--folder",
                    split,
                    "--judge-model",
                    args.rq_judge_model,
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
