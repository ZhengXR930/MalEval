#!/usr/bin/env python3
"""Run MalEval static analysis from packaged APKs.

This wrapper exposes the APK-to-artifact part of the pipeline as a reviewer
friendly CLI. It reuses the original entrypoint and reachable-function
extractors and writes outputs with the same schemas as the released artifacts:
entrypoints, call_chain, functions, and reachable_func.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import utils
from get_entrypoint import EntrypointExtractor
from get_functionality_summary import SingleFuncSummary
from get_reachable_func import ReachableFuncExtractor


def build_output_path_dict(output_root: Path) -> dict:
    path_dict = deepcopy(utils.get_path_dict())
    output_root = output_root.resolve()
    path_dict["result"] = str(output_root)
    path_dict["entrypoint"] = str(output_root / "entrypoints")
    path_dict["call_chain"] = str(output_root / "call_chain")
    path_dict["decompile"] = path_dict["call_chain"]
    path_dict["functions"] = str(output_root / "functions")
    path_dict["reachable_func"] = str(output_root / "reachable_func")
    for key in ("entrypoint", "call_chain", "functions", "reachable_func"):
        Path(path_dict[key]).mkdir(parents=True, exist_ok=True)
    return path_dict


def apk_folder_for_split(path_dict: dict, folder_name: str) -> Path:
    normalized = utils.normalize_sample_folder_name(folder_name)
    if normalized == "malradar":
        return Path(path_dict["repo_root"]) / "artifacts" / "apk" / "malradar"
    if normalized == "new":
        return Path(path_dict["repo_root"]) / "artifacts" / "apk" / "new"
    if normalized == "benign":
        return Path(path_dict["repo_root"]) / "artifacts" / "apk" / "benign"
    raise ValueError(f"Unsupported folder name: {folder_name}")


def select_apks(folder_name: str, apk: str | None, sample_size: int | None, seed: int) -> list[str]:
    indexed = list(utils.get_indexed_apk_list(folder_name))
    if apk:
        if apk not in indexed:
            raise ValueError(f"{apk} is not listed in info for split {folder_name}")
        return [apk]
    if sample_size is None:
        return indexed
    if sample_size > len(indexed):
        raise ValueError(f"sample-size {sample_size} exceeds split size {len(indexed)}")
    rng = random.Random(seed)
    return sorted(rng.sample(indexed, sample_size))


def remove_existing_outputs(path_dict: dict, folder_name: str, apk_name: str) -> None:
    targets = [
        Path(path_dict["entrypoint"]) / folder_name / f"{apk_name}.txt",
        Path(path_dict["call_chain"]) / folder_name / f"{apk_name}.jsonl",
        Path(path_dict["functions"]) / folder_name / f"{apk_name}.json",
        Path(path_dict["reachable_func"]) / folder_name / f"{apk_name}.json",
    ]
    for target in targets:
        if target.exists():
            target.unlink()


def run_one(path_dict: dict, folder_name: str, apk_name: str, force: bool) -> dict:
    apk_folder = apk_folder_for_split(path_dict, folder_name)
    apk_path = apk_folder / f"{apk_name}.apk"
    if not apk_path.is_file():
        raise FileNotFoundError(f"missing packaged APK: {apk_path}")

    if force:
        remove_existing_outputs(path_dict, folder_name, apk_name)

    entrypoint_dir = Path(path_dict["entrypoint"]) / folder_name
    entrypoint_dir.mkdir(parents=True, exist_ok=True)
    entrypoint_path = entrypoint_dir / f"{apk_name}.txt"

    call_chain_dir = Path(path_dict["call_chain"]) / folder_name
    call_chain_dir.mkdir(parents=True, exist_ok=True)
    call_chain_path = call_chain_dir / f"{apk_name}.jsonl"

    print(f"[static] {folder_name}/{apk_name}", flush=True)

    entrypoint_extractor = EntrypointExtractor(path_dict, folder_name)
    if entrypoint_path.exists():
        print(f"  entrypoints: reuse {entrypoint_path}", flush=True)
    else:
        entrypoints = entrypoint_extractor.get_entrypoints(str(apk_path))
        with entrypoint_path.open("w", encoding="utf-8") as f:
            for cls, method in entrypoints:
                f.write(f"{cls} -> {method}\n")
        print(f"  entrypoints: {len(entrypoints)}", flush=True)

    reachable_extractor = ReachableFuncExtractor(path_dict, folder_name)
    if call_chain_path.exists():
        print(f"  call_chain: reuse {call_chain_path}", flush=True)
    else:
        reachable_extractor.process_single_apk(apk_name)

    summary_helper = SingleFuncSummary("gpt", folder_name, path_dict)
    summary_helper._get_func_for_each_apk(apk_name)

    function_path = Path(path_dict["functions"]) / folder_name / f"{apk_name}.json"
    reachable_path = Path(path_dict["reachable_func"]) / folder_name / f"{apk_name}.json"
    with function_path.open("r", encoding="utf-8") as f:
        function_count = len(json.load(f))
    with reachable_path.open("r", encoding="utf-8") as f:
        reachable_count = len(json.load(f))

    return {
        "folder": folder_name,
        "apk": apk_name,
        "entrypoints": str(entrypoint_path),
        "call_chain": str(call_chain_path),
        "functions": str(function_path),
        "reachable_func": str(reachable_path),
        "function_count": function_count,
        "reachable_function_count": reachable_count,
    }


def copy_to_artifacts(output_root: Path, folder_name: str, apk_names: list[str]) -> None:
    artifact_root = Path(utils.get_path_dict()["artifacts"])
    for stage, suffix in (
        ("entrypoints", ".txt"),
        ("call_chain", ".jsonl"),
        ("functions", ".json"),
        ("reachable_func", ".json"),
    ):
        dst_dir = artifact_root / stage / folder_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for apk_name in apk_names:
            src = output_root / stage / folder_name / f"{apk_name}{suffix}"
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run static analysis from packaged APKs to MalEval artifact schemas."
    )
    parser.add_argument("--folder", required=True, choices=["malradar", "new", "benign"])
    parser.add_argument("--apk", help="Run one APK sha256 from the selected split.")
    parser.add_argument("--sample-size", type=int, help="Run a random subset from the selected split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="results/static_analysis",
        help="Where to write generated entrypoints/call_chain/functions/reachable_func.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs in output-root.")
    parser.add_argument(
        "--copy-to-artifacts",
        action="store_true",
        help="Copy generated files into artifacts/. Use only when intentionally replacing released artifacts.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    path_dict = build_output_path_dict(output_root)
    apk_names = select_apks(args.folder, args.apk, args.sample_size, args.seed)

    results = []
    for index, apk_name in enumerate(apk_names, start=1):
        print(f"[{index}/{len(apk_names)}]", flush=True)
        results.append(run_one(path_dict, args.folder, apk_name, args.force))

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"static_analysis_{args.folder}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"summary: {summary_path}", flush=True)

    if args.copy_to_artifacts:
        copy_to_artifacts(output_root, args.folder, apk_names)
        print("copied generated static-analysis outputs into artifacts/", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
