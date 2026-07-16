import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from threading import Lock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import utils
import call_llm
from get_context_summary import ContextFuncSummary
from get_entrypoint import EntrypointExtractor
from get_functionality_summary import SingleFuncSummary
from get_reachable_func import ReachableFuncExtractor


class UsageCollector:
    def __init__(self):
        self._lock = Lock()
        self.reset()

    def __call__(self, event):
        with self._lock:
            self.events.append(event)

    def reset(self):
        with self._lock:
            self.events = []

    def snapshot(self):
        with self._lock:
            events = list(self.events)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for event in events:
            usage = event.get("usage", {})
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
        return {
            "request_count": len(events),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


def build_benchmark_path_dict(base_path_dict, benchmark_name):
    benchmark_root = os.path.join(base_path_dict["result"], benchmark_name)
    path_dict = deepcopy(base_path_dict)
    path_dict["result"] = benchmark_root
    path_dict["entrypoint"] = os.path.join(benchmark_root, "entrypoints")
    path_dict["call_chain"] = os.path.join(benchmark_root, "call_chain")
    path_dict["decompile"] = path_dict["call_chain"]
    path_dict["summary"] = os.path.join(benchmark_root, "summary")
    path_dict["single_func_summary"] = path_dict["summary"]
    path_dict["reachable_func"] = os.path.join(benchmark_root, "reachable_func")
    path_dict["functions"] = os.path.join(benchmark_root, "functions")
    path_dict["context"] = os.path.join(benchmark_root, "context")
    path_dict["context_summary"] = os.path.join(benchmark_root, "context_summary")
    path_dict["benchmark"] = benchmark_root
    for key in ("result", "entrypoint", "call_chain", "summary", "reachable_func", "functions", "context", "context_summary"):
        os.makedirs(path_dict[key], exist_ok=True)
    return path_dict


def get_apk_folder(path_dict, folder_name):
    normalized = utils.normalize_sample_folder_name(folder_name)
    if normalized == "malradar":
        return path_dict["archived_apk"]
    if normalized == "new":
        return path_dict["latest_apk"]
    if normalized == "benign":
        return path_dict["benign_apk"]
    raise ValueError(f"Unsupported folder name: {folder_name}")


def select_samples(path_dict, folder_name, sample_size, seed):
    apk_folder = get_apk_folder(path_dict, folder_name)
    candidates = [
        apk_name
        for apk_name in utils.get_indexed_apk_list(folder_name)
        if os.path.exists(os.path.join(apk_folder, f"{apk_name}.apk"))
    ]
    if len(candidates) < sample_size:
        raise ValueError(f"{folder_name} only has {len(candidates)} APKs, fewer than requested {sample_size}")
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, sample_size))


def distribute_total_samples(total_samples, folder_names):
    base = total_samples // len(folder_names)
    remainder = total_samples % len(folder_names)
    distribution = {}
    for idx, folder_name in enumerate(folder_names):
        distribution[folder_name] = base + (1 if idx < remainder else 0)
    return distribution


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stage_entrypoint(extractor, apk_name, apk_path, entrypoint_path):
    started_at = time.perf_counter()
    metrics = {
        "status": "success",
        "entrypoint_count": 0,
        "seconds": 0.0,
    }
    try:
        entrypoints = extractor.get_entrypoints(apk_path)
        os.makedirs(os.path.dirname(entrypoint_path), exist_ok=True)
        with open(entrypoint_path, "w", encoding="utf-8") as f:
            for cls, method in entrypoints:
                f.write(f"{cls} -> {method}\n")
        metrics["entrypoint_count"] = len(entrypoints)
    except Exception as exc:
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    metrics["seconds"] = time.perf_counter() - started_at
    return metrics


def stage_reachable(extractor, apk_name, entrypoint_path, call_chain_path):
    started_at = time.perf_counter()
    metrics = {
        "status": "success",
        "reachable_function_count": 0,
        "seconds": 0.0,
    }
    try:
        entrypoints = extractor._load_entrypoints(entrypoint_path) if os.path.exists(entrypoint_path) else []
        rows = extractor.processing(apk_name, entrypoints)
        metrics["reachable_function_count"] = len(rows)
        if os.path.exists(call_chain_path):
            metrics["reachable_function_count"] = len(load_jsonl(call_chain_path))
    except Exception as exc:
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    metrics["seconds"] = time.perf_counter() - started_at
    return metrics


def stage_functionality(summary_helper, apk_name, usage_collector, output_path):
    usage_collector.reset()
    started_at = time.perf_counter()
    metrics = {
        "status": "success",
        "function_count": 0,
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "seconds": 0.0,
    }
    try:
        summary_helper._get_func_for_each_apk(apk_name)
        function_file_path = os.path.join(summary_helper.path_dict["functions"], summary_helper.folder_name, f"{apk_name}.json")
        func_dict = load_json(function_file_path, default={}) or {}
        metrics["function_count"] = len(func_dict)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rows = []
        if func_dict:
            with ThreadPoolExecutor(max_workers=summary_helper.max_workers_funcs) as executor:
                future_to_name = {
                    executor.submit(summary_helper._summarize_one, func_name, func_src): func_name
                    for func_name, func_src in func_dict.items()
                }
                for future in as_completed(future_to_name):
                    func_name = future_to_name[future]
                    try:
                        rows.append(future.result())
                    except Exception as exc:
                        rows.append({"function": func_name, "summary": "None", "error": str(exc)})
            write_jsonl(output_path, rows)
            summary_helper._retry_none_summary_single_func(apk_name)
        else:
            write_jsonl(output_path, rows)
    except Exception as exc:
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    token_stats = usage_collector.snapshot()
    metrics.update(token_stats)
    metrics["seconds"] = time.perf_counter() - started_at
    return metrics


def stage_context(context_helper, apk_name, usage_collector, output_path):
    usage_collector.reset()
    started_at = time.perf_counter()
    metrics = {
        "status": "success",
        "function_count": 0,
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "seconds": 0.0,
    }
    try:
        context_helper.prepare_context(apk_name)
        context_path = os.path.join(
            context_helper.path_dict["context"],
            context_helper.model_name,
            context_helper.folder_name,
            f"{apk_name}.json",
        )
        context_data = load_json(context_path, default={}) or {}
        metrics["function_count"] = len(context_data)
        rows = []
        if context_data:
            with ThreadPoolExecutor(max_workers=context_helper.max_workers_funcs) as executor:
                future_to_name = {
                    executor.submit(
                        context_helper._summary_one,
                        func_name,
                        func_data["source"],
                        func_data["callers"],
                        func_data["callees"],
                    ): func_name
                    for func_name, func_data in context_data.items()
                }
                for future in as_completed(future_to_name):
                    func_name = future_to_name[future]
                    try:
                        rows.append(future.result())
                    except Exception as exc:
                        rows.append(
                            {
                                "function": func_name,
                                "summary": f"error: {exc}",
                                "sensitivity_score": -1,
                                "reasoning": "Processing failed",
                            }
                        )
        row_map = {row["function"]: row for row in rows}
        ordered_rows = [row_map[func_name] for func_name in context_data.keys() if func_name in row_map]
        write_jsonl(output_path, ordered_rows)
    except Exception as exc:
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    token_stats = usage_collector.snapshot()
    metrics.update(token_stats)
    metrics["seconds"] = time.perf_counter() - started_at
    return metrics


def build_summary(rows):
    stage_names = ("stage1_entrypoint", "stage2_reachable", "stage3_functionality", "stage4_context")
    groups = {"overall": rows}
    for folder in sorted({row["folder"] for row in rows}):
        groups[folder] = [row for row in rows if row["folder"] == folder]

    summary = {}
    for group_name, group_rows in groups.items():
        stage_summary = {}
        for stage_name in stage_names:
            stage_rows = [row[stage_name] for row in group_rows]
            success_rows = [row for row in stage_rows if row.get("status") == "success"]
            count = len(stage_rows)
            stage_summary[stage_name] = {
                "samples": count,
                "success_samples": len(success_rows),
                "avg_seconds_all": round(sum(row.get("seconds", 0.0) for row in stage_rows) / count, 6) if count else 0.0,
                "avg_seconds_success": round(sum(row.get("seconds", 0.0) for row in success_rows) / len(success_rows), 6) if success_rows else 0.0,
            }
            if stage_name in ("stage3_functionality", "stage4_context"):
                stage_summary[stage_name].update(
                    {
                        "avg_prompt_tokens": round(sum(row.get("prompt_tokens", 0) for row in stage_rows) / count, 2) if count else 0.0,
                        "avg_completion_tokens": round(sum(row.get("completion_tokens", 0) for row in stage_rows) / count, 2) if count else 0.0,
                        "avg_total_tokens": round(sum(row.get("total_tokens", 0) for row in stage_rows) / count, 2) if count else 0.0,
                        "total_prompt_tokens": sum(row.get("prompt_tokens", 0) for row in stage_rows),
                        "total_completion_tokens": sum(row.get("completion_tokens", 0) for row in stage_rows),
                        "total_tokens": sum(row.get("total_tokens", 0) for row in stage_rows),
                    }
                )
        summary[group_name] = stage_summary
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark the 4-step preparation pipeline on sampled APKs.")
    parser.add_argument("--model", default="deepseek", choices=["deepseek", "qwen", "llama", "coder", "gpt", "gemini", "claude"])
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--total-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folders", nargs="+", default=["benign", "malradar", "new"])
    parser.add_argument("--benchmark-name", default="preparation_benchmark")
    args = parser.parse_args()

    if args.model != "deepseek":
        raise ValueError("This benchmark script is currently intended for DeepSeek-backed step 3 and 4 runs.")

    base_path_dict = utils.get_path_dict()
    bench_path_dict = build_benchmark_path_dict(base_path_dict, args.benchmark_name)
    benchmark_dir = bench_path_dict["benchmark"]

    if args.total_samples is not None:
        sample_distribution = distribute_total_samples(args.total_samples, args.folders)
    else:
        sample_distribution = {folder_name: args.sample_size for folder_name in args.folders}

    selected_samples = {}
    for offset, folder_name in enumerate(args.folders):
        selected_samples[folder_name] = select_samples(
            base_path_dict,
            folder_name,
            sample_distribution[folder_name],
            args.seed + offset,
        )

    write_json(
        os.path.join(benchmark_dir, "selected_samples.json"),
        {
            "seed": args.seed,
            "sample_size_per_folder": None if args.total_samples is not None else args.sample_size,
            "total_samples": args.total_samples,
            "folders": args.folders,
            "sample_distribution": sample_distribution,
            "selected_samples": selected_samples,
        },
    )

    usage_collector = UsageCollector()
    call_llm.set_usage_hook(usage_collector)

    results = []
    for folder_name in args.folders:
        apk_folder = get_apk_folder(base_path_dict, folder_name)
        entrypoint_extractor = EntrypointExtractor(bench_path_dict, folder_name)
        reachable_extractor = ReachableFuncExtractor(bench_path_dict, folder_name)
        summary_helper = SingleFuncSummary(args.model, folder_name, bench_path_dict)
        context_helper = ContextFuncSummary(args.model, folder_name, bench_path_dict)

        for index, apk_name in enumerate(selected_samples[folder_name], start=1):
            print(f"[{folder_name}] {index}/{len(selected_samples[folder_name])} -> {apk_name}")
            apk_path = os.path.join(apk_folder, f"{apk_name}.apk")
            entrypoint_path = os.path.join(bench_path_dict["entrypoint"], folder_name, f"{apk_name}.txt")
            call_chain_path = os.path.join(bench_path_dict["call_chain"], folder_name, f"{apk_name}.jsonl")
            summary_output_path = os.path.join(bench_path_dict["summary"], args.model, folder_name, f"{apk_name}.jsonl")
            context_output_path = os.path.join(bench_path_dict["context_summary"], args.model, folder_name, f"{apk_name}.jsonl")

            row = {
                "folder": folder_name,
                "apk_name": apk_name,
                "apk_path": apk_path,
                "stage1_entrypoint": stage_entrypoint(entrypoint_extractor, apk_name, apk_path, entrypoint_path),
                "stage2_reachable": stage_reachable(reachable_extractor, apk_name, entrypoint_path, call_chain_path),
                "stage3_functionality": stage_functionality(summary_helper, apk_name, usage_collector, summary_output_path),
                "stage4_context": stage_context(context_helper, apk_name, usage_collector, context_output_path),
            }
            results.append(row)
            write_json(os.path.join(benchmark_dir, "summary.json"), build_summary(results))
            write_jsonl(os.path.join(benchmark_dir, "per_sample_metrics.jsonl"), results)

    summary = build_summary(results)
    write_json(os.path.join(benchmark_dir, "summary.json"), summary)
    write_jsonl(os.path.join(benchmark_dir, "per_sample_metrics.jsonl"), results)

    csv_path = os.path.join(benchmark_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "folder",
                "apk_name",
                "stage1_status",
                "stage1_seconds",
                "stage1_entrypoint_count",
                "stage2_status",
                "stage2_seconds",
                "stage2_reachable_function_count",
                "stage3_status",
                "stage3_seconds",
                "stage3_function_count",
                "stage3_request_count",
                "stage3_prompt_tokens",
                "stage3_completion_tokens",
                "stage3_total_tokens",
                "stage4_status",
                "stage4_seconds",
                "stage4_function_count",
                "stage4_request_count",
                "stage4_prompt_tokens",
                "stage4_completion_tokens",
                "stage4_total_tokens",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["folder"],
                    row["apk_name"],
                    row["stage1_entrypoint"].get("status"),
                    row["stage1_entrypoint"].get("seconds"),
                    row["stage1_entrypoint"].get("entrypoint_count"),
                    row["stage2_reachable"].get("status"),
                    row["stage2_reachable"].get("seconds"),
                    row["stage2_reachable"].get("reachable_function_count"),
                    row["stage3_functionality"].get("status"),
                    row["stage3_functionality"].get("seconds"),
                    row["stage3_functionality"].get("function_count"),
                    row["stage3_functionality"].get("request_count"),
                    row["stage3_functionality"].get("prompt_tokens"),
                    row["stage3_functionality"].get("completion_tokens"),
                    row["stage3_functionality"].get("total_tokens"),
                    row["stage4_context"].get("status"),
                    row["stage4_context"].get("seconds"),
                    row["stage4_context"].get("function_count"),
                    row["stage4_context"].get("request_count"),
                    row["stage4_context"].get("prompt_tokens"),
                    row["stage4_context"].get("completion_tokens"),
                    row["stage4_context"].get("total_tokens"),
                ]
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
