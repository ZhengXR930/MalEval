import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken
import utils


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent


def get_external_paths() -> Dict[str, Path]:
    path_dict = utils.get_path_dict()
    return {key: Path(value) for key, value in path_dict.items()}


def load_gt_info(folder_name: str) -> Dict[str, Dict[str, Any]]:
    info_root = get_external_paths()["info"]
    mapping = {
        "malradar": info_root / "archived_sample_info.json",
        "new": info_root / "latest_sample_info.json",
        "benign": info_root / "benign_sample_info.json",
    }
    gt_path = mapping.get(folder_name)
    if gt_path is None or not gt_path.exists():
        if folder_name == "benign":
            return {}
        raise FileNotFoundError(f"Ground-truth info not found for folder '{folder_name}': {gt_path}")
    with gt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_indexed_apk_list(folder_name: str) -> List[str]:
    return sorted(load_gt_info(folder_name).keys())


def get_lamd_split_dir(folder_name: str) -> Optional[str]:
    mapping = {
        "new": "new_malware",
        "malradar": "selected_malradar",
        "benign": "selected_benign",
    }
    return mapping.get(folder_name)


def collect_jsonl_apk_names(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted(path.stem for path in folder.glob("*.jsonl"))


def count_jsonl_records(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def rank_apks_by_call_chain_size(call_chain_folder: Path, indexed_apks: Optional[Sequence[str]] = None) -> List[str]:
    apk_names = list(indexed_apks) if indexed_apks is not None else collect_jsonl_apk_names(call_chain_folder)
    ranked = []
    for apk_name in apk_names:
        call_chain_path = call_chain_folder / f"{apk_name}.jsonl"
        if call_chain_path.exists():
            ranked.append((0, count_jsonl_records(call_chain_path), apk_name))
        else:
            ranked.append((1, float("inf"), apk_name))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [apk_name for _, _, apk_name in ranked]


def collect_directory_apk_names(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted(path.name for path in folder.iterdir() if path.is_dir())


def get_llm_client(model: str = "qwen"):
    from call_llm import (
        call_claude,
        call_coder,
        call_deepseek,
        call_gemini,
        call_gpt,
        call_llama,
        call_qwen3,
    )

    if model == "qwen":
        return call_qwen3
    if model == "deepseek":
        return call_deepseek
    if model == "llama":
        return call_llama
    if model == "coder":
        return call_coder
    if model == "gpt":
        return call_gpt
    if model == "gemini":
        return call_gemini
    if model == "claude":
        return call_claude
    raise ValueError(f"Invalid model: {model}")


def count_tokens(text: str, model_name: str) -> int:
    if model_name in {"qwen", "coder"}:
        from call_llm import get_qwen_tokenizer

        tokenizer = get_qwen_tokenizer()
        return len(tokenizer.encode(text))
    if model_name == "gemini":
        return max(1, len(text) // 4)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def get_model_budget(model_name: str) -> Tuple[int, int, int]:
    if model_name == "qwen":
        return (40960, 4096, 200)
    if model_name == "deepseek":
        return (64000, 4096, 500)
    if model_name in {"llama", "coder"}:
        return (32768, 4096, 500)
    if model_name == "gpt":
        return (128000, 4096, 500)
    if model_name == "gemini":
        return (1048576, 8192, 500)
    if model_name == "claude":
        return (200000, 6000, 500)
    raise ValueError(f"Unsupported model: {model_name}")


def pack_blocks_with_budget(
    *,
    model_name: str,
    system_prompt: str,
    header: str,
    blocks: Sequence[str],
) -> Tuple[str, int, bool]:
    token_limit, reserved_for_completion, safety_margin = get_model_budget(model_name)
    system_tokens = count_tokens(system_prompt, model_name)
    available_user_tokens = token_limit - reserved_for_completion - safety_margin - system_tokens
    if available_user_tokens <= 0:
        raise ValueError("System prompt is too long for the configured model budget.")

    current_query = header.strip()
    included_count = 0
    truncated = False

    if count_tokens(current_query, model_name) > available_user_tokens:
        raise ValueError("Query header already exceeds the available user token budget.")

    for block in blocks:
        candidate = f"{current_query}\n\n{block}".strip()
        if count_tokens(candidate, model_name) > available_user_tokens:
            truncated = True
            break
        current_query = candidate
        included_count += 1

    return current_query, included_count, truncated


def estimate_total_tokens(
    *,
    model_name: str,
    system_prompt: str,
    final_user_message: str,
    result: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    prompt_tokens = count_tokens(system_prompt, model_name) + count_tokens(final_user_message, model_name)
    response_tokens = 0
    if result is not None:
        response_tokens = count_tokens(json.dumps(result, ensure_ascii=False), model_name)
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens_est": response_tokens,
        "total_tokens_est": prompt_tokens + response_tokens,
    }


def normalize_function_reference(name: Any) -> str:
    text = " ".join(str(name).strip().split())
    if not text:
        return ""

    text = text.strip("`\"'")

    patterns = [
        re.compile(r"^(?P<class>L?[\w/$.-]+);?\s+(?P<method>[\w$<>]+)\b"),
        re.compile(r"^(?P<class>[\w.$/]+)\s*:\s*.*?\b(?P<method>[\w$<>]+)\s*\([^)]*\)\s*$"),
        re.compile(r"^(?P<class>[\w.$/]+)\s*:\s*.*?\b(?P<method>[\w$<>]+)\s*$"),
        re.compile(r"^(?P<class>.+)\.(?P<method>[\w$<>]+)(?:\s*\(|$)"),
    ]

    for pattern in patterns:
        match = pattern.match(text)
        if not match:
            continue
        class_name = match.group("class").strip()
        method_name = match.group("method").strip()
        if class_name.startswith("L") and "/" in class_name:
            class_name = class_name[1:]
        class_name = class_name.rstrip(";").replace("/", ".")
        if class_name and method_name:
            return f"{class_name}.{method_name}"

    return text


def compute_sas_score(result: Dict[str, Any], final_user_message: str) -> float:
    input_functions = {
        normalized
        for name in re.findall(r"Function:\s+([^\n]+)", final_user_message)
        if (normalized := normalize_function_reference(name))
    }
    report_functions = set()

    for evidence in result.get("atom_evidence", []):
        support_funcs = evidence.get("support_functions", [])
        if isinstance(support_funcs, list):
            report_functions.update(
                normalized
                for item in support_funcs
                if (normalized := normalize_function_reference(item))
            )

    if not report_functions:
        return 0.0

    matches = report_functions.intersection(input_functions)
    return len(matches) / len(report_functions)


def write_statistics(
    *,
    log_path: Path,
    header_lines: Iterable[str],
    processed_stats: List[Dict[str, Any]],
    failed_apks: List[str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"{line}\n")

        f.write(f"processed_apks: {len(processed_stats)}\n")
        f.write(f"failed_apks: {len(failed_apks)}\n")
        if failed_apks:
            f.write(f"failed_apk_list: {failed_apks}\n")

        if not processed_stats:
            return

        numeric_keys = [
            "prompt_tokens",
            "response_tokens_est",
            "total_tokens_est",
            "included_primary_items",
            "included_total_functions",
            "generation_seconds",
        ]
        for key in numeric_keys:
            values = [float(item[key]) for item in processed_stats if key in item]
            if values:
                f.write(f"avg_{key}: {sum(values) / len(values):.2f}\n")
                f.write(f"min_{key}: {min(values):.2f}\n")
                f.write(f"max_{key}: {max(values):.2f}\n")
