
import os
import json
import hashlib
from urllib.parse import urlsplit


try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_ROOT)
WORKSPACE_ROOT = REPO_ROOT

artifact_folder = os.path.join(REPO_ROOT, "artifacts")
result_folder = os.path.join(REPO_ROOT, "results")
dataset_folder = os.path.join(REPO_ROOT, "dataset")
log_folder = os.path.join(REPO_ROOT, "log")
info_folder = os.path.join(REPO_ROOT, "info")

path_dict = {
    "repo_root": REPO_ROOT,
    "workspace_root": WORKSPACE_ROOT,
    "artifacts": artifact_folder,
    "result": result_folder,
    "entrypoint": f"{artifact_folder}/entrypoints",
    "call_chain": f"{artifact_folder}/call_chain",
    "decompile": f"{artifact_folder}/call_chain",
    "log": log_folder,
    "info": info_folder,

    "archived_apk": f"{dataset_folder}/selected_malradar",
    "latest_apk": f"{dataset_folder}/new_malware",
    "benign_apk": f"{dataset_folder}/selected_benign",

    
    "summary": f"{result_folder}/summary",
    "single_func_summary": f"{result_folder}/summary",
    "reachable_func": f"{result_folder}/reachable_func",
    "functions": f"{result_folder}/functions",
    "context": f"{result_folder}/context",
    "context_summary": f"{result_folder}/context_summary",
    "no_context_summary": f"{result_folder}/no_context_summary",
    "behavior": f"{result_folder}/behavior",
    "topk_removal_behavior": f"{result_folder}/topk_removal_behavior",
    "meta_behavior": f"{result_folder}/meta_behavior",
    "topk_removal_meta_behavior": f"{result_folder}/topk_removal_meta_behavior",
    "no_context_behavior": f"{result_folder}/no_context_behavior",
    "topk_removal_no_context_behavior": f"{result_folder}/topk_removal_no_context_behavior",
    "raw_code_behavior": f"{result_folder}/raw_code_behavior",
    "slicing_behavior": f"{result_folder}/slicing_behavior",

    "meta_info": f"{result_folder}/meta_info",
    "reports": f"{artifact_folder}/reports",
    "lamd_dataset": os.path.join(REPO_ROOT, "lamd-processed-dataset"),
    }


def get_path_dict():
    create_folder()
    return path_dict


def normalize_sample_folder_name(folder_name):
    mapping = {
        "archived": "malradar",
        "latest": "new",
        "malradar": "malradar",
        "new": "new",
        "benign": "benign",
    }
    normalized = mapping.get(folder_name)
    if normalized is None:
        raise ValueError(f"Unsupported folder name: {folder_name}")
    return normalized


def get_sample_info_path(folder_name):
    normalized = normalize_sample_folder_name(folder_name)
    if normalized == "malradar":
        return os.path.join(path_dict["info"], "archived_sample_info.json")
    if normalized == "new":
        return os.path.join(path_dict["info"], "latest_sample_info.json")
    return os.path.join(path_dict["info"], "benign_sample_info.json")


def load_sample_info(folder_name):
    info_path = get_sample_info_path(folder_name)
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_indexed_apk_list(folder_name):
    return sorted(load_sample_info(folder_name).keys())

def load_apk_list_from_csv(csv_path):
    if pd is None:
        raise ModuleNotFoundError("pandas is required to load APK lists from CSV files.")
    df = pd.read_csv(csv_path)
    return df["sha256"].tolist()

def load_apk_list_from_txt(txt_path):
    with open(txt_path, "r") as f:
        return f.read().splitlines()

def create_folder():
    create_keys = [
        "artifacts",
        "result",
        "log",
        "info",
        "summary",
        "single_func_summary",
        "reachable_func",
        "functions",
        "context",
        "context_summary",
        "no_context_summary",
        "behavior",
        "topk_removal_behavior",
        "meta_behavior",
        "topk_removal_meta_behavior",
        "no_context_behavior",
        "topk_removal_no_context_behavior",
        "raw_code_behavior",
        "slicing_behavior",
        "meta_info",
        "reports",
    ]
    for key in create_keys:
        path = path_dict[key]
        if not os.path.exists(path):
            os.makedirs(path)

def get_llm_client(model="qwen"):
    from call_llm import (
        call_claude,
        call_coder,
        call_deepseek,
        call_gemini,
        call_gpt,
        call_gpt5,
        call_llama,
        call_qwen3,
    )

    if model == "qwen":
        return call_qwen3
    elif model == "deepseek":
        return call_deepseek
    elif model == "llama":
        return call_llama
    elif model == "coder":
        return call_coder
    elif model == "gpt":
        return call_gpt
    elif model == "gpt-5":
        return call_gpt5
    elif model == "gemini":
        return call_gemini
    elif model == "claude":
        return call_claude
    else:
        raise ValueError(f"Invalid model: {model}")

def check_file_exist(folder_name):
    indexed_apks = get_indexed_apk_list(folder_name)
    sample_folder = os.path.join(path_dict["call_chain"], normalize_sample_folder_name(folder_name))
    if not os.path.exists(sample_folder):
        return indexed_apks

    ranked_apks = []
    for apk_name in indexed_apks:
        call_chain_path = os.path.join(sample_folder, f"{apk_name}.jsonl")
        if not os.path.exists(call_chain_path):
            ranked_apks.append((1, float("inf"), apk_name))
            continue
        with open(call_chain_path, "r", encoding="utf-8") as f:
            record_count = sum(1 for _ in f)
        ranked_apks.append((0, record_count, apk_name))

    ranked_apks.sort(key=lambda item: (item[0], item[1], item[2]))
    return [apk_name for _, _, apk_name in ranked_apks]

def make_doc_id(url: str) -> str:
    """Map a report URL to the stable doc folder name used under result/reports."""
    s = urlsplit(url)
    domain = s.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    
    path = (s.path or "/").strip("/").replace("/", "-")
    path = path[:40] if path else "root"
    
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8] 
    return f"{domain}.{path}.{h}"
