
import os
import json
from call_llm import call_deepseek, call_qwen3, call_llama, call_coder, call_gpt, call_gemini, call_claude
import pandas as pd
import re
import hashlib
from urllib.parse import urlsplit, unquote


result_folder = "./result"
dataset_folder = "./dataset"
log_folder = "./log"
info_folder = "./info"

path_dict = {
    "result": result_folder,
    "entrypoint": f"{result_folder}/entrypoints",
    "call_chain": f"{result_folder}/call_chain",
    "log": log_folder,
    "info": info_folder,

    "archived_apk": f"{dataset_folder}/selected_malradar",
    "latest_apk": f"{dataset_folder}/new_malware",
    "benign_apk": f"{dataset_folder}/selected_benign",

    
    "single_func_summary": f"{result_folder}/summary",
    "reachable_func": f"{result_folder}/reachable_func",
    "functions": f"{result_folder}/functions",
    "context": f"{result_folder}/context",
    "context_summary": f"{result_folder}/context_summary",
    "no_context_summary": f"{result_folder}/no_context_summary",
    "behavior": f"{result_folder}/behavior",
    "meta_behavior": f"{result_folder}/meta_behavior",
    "no_context_behavior": f"{result_folder}/no_context_behavior",

    "meta_info": f"{result_folder}/meta_info",
    "report": f"{result_folder}/report",
    }


def get_path_dict():
    create_folder()
    return path_dict

def load_apk_list_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df["sha256"].tolist()

def load_apk_list_from_txt(txt_path):
    with open(txt_path, "r") as f:
        return f.read().splitlines()

def create_folder():
    for path in path_dict.values():
        if not os.path.exists(path):
            os.makedirs(path)

def get_llm_client(model="qwen"):
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
    elif model == "gemini":
        return call_gemini
    elif model == "claude":
        return call_claude
    else:
        raise ValueError(f"Invalid model: {model}")

def check_file_exist(folder_name):
    sample_folder = os.path.join(path_dict["call_chain"], folder_name)
    sample_list = os.listdir(sample_folder)
    sample_list = [os.path.basename(x).split(".")[0].lower() for x in sample_list]
    num_record = {}
    for apk_name in sample_list:
        call_chain_path = os.path.join(path_dict["call_chain"], folder_name, f"{apk_name}.jsonl")
        samples = []
        if os.path.exists(call_chain_path):
            with open(call_chain_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(data)
            num_record[apk_name] = len(samples)
        else:
            print(f"call_chain_path: {call_chain_path} not found")
    
    # rank from low to high
    num_record = sorted(num_record.items(), key=lambda x: x[1])
    final_sample_list = [x[0] for x in num_record]

    return final_sample_list

def make_doc_id(url: str) -> str:
    s = urlsplit(url)
    domain = s.netloc.lower().removeprefix("www.")
    last = ([p for p in s.path.split("/") if p] or ["index"])[-1]
    slug = re.sub(r'[^a-z0-9]+','-', unquote(last.lower())).strip('-')[:50] or "doc"
    h8 = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{domain}.{slug}.{h8}"


