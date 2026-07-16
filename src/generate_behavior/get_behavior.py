import argparse
import os
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from generate_behavior.ablation_behavior_utils import compute_sas_score, estimate_total_tokens, write_statistics
from prompts.behavior_prompt import BEHAVIOR_SYSTEM_PROMPT, BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING


class BehaviorExtractor:
    def __init__(self, path_dict, model_name, folder_name, gt_info_path, max_workers=4, force=False):
        self.path_dict = path_dict
        self.model_name = model_name
        self.folder_name = folder_name
        self.max_workers = max_workers
        self.force = force
        
        self.gt_info_path = gt_info_path
        if os.path.exists(self.gt_info_path) and self.folder_name != "benign":
            with open(self.gt_info_path, "r") as f:
                self.gt_info = json.load(f)
        elif self.folder_name == "benign":
            print(f"processing benign apk")
            self.gt_info = {}
        else:
            print(f"gt_info_path: {self.gt_info_path} does not exist")
            exit()

    def _build_query(self, apk_name, flag="context", top_k=300, mode="topk_removal", n_remove=0):
        if flag == "context":
            context_sumary_path = os.path.join(self.path_dict["context_summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")
        elif flag == "no_context":
            context_sumary_path = os.path.join(self.path_dict["no_context_summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")
        else:
            print(f"flag: {flag} is not valid")
            exit()
        with open(context_sumary_path, "r") as f:
            lines = f.readlines()
        function_details = [json.loads(line) for line in lines]
        function_details.sort(key=lambda x: x.get('sensitivity_score', 0), reverse=True)
        available_primary_items = len(function_details)
        
        if mode == "normal":
            function_details = function_details[:top_k]
        elif mode == "topk_removal":
            original_count = len(function_details)
            n_remove = max(1, int(original_count * 0.1))
            function_details = function_details[n_remove:]
            function_details = function_details[:top_k]
        
        evidence_lines = []
        for item in function_details:
            evidence_lines.append(
                f"Function: {item['function']}\n"
                f"  - Summary: {item['summary']}\n"
                f"  - Sensitivity Score: {item['sensitivity_score']}"
            )
        evidence_block = "\n".join(evidence_lines)

        query = f"""
                Please analyze the following function summaries from an Android APK and generate a final malware report in the specified JSON format.

                **EVIDENCE (Function Summaries, sorted by sensitivity):**
                ---
                {evidence_block}
                ---
                """

        input_stats = {
            "available_primary_items": min(available_primary_items, top_k) if mode == "normal" else len(function_details),
            "included_primary_items": len(function_details),
            "included_total_functions": len(function_details),
            "missing_from_call_chain": 0,
            "truncated_by_budget": False,
        }
        return query, input_stats

    def _get_path_folder(self, flag, mode):
        """Get the path folder based on flag and mode combination."""
        if flag == "context":
            if mode == "normal":
                return "behavior"
            elif mode == "topk_removal":
                return "topk_removal_behavior"
        elif flag == "no_context":
            if mode == "normal":
                return "no_context_behavior"
            elif mode == "topk_removal":
                return "topk_removal_no_context_behavior"
        print(f"Invalid flag: {flag} or mode: {mode}")
        exit()

    def get_behavior_single_apk(self, apk_name, flag="context", mode="normal"):
        path_folder = self._get_path_folder(flag, mode)
        output_path = os.path.join(self.path_dict[path_folder], f"{self.model_name}", self.folder_name, f"{apk_name}.json")
        if os.path.exists(output_path) and not self.force:
            return None

        query, input_stats = self._build_query(apk_name, flag, mode=mode)
        generation_start = time.perf_counter()
        if self.model_name in ["llama", "coder"]:
            system_prompt = BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING
            result, final_user_message = utils.get_llm_client(model=self.model_name)(query, system_prompt)
        else:
            system_prompt = BEHAVIOR_SYSTEM_PROMPT
            result, final_user_message = utils.get_llm_client(model=self.model_name)(query, system_prompt)
        if result is None or not isinstance(result, dict):
            print(f"[Warning] Invalid result for {apk_name}: {type(result)}")
            return None
        if self.folder_name != "benign":
            gt_type, gt_family = self.gt_info[apk_name]["type"], self.gt_info[apk_name]["family"]
        else:
            gt_type, gt_family = "benign", "benign"

        result["gt_type"] = gt_type
        result["gt_family"] = gt_family
        result["sas_score"] = compute_sas_score(result, final_user_message)
        token_stats = estimate_total_tokens(
            model_name=self.model_name,
            system_prompt=system_prompt,
            final_user_message=final_user_message,
            result=result,
        )
        result["input_stats"] = {
            **input_stats,
            **token_stats,
            "generation_seconds": time.perf_counter() - generation_start,
        }
        
        os.makedirs(os.path.join(self.path_dict[path_folder], f"{self.model_name}", self.folder_name), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    def get_behavior_all(self, apk_list, flag="context", mode="normal"):
        assert flag in ["context", "no_context"]
        assert mode in ["normal", "topk_removal"]
        
        log_dir = os.path.join(self.path_dict["info"], 'statistic')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.model_name}_{self.folder_name}_{flag}_{mode}.txt")
        
        rejected_count = 0
        count = 0

        path_folder = self._get_path_folder(flag, mode)
            
        if self.folder_name != "benign" and self.gt_info:
            apk_list_filtered = [apk for apk in apk_list if apk in self.gt_info]
            apk_list_filtered = sorted(apk_list_filtered)
            apk_list = apk_list_filtered
        
        to_skip = []
        for apk_name in apk_list:
            behavior_path = os.path.join(self.path_dict[path_folder], self.model_name, self.folder_name, f"{apk_name}.json")
            if os.path.exists(behavior_path) and not self.force:
                to_skip.append(apk_name)
        apk_list = [apk_name for apk_name in apk_list if apk_name not in to_skip]
        
        print(f"Processing {len(apk_list)} APKs (skipped {len(to_skip)} already processed)")


        failed_apks = []
        processed_stats = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_apk = {
                executor.submit(self.get_behavior_single_apk, apk_name, flag, mode): apk_name
                for apk_name in apk_list
            }

            progress_bar = tqdm(total=len(future_to_apk), desc="behavior", unit="apk")
            for future in as_completed(future_to_apk):
                apk_name = future_to_apk[future]
                result = future.result()
                count += 1
                if result is None:
                    rejected_count += 1
                    failed_apks.append(apk_name)
                else:
                    processed_stats.append(result.get("input_stats", {}))
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"done={count} failed={rejected_count} last={apk_name[:8]}")
            progress_bar.close()
            time.sleep(2)
        print(f"rejected count: {rejected_count} in {len(apk_list)}")

        write_statistics(
            log_path=Path(log_path),
            header_lines=[
                f"model: {self.model_name}",
                f"folder: {self.folder_name}",
                f"flag: {flag}",
                f"mode: {mode}",
                f"rejected_count: {rejected_count} in {len(apk_list)}",
            ],
            processed_stats=processed_stats,
            failed_apks=failed_apks,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate malware behavior reports from function summaries.")
    parser.add_argument("--model", default="deepseek", choices=["claude", "gpt", "qwen", "deepseek", "llama", "coder", "gemini"])
    parser.add_argument("--folder", default="new", choices=["malradar", "new", "benign"])
    parser.add_argument("--flag", default="context", choices=["context", "no_context"])
    parser.add_argument("--mode", default="normal", choices=["normal", "topk_removal"])
    parser.add_argument("--apk", help="Process a single APK sha256")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-generate reports even if output files already exist")
    return parser.parse_args()


def get_gt_info_path(path_dict, folder_name):
    if folder_name == "malradar":
        return os.path.join(path_dict["info"], "archived_sample_info.json")
    if folder_name == "new":
        return os.path.join(path_dict["info"], "latest_sample_info.json")
    if folder_name == "benign":
        return os.path.join(path_dict["info"], "benign_sample_info.json")
    raise ValueError(f"Unsupported folder: {folder_name}")


if __name__ == "__main__":
    args = parse_args()
    path_dict = utils.get_path_dict()
    gt_info_path = get_gt_info_path(path_dict, args.folder)
    behavior_extractor = BehaviorExtractor(
        path_dict,
        args.model,
        args.folder,
        gt_info_path,
        max_workers=args.max_workers,
        force=args.force,
    )

    if args.apk:
        behavior_extractor.get_behavior_single_apk(args.apk, flag=args.flag, mode=args.mode)
    else:
        apk_list = utils.check_file_exist(args.folder)
        behavior_extractor.get_behavior_all(apk_list, flag=args.flag, mode=args.mode)
