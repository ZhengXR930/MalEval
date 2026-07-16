import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse
import shutil
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils
from prompts import NO_CONTEXT_SUMMARY_SYSTEM_PROMPT as SYSTEM_QUERY


class NoContextFuncSummary:
    def __init__(self, model_name, folder_name, path_dict):
        self.model_name = model_name
        self.folder_name = folder_name
        self.path_dict = path_dict
        self.system_query = SYSTEM_QUERY
        self.max_workers_funcs = 4
        self.max_workers_apk = 4

    def _ensure_reachable_func_file(self, apk_name):
        output_path = os.path.join(self.path_dict["reachable_func"], self.folder_name, f"{apk_name}.json")
        if os.path.exists(output_path):
            return output_path

        released_path = os.path.join(self.path_dict["artifacts"], "reachable_func", self.folder_name, f"{apk_name}.json")
        if os.path.exists(released_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(released_path, output_path)
            return output_path

        self._get_func_for_each_apk(apk_name)
        return output_path

    def _get_func_for_each_apk(self, apk_name):
        all_samples = []
        call_chain_file = os.path.join(self.path_dict["decompile"], self.folder_name, f"{apk_name}.jsonl")
        with open(call_chain_file, "r", encoding="utf-8") as f:
            for line in f:
                func_data = json.loads(line)
                all_samples.append(func_data)

        func_dict = {}
        reachable_func_dict = {}

        for sample in all_samples:
            func_name = sample["func"]
            func_src = sample["source"]
            caller_list = sample["callers"]
            callee_list = sample["callees"]
            caller_name = [x["name"] for x in caller_list]
            caller_src = [x["source"] for x in caller_list]
            callee_name = [x["name"] for x in callee_list]
            callee_src = [x["source"] for x in callee_list]

            if "Failed to decompile" in func_src:
                continue

            func_dict[func_name] = func_src
            reachable_func_dict[func_name] = func_src

            for name, src in zip(caller_name, caller_src):
                if "Failed to decompile" in src:
                    continue
                func_dict[name] = src

            for name, src in zip(callee_name, callee_src):
                if "Failed to decompile" in src:
                    continue
                func_dict[name] = src

        os.makedirs(os.path.join(self.path_dict["functions"], self.folder_name), exist_ok=True)
        os.makedirs(os.path.join(self.path_dict["reachable_func"], self.folder_name), exist_ok=True)
        with open(os.path.join(self.path_dict["functions"], self.folder_name, f"{apk_name}.json"), "w", encoding="utf-8") as f:
            json.dump(func_dict, f)
        with open(os.path.join(self.path_dict["reachable_func"], self.folder_name, f"{apk_name}.json"), "w", encoding="utf-8") as f:
            json.dump(reachable_func_dict, f)

    def _build_query(self, func_name, func_src):
        prompt_parts = [
            "Please analyze the following function based on the comprehensive context provided.",
            "",
            f"### Function to Analyze: `{func_name}`",
            "",
            "#### Source Code:",
            "```java",
            func_src.strip(),
            "```",
            "",
        ]
        prompt_parts.append("---")
        prompt_parts.append(
            "Based on all the information above (the function's own code), provide your analysis in the required JSON format."
        )
        return "\n".join(prompt_parts)

    def _parse_or_none(self, raw, func_name):
        try:
            if isinstance(raw, str):
                match = re.search(r"\{[\s\S]+\}", raw)
                if not match:
                    return None
                raw = match.group(0)
                obj = json.loads(raw)
            else:
                obj = raw

            if "function" not in obj:
                obj["function"] = func_name
            return obj
        except Exception:
            return None

    def _summary_one(self, func_name, func_src):
        query = self._build_query(func_name, func_src)
        response, _ = utils.get_llm_client(model=self.model_name)(query, self.system_query)

        obj = self._parse_or_none(response, func_name)
        if obj is None:
            return {
                "function": func_name,
                "summary": "None",
                "sensitivity_score": -1,
                "reasoning": "Failed to parse model output",
            }

        return {
            "function": func_name,
            "summary": obj.get("summary", "None"),
            "sensitivity_score": obj.get("sensitivity_score", -1),
            "reasoning": obj.get("reasoning", ""),
        }

    def get_single_func_apk(self, apk_name):
        function_path = self._ensure_reachable_func_file(apk_name)
        output_path = os.path.join(self.path_dict["no_context_summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(function_path, "r", encoding="utf-8") as f:
                function_data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Context source file not found for {apk_name} at {function_path}. Skipping.")
            return

        existing_results = {}
        tasks_to_process = {}

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    try:
                        summary_obj = json.loads(line)
                        func_name = summary_obj.get("function")
                        if not func_name:
                            continue
                        if summary_obj.get("sensitivity_score", -1) != -1:
                            existing_results[func_name] = summary_obj
                    except (json.JSONDecodeError, AttributeError):
                        print(f"[WARNING] Skipping corrupted line in {output_path}: {line.strip()}")

        for func_name, func_data in function_data.items():
            if func_name not in existing_results:
                tasks_to_process[func_name] = func_data

        if not tasks_to_process:
            print(f"All {len(function_data)} functions for {apk_name} are already analyzed correctly. Skipping.")
            return

        print(
            f"Found {len(existing_results)} existing valid analyses for {apk_name}. "
            f"Analyzing {len(tasks_to_process)} new/failed functions."
        )

        newly_completed_results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers_funcs) as ex:
            future_to_func = {
                ex.submit(self._summary_one, func_name, data): func_name
                for func_name, data in tasks_to_process.items()
            }

            progress_bar = tqdm(as_completed(future_to_func), total=len(future_to_func), desc=f"Analyzing {apk_name}")
            for fut in progress_bar:
                func_name = future_to_func[fut]
                try:
                    obj = fut.result()
                    newly_completed_results[func_name] = obj
                except Exception as e:
                    print(f"\n[ERROR] Analysis failed for {func_name}: {e}")
                    newly_completed_results[func_name] = {
                        "function": func_name,
                        "summary": f"error: {e}",
                        "sensitivity_score": -1,
                        "reasoning": "Processing failed",
                    }

        print(f"Finished processing for {apk_name}. Merging and writing results...")

        final_results = existing_results.copy()
        final_results.update(newly_completed_results)
        temp_output_path = output_path + ".tmp"

        try:
            with open(temp_output_path, "w", encoding="utf-8") as f_out:
                for func_name in function_data.keys():
                    if func_name in final_results:
                        f_out.write(json.dumps(final_results[func_name], ensure_ascii=False) + "\n")

            os.replace(temp_output_path, output_path)
            print(f"Successfully wrote {len(final_results)} unique records to {output_path}")
        except Exception as e:
            print(f"[FATAL] Failed to write final results for {apk_name}: {e}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

    def get_no_context_summary_all(self, apk_list):
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = [ex.submit(self.get_single_func_apk, apk_name) for apk_name in tqdm(apk_list)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="get_no_context_summary_all"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate no-context sensitivity summaries from released function artifacts."
    )
    parser.add_argument(
        "--model",
        default="gpt",
        choices=["gemini", "gpt", "claude", "qwen", "deepseek", "llama", "coder"],
    )
    parser.add_argument(
        "--folder",
        default="benign",
        choices=["malradar", "new", "benign"],
    )
    parser.add_argument("--apk", help="Process a single APK sha256")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_dict = utils.get_path_dict()
    no_context_summary = NoContextFuncSummary(args.model, args.folder, path_dict)
    if args.apk:
        no_context_summary.get_single_func_apk(args.apk)
    else:
        apk_list = utils.check_file_exist(args.folder)
        no_context_summary.get_no_context_summary_all(apk_list)
