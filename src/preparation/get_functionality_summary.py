import os
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import utils
import re



SYSTEM_QUERY = """
You are a code analyst. Your job is to write concise, neutral functionality summaries for individual functions from decompiled Android apps. 
Rules:
- Describe WHAT the function does, not whether it is good/bad.
- Do NOT speculate about security, privacy, or “maliciousness”. No risk words (e.g., malicious, dangerous).
- Keep it self-contained and implementation-agnostic (avoid repeating trivial low-level steps).
- Output ONLY the final answer in the requested JSON format. Do not include reasoning. output ONLY JSON in format:
{"function": "<function signature>", "summary": "<summary>"}
"""


class SingleFuncSummary:
    def __init__(self, model_name, folder_name, path_dict):
        self.model_name = model_name
        self.folder_name = folder_name
        self.path_dict = path_dict
        self.system_query = SYSTEM_QUERY
        self.max_workers_funcs = 4
        self.max_workers_apk = 4


    def _get_func_for_each_apk(self, apk_name):
        all_samples = []
        call_chain_file = os.path.join(self.path_dict["decompile"], self.folder_name, f"{apk_name}.jsonl")
        with open(call_chain_file, "r") as f:
            for line in f:
                func_data = json.loads(line)
                all_samples.append(func_data)

        func_dict = {}
        reachable_func_dict = {}
        context_func_dict = {}

        for sample in all_samples:
            func_name = sample["func"]
            func_src = sample["source"]
            caller_list = sample["callers"]
            callee_list = sample["callees"]
            caller_name = [x["name"] for x in caller_list]
            caller_src = [x["source"] for x in caller_list]
            callee_name = [x["name"] for x in callee_list]
            callee_src = [x["source"] for x in callee_list]


            if 'Failed to decompile' in func_src:
                continue

            func_dict[func_name] = func_src
            reachable_func_dict[func_name] = func_src
            for name, src in zip(caller_name, caller_src):
                if 'Failed to decompile' in src:
                    continue
                func_dict[name] = src
                context_func_dict[name] = src
            for name, src in zip(callee_name, callee_src):
                if 'Failed to decompile' in src:
                    continue
                func_dict[name] = src
                context_func_dict[name] = src
        
        os.makedirs(os.path.join(self.path_dict["functions"], self.folder_name), exist_ok=True)
        os.makedirs(os.path.join(self.path_dict["reachable_func"], self.folder_name), exist_ok=True)
        with open(os.path.join(self.path_dict["functions"], self.folder_name, f"{apk_name}.json"), "w") as f:
            json.dump(func_dict, f)
        with open(os.path.join(self.path_dict["reachable_func"], self.folder_name, f"{apk_name}.json"), "w") as f:
            json.dump(reachable_func_dict, f)

    def _parse_or_none(self, raw, func_name):
        try:
            if isinstance(raw, str):
                match = re.search(r'\{[\s\S]+\}', raw)
                if not match:
                    return None
                raw = match.group(0)
                obj = json.loads(raw)
            else:
                obj = raw

            if "function" not in obj or not obj["function"]:
                obj["function"] = func_name
            if "summary" in obj:
                obj["summary"] = (obj["summary"] or "").strip()
            return obj
        except Exception:
            return None


    def _summarize_one(self, func_name, func_src):
        query = self._build_query(func_name, func_src)
        result, _ = utils.get_llm_client(model=self.model_name)(query, self.system_query)
        parsed = self._parse_or_none(result, func_name)

        if (parsed is None) or (not parsed.get("summary")):
            repair_hint = (
                '\n\nRead carefully on function name and source code, return ONLY valid JSON like: '
                f'{{"function":"{func_name}","summary":"<1-2 sentences>"}}'
            )
            result, _ = utils.get_llm_client(model=model_name)(query + repair_hint, self.system_query)
            parsed = self._parse_or_none(result, func_name)

            if parsed is None:
                parsed = {"function": func_name, "summary": "None"}
        out_obj = {"function": func_name, "summary": parsed.get("summary", "None").strip()}
        return out_obj


    def _get_summary_single_func(self,apk_name):
        function_file_path = os.path.join(self.path_dict["functions"], self.folder_name, f"{apk_name}.json")
        output_path = os.path.join(self.path_dict["summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(function_file_path, "r", encoding="utf-8") as f:
            func_dict = json.load(f)

        processed_functions = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    try:
                        summary_obj = json.loads(line)
                        # Add the function name to our set of completed tasks
                        if "function" in summary_obj and summary_obj["summary"] != "None":
                            processed_functions.add(summary_obj["function"])
                    except json.JSONDecodeError:
                        print(f"[WARNING] Skipping corrupted line in {output_path}: {line.strip()}")

        tasks_to_process = []
        for func_name, func_src in func_dict.items():
            if func_name not in processed_functions:
                tasks_to_process.append((func_name, func_src))


        if not tasks_to_process:
            print(f"All {len(func_dict)} functions for {apk_name} are already summarized. Skipping.")
            return output_path


        print(f"Found {len(processed_functions)} existing summaries for {apk_name}. "
            f"Summarizing {len(tasks_to_process)} new functions.")

        with ThreadPoolExecutor(max_workers=self.max_workers_funcs) as ex, \
            open(output_path, "a", encoding="utf-8") as fout: # Use "a" (append) instead of "w" (write)

            futures = [ex.submit(self._summarize_one, fn, src) for fn, src in tasks_to_process]

            for fut in tqdm(as_completed(futures), total=len(tasks_to_process), desc=f"Summarizing {apk_name}"):
                try:
                    obj = fut.result()
                    if obj:
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"\n[ERROR] A future failed for {apk_name}: {e}")


    def _build_query(self, func_name, func_src):
        json_template = {
            "function": func_name,
            "summary": "<one or two sentences describing functionality>",
        }
        return (
            "You are a code analyst. Your job is to write concise, neutral functionality summaries for individual functions from decompiled Android apps.\n"
            "Do NOT judge security; 1–2 sentences; output ONLY JSON.\n\n"
            f"Function Name: {func_name}\n\n"
            f"Source Code:\n{func_src}\n\n"
            f"Return your answer as JSON:\n{json.dumps(json_template, ensure_ascii=False)}"
        )

    def query_summary(self, apk_names):
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = {ex.submit(self._get_summary_single_func, name): name for name in apk_names}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"APK-SUMMARY for {self.model_name}"):
                name = futures[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    print(f"[ERR] {name}: {e}")
        end_time = time.time()
        print(f"time cost: {end_time - start_time} seconds")

    def _retry_none_summary_single_func(self, apk_name):
        summary_path = os.path.join(self.path_dict["summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")
        function_dict_path = os.path.join(self.path_dict["functions"], self.folder_name, f"{apk_name}.json")

        try:
            with open(function_dict_path, "r") as f:
                func_src_dict = json.load(f)
            
            with open(summary_path, "r") as f:
                original_data = [json.loads(line) for line in f]
                
        except FileNotFoundError as e:
            return {"apk": apk_name, "status": "error", "message": f"File not found: {e}"}
        tasks_to_retry = []
        for index, func_data in enumerate(original_data):
            if func_data.get("summary") == "None":
                func_name = func_data["function"]
                src = func_src_dict.get(func_name, "")
                if 'Failed to decompile' not in src:
                    tasks_to_retry.append((index, func_data))

        if not tasks_to_retry:
            return {"apk": apk_name, "status": "skipped", "message": "No 'None' summaries to retry."}

        retried_count = 0
        still_none_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as executor:
            future_to_index = {}

            llm_client = utils.get_llm_client(model=self.model_name)
            for index, func_data in tasks_to_retry:
                func_name = func_data["function"]
                src = func_src_dict.get(func_name, "")
                query = self._build_query(func_name, src)
                repair_hint = (
                    '\n\nPlease read carefully on function name and source code, try your best to understand the functionality of the function, return ONLY valid JSON like: '
                    f'{{"function":"{func_name}","summary":"<1-2 sentences>"}}'
                )
                
                future = executor.submit(llm_client, query + repair_hint, self.system_query)
                future_to_index[future] = index

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"Retrying 'None' Summaries for {apk_name}"):
                original_index = future_to_index[future]
                func_name = original_data[original_index]["function"]
                try:
                    result_text, _ = future.result()
                    parsed = self._parse_or_none(result_text, func_name)
                    if parsed is None:
                        parsed = {"function": func_name, "summary": "None"}
                    new_summary = parsed.get("summary", "None").strip()
                    original_data[original_index]["summary"] = new_summary
                    if new_summary != "None":
                        retried_count += 1
                except Exception as e:
                    still_none_count += 1
                    print(f"\n[ERROR] Failed to retry summary for {func_name} in {apk_name}: {e}")
                    original_data[original_index]["summary"] = "None" 

        with open(summary_path, "w") as f:
            for new_func_data in original_data:
                f.write(json.dumps(new_func_data, ensure_ascii=False) + "\n")
                
        return {"apk": apk_name, "status": "success", "retried_count": retried_count, "still_none_count": still_none_count}

    def retry_none_summary(self,apk_list):
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as executor:
            future_to_apk = {executor.submit(self._retry_none_summary_single_func, apk): apk for apk in apk_list}

            progress_bar = tqdm(as_completed(future_to_apk), total=len(apk_list), desc="Retrying 'None' Summaries for APKs")
            for future in progress_bar:
                apk_name = future_to_apk[future]
                try:
                    result = future.result()
                    results.append(result)
                    progress_bar.set_postfix(last_apk=apk_name, status=result.get('status'))
                except Exception as e:
                    print(f"\n[FATAL] An unexpected error occurred while processing {apk_name}: {e}")
                    results.append({"apk": apk_name, "status": "fatal_error", "message": str(e)})
        print(results)
        print("\n--- Retry process finished. Summary of results: ---")

    def get_single_func_summary(self, apk_list):
        for apk_name in tqdm(apk_list, desc=f"Prepare function dict APKs for {self.model_name}"):
            self._get_func_for_each_apk(apk_name)
        self.query_summary(apk_list)
        self.retry_none_summary(apk_list)


if __name__ == "__main__":
    model_name = "llama"
    folder_name = "benign"
    path_dict = utils.get_path_dict()
    functionality_summary = SingleFuncSummary(model_name, folder_name, path_dict)
    apk_list = utils.check_file_exist(folder_name)
    functionality_summary.get_single_func_summary(apk_list)





