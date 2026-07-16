import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse

import sys
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils
from prompts import CONTEXT_SUMMARY_SYSTEM_PROMPT as SYSTEM_QUERY

class ContextFuncSummary:
    def __init__(self, model_name, folder_name, path_dict):
        self.model_name = model_name
        self.folder_name = folder_name
        self.path_dict = path_dict
        self.system_query = SYSTEM_QUERY
        self.max_workers_funcs = 4
        self.max_workers_apk = 4

    def prepare_context(self, apk_name):
        summary_path = os.path.join(self.path_dict["summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")
        context_path = os.path.join(self.path_dict["context"], self.model_name, self.folder_name, f"{apk_name}.json")
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        if os.path.exists(context_path):
            print(f"context_path: {context_path} already exists")
            return
        
        
        func_to_summary = {}
        with open(summary_path, "r") as f:
            for line in f:
                data = json.loads(line)
                func_name = data["function"]
                func_summary = data["summary"]
                if func_summary == "None":
                    print(f"func_summary is None: {func_name}")
                    continue
                func_to_summary[func_name] = func_summary


        call_chain_path = os.path.join(self.path_dict["decompile"], self.folder_name, f"{apk_name}.jsonl")
        call_to_summary = {}
        reachable_func = []
        with open(call_chain_path, "r") as f:
            for line in f:
                data = json.loads(line)
                func_name = data["func"]
                func_src = data["source"]
                caller_list = data["callers"]
                caller_name = [x["name"] for x in caller_list]
                caller_src = [x["source"] for x in caller_list]

                caller_summary = {}
                for caller, caller_src in zip(caller_name, caller_src):
                    if 'Failed to decompile' in caller_src:
                        # print(f"Failed to decompile: {caller}")
                        continue
                    if caller not in func_to_summary:
                        print(f"caller not in func_to_summary: {caller}")
                        continue
                    caller_summary[caller] = func_to_summary[caller]

                callee_list = data["callees"]
                callee_name = [x["name"] for x in callee_list]
                callee_src = [x["source"] for x in callee_list]



                callee_summary = {}
                for callee, callee_src in zip(callee_name, callee_src):
                    if 'Failed to decompile' in callee_src:
                        # print(f"Failed to decompile: {callee}")
                        continue
                    if callee not in func_to_summary:
                        print(f"callee not in func_to_summary: {callee}")
                        continue
                    callee_summary[callee] = func_to_summary[callee]

                call_to_summary[func_name] = {
                    "function": func_name,
                    "source": func_src,
                    "callers": [{"name": n, "summary": s} for n, s in caller_summary.items()],
                    "callees": [{"name": n, "summary": s} for n, s in callee_summary.items()]
                }
                reachable_func.append(func_name)

        assert len(call_to_summary) == len(reachable_func)

        with open(context_path, "w") as f:
            json.dump(call_to_summary, f, ensure_ascii=False, indent=4)
        

    
    def _build_query(self, func_name, func_src, callers, callees):
        """
        Builds a clear and concise user prompt for the LLM, providing full context
        for function analysis.
        """
        prompt_parts = [
            f"Please analyze the following function based on the comprehensive context provided.",
            "",
            f"### Function to Analyze: `{func_name}`",
            "",
            "#### Source Code:",
            "```java",
            func_src.strip(),
            "```",
            ""
        ]

        # --- Caller Information ---
        prompt_parts.append("### Context: Callers (Functions that call this function)")
        if not callers:
            prompt_parts.append("- None provided.")
        else:
            # Using a more readable list format
            caller_details = []
            for caller in callers:
                caller_details.append(
                    f"- **Name:** `{caller.get('name', 'N/A')}`\n"
                    f"  - **Summary:** {caller.get('summary', 'No summary available.')}"
                )
            prompt_parts.append("\n".join(caller_details))
        prompt_parts.append("")

        # --- Callee Information ---
        prompt_parts.append("### Context: Callees (Functions called by this function)")
        if not callees:
            prompt_parts.append("- None provided.")
        else:
            callee_details = []
            for callee in callees:
                callee_details.append(
                    f"- **Name:** `{callee.get('name', 'N/A')}`\n"
                    f"  - **Summary:** {callee.get('summary', 'No summary available.')}"
                )
            prompt_parts.append("\n".join(callee_details))
        prompt_parts.append("")

        # --- Final Instruction ---
        prompt_parts.append("---")
        prompt_parts.append("Based on all the information above (the function's own code and the context from its callers and callees), provide your analysis in the required JSON format.")

        return "\n".join(prompt_parts)

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
            return obj
        except Exception:
            return None

    
    def _summary_one(self, func_name, func_src, callers, callees):
        query = self._build_query(func_name, func_src, callers, callees)
        response, final_user_message = utils.get_llm_client(model=self.model_name)(query, self.system_query)

        obj = self._parse_or_none(response, func_name)
        output = {'function': func_name, 'summary': obj['summary'], 'sensitivity_score': obj['sensitivity_score'], 'reasoning': obj['reasoning']}
        return output


    def get_context_apk(self, apk_name):
        context_path = os.path.join(self.path_dict["context"], self.model_name,self.folder_name,f"{apk_name}.json")
        output_path = os.path.join(self.path_dict["context_summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(context_path):
            self.prepare_context(apk_name)

        try:
            with open(context_path, "r", encoding="utf-8") as f:
                context_data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Context source file not found for {apk_name} at {context_path}. Skipping.")
            
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

        for func_name, func_data in context_data.items():
            if func_name not in existing_results:
                tasks_to_process[func_name] = func_data

        if not tasks_to_process:
            print(f"All {len(context_data)} functions for {apk_name} are already analyzed correctly. Skipping.")
            return

        print(f"Found {len(existing_results)} existing valid analyses for {apk_name}. "
            f"Analyzing {len(tasks_to_process)} new/failed functions.")

        newly_completed_results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers_funcs) as ex:
            future_to_func = {
                ex.submit(self._summary_one, func_name, data["source"], data["callers"], data["callees"]): func_name
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
                    obj = {
                        "function": func_name,
                        "summary": f"error: {e}",
                        "sensitivity_score": -1,
                        "reasoning": "Processing failed"
                    }
                    print(f"error obj: {obj}")
                    newly_completed_results[func_name] = obj

        print(f"Finished processing for {apk_name}. Merging and writing results...")

        final_results = existing_results.copy()
        final_results.update(newly_completed_results)

        temp_output_path = output_path + ".tmp"
        
        try:
            with open(temp_output_path, "w", encoding="utf-8") as f_out:
                for func_name in context_data.keys():
                    if func_name in final_results:
                        f_out.write(json.dumps(final_results[func_name], ensure_ascii=False) + "\n")
            
            os.replace(temp_output_path, output_path)
            print(f"Successfully wrote {len(final_results)} unique records to {output_path}")

        except Exception as e:
            print(f"[FATAL] Failed to write final results for {apk_name}: {e}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

    def get_context_summary_all(self,apk_list):
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = [ex.submit(self.get_context_apk, apk_name) for apk_name in tqdm(apk_list)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="get_context_summary_all"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate context-aware sensitivity summaries from released call-chain and function artifacts.")
    parser.add_argument("--model", default="gpt", choices=["gemini", "gpt", "claude", "qwen", "deepseek", "llama", "coder"])
    parser.add_argument("--folder", default="benign", choices=["malradar", "new", "benign"])
    parser.add_argument("--apk", help="Process a single APK sha256")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_dict = utils.get_path_dict()
    context_summary = ContextFuncSummary(args.model, args.folder, path_dict)
    if args.apk:
        context_summary.get_context_apk(args.apk)
    else:
        apk_list = utils.check_file_exist(args.folder)
        context_summary.get_context_summary_all(apk_list)
    
