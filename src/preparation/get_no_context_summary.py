import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
import re

path_dict = utils.get_path_dict()


SYSTEM_QUERY = """
You are an expert Android security analyst and reverse engineer.
Your goal is to analyze a function from a decompiled Android APK and produce a structured JSON output containing two distinct parts:
1.  An objective, factual `summary` of the function's purpose.
2.  A `sensitivity_score` (0-10) that quantifies the sensitivity of the operations performed, along with a brief `reasoning` for that score.
## Input
You will be given a single function's name and code.
## Output Format
You MUST provide your final answer ONLY in the following JSON format. Do not output any text before or after this JSON block.
{
  "summary": "A concise, factual description of what the function does.",
  "sensitivity_score": <An integer from 0 to 10>,
  "reasoning": "A brief explanation for why this score was given, referencing the sensitive operations."
}

## Task 1: How to write the `summary`
This part must be strictly objective and descriptive. Follow these rules:
- **WHAT, not WHY or HOW BAD**: Describe exactly WHAT the function does (e.g., "Reads contact list and sends it over an HTTP connection"). Do NOT speculate on intent (e.g., "Steals user data").
- **No Judgmental Words**: Avoid any words related to risk, security, privacy, or malice (e.g., malicious, dangerous, suspicious, leak, spyware).
- **Implementation-Agnostic**: Focus on the high-level outcome. Avoid detailing trivial programming steps (e.g., "initializes a variable, enters a loop, calls a library function").

## Special Rule for Native Functions
**If the function's source code contains the `native` keyword, its implementation is hidden. In this specific case, you MUST follow these steps:**
1.  Your `summary` **MUST** begin with the exact phrase "This is a native function."
2.  After that, infer its most likely purpose based **ONLY** on its name, its parameters, and the context from its callers.
3.  **Example**: For a function named `native void interface5()` called by `load()`, a good summary is: "This is a native function. Given its generic name and being called from a 'load' function, it likely performs a native-level initialization or check required by the application's framework."

## Task 2: How to determine the `sensitivity_score` and `reasoning`
This part requires your expert judgment to quantify how sensitive the function's actions are.

**Definition of "Sensitivity"**: Sensitivity measures the potential for a function to impact user privacy, system security, or device resources. A legitimate backup function and a malicious spyware function can both have a high sensitivity score.
**Scoring Rubric (0-10):**
- **0-1 (None/Minimal)**: The function performs harmless operations, like internal calculations, simple string manipulation, or basic UI updates.
- **2-4 (Low)**: The function interacts with non-critical device resources. Examples: reading device model, checking network status, accessing public storage directories.
- **5-7 (Moderate)**: The function performs clearly sensitive actions that require user permission. Examples: initiating a network connection, accessing the camera or microphone, reading the calendar, or getting coarse location data.
- **8-10 (High/Critical)**: The function performs actions with significant privacy or security implications. Examples: reading contacts/SMS messages, getting precise location, using accessibility services, installing/deleting other apps, performing cryptographic operations on user data, sending user data to a remote server.
**`reasoning` Field**: In one sentence, explain which specific operations justify the score you assigned. Example: "Reasoning: The function accesses the user's contact list and sends data over the network."
"""

class NoContextFuncSummary:
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
    
    def _build_query(self, func_name, func_src):
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

        # --- Final Instruction ---
        prompt_parts.append("---")
        prompt_parts.append("Based on all the information above (the function's own code), provide your analysis in the required JSON format.")

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
    

    def _summary_one(self, func_name, func_src):
        query = self._build_query(func_name, func_src)
        response, final_user_message = utils.get_llm_client(model=self.model_name)(query, self.system_query)

        obj = self._parse_or_none(response, func_name)
        output = {'function': func_name, 'summary': obj['summary'], 'sensitivity_score': obj['sensitivity_score'], 'reasoning': obj['reasoning']}
        return output


    def get_single_func_apk(self, apk_name):
        function_path = os.path.join(self.path_dict["reachable_func"], self.folder_name, f"{apk_name}.json")
        output_path = os.path.join(self.path_dict["no_context_summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")

        if os.path.exists(output_path):
            print(f"no context summary for {apk_name} already exists")
            return

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

        print(f"Found {len(existing_results)} existing valid analyses for {apk_name}. "
            f"Analyzing {len(tasks_to_process)} new/failed functions.")

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
                    obj = {
                        "function": func_name,
                        "summary": f"error: {e}",
                        "sensitivity_score": -1,
                        "reasoning": "Processing failed"
                    }
                    newly_completed_results[func_name] = obj

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


    def get_no_context_summary_all(self,apk_list):
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = [ex.submit(self.get_single_func_apk, apk_name) for apk_name in tqdm(apk_list)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="get_no_context_summary_all"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"error: {e}")

if __name__ == "__main__":
    model_name = "llama"
    folder_name = "benign"
    path_dict = utils.get_path_dict()
    no_context_summary = NoContextFuncSummary(model_name, folder_name, path_dict)
    apk_list = utils.check_file_exist(folder_name)
    no_context_summary.get_no_context_summary_all(apk_list)