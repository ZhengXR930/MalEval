import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
import re


SYSTEM_QUERY = """
You are an expert Android security analyst and reverse engineer.
Your goal is to analyze a function from a decompiled Android APK and produce a structured JSON output containing two distinct parts:
1.  An objective, factual `summary` of the function's purpose.
2.  A `sensitivity_score` (0-10) that quantifies the sensitivity of the operations performed, along with a brief `reasoning` for that score.
## Input
You will be given a single function's name and code, its 1-hop callers and callees.
## Output Format
You MUST provide your final answer ONLY in the following JSON format. Do not output any text before or after this JSON block.
{
  "summary": "A concise, factual description of what the function does.",
  "sensitivity_score": <An integer from 0 to 10>,
  "reasoning": "A brief explanation for why this score was given, referencing the sensitive operations."
}

## Contextual Analysis Guidance
**Callers**: Analyze the summaries of the functions that call this one (`Callers`) to understand the potential origin and state of the data being passed into the function you are analyzing.
**Callees**: Analyze the summaries of the functions called by this one (`Callees`) to understand the ultimate destination or impact of the function's operations. This is crucial for determining the full scope of its actions.

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
        with open(context_path, "r") as f:
            context_data = json.load(f)

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

if __name__ == "__main__":
    model_name = "llama"
    folder_name = "benign"
    path_dict = utils.get_path_dict()
    context_summary = ContextFuncSummary(model_name, folder_name, path_dict)
    apk_list = utils.check_file_exist(folder_name)
    context_summary.get_context_summary_all(apk_list)
    
