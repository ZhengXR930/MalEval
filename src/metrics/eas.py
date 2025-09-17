import os
import json
from tqdm import tqdm
from collections import defaultdict
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
from collections import defaultdict

SYSTEM_PROMPT = """
You are an expert Android app security analyst. 
Your task is to evaluate whether the provided function summaries truly support the claimed behaviors detected in an Android app.

For each behavior, you'll receive:
1. The behavior description
2. The confidence level in this behavior
3. The evidence supporting this behavior
4. Related functions with their summaries, sensitivity scores, and reasoning

You must output STRICTLY in valid JSON format. 
Do not include any text outside the JSON.

The output must be a JSON object with the following structure:

{
  "behaviors": [
    {
      "behavior": "<string, the behavior description>",
      "support_score": <float, between 0 and 1>,
      "reasoning": "<string, brief explanation of reasoning>"
    },
    ...
  ],
  "overall_score": <float, between 0 and 1>
}

Scoring Standards:
- 0.0 – 0.3 → Not supported: functions do not show capability for this behavior.
- 0.3 – 0.6 → Partially supported: some weak or indirect evidence, but not conclusive.
- 0.6 – 1.0 → Fully supported: strong and direct evidence in function summaries.
- `support_score` should reflect how strongly the functions support the claimed behavior, not how confident the model is in its own judgment.
- `overall_score` should be the average of all behavior support scores.

Rules:
- Do NOT add fields not listed above.
- Do NOT include comments or explanations outside of the JSON.
- Ensure the JSON is syntactically correct and parsable.
"""


class EvidenceCorrectness:
    def __init__(self, path_dict, model_name, folder_name, max_workers=24):
        self.path_dict = path_dict
        self.model_name = model_name
        self.folder_name = folder_name
        self.max_workers = max_workers


    def _build_query(self, apk_name, flag):
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")
        
        context_summary_path = os.path.join(self.path_dict["context_summary"], self.model_name, self.folder_name, f"{apk_name}.jsonl")
        context_summary_map = defaultdict(dict)
        with open(context_summary_path, "r") as f:
            for line in f:
                data = json.loads(line)
                context_summary_map[data["function"]]["summary"] = data["summary"]
                context_summary_map[data["function"]]["sensitivity_score"] = data["sensitivity_score"]
                context_summary_map[data["function"]]["reasoning"] = data["reasoning"]
        
        report_path = os.path.join(self.path_dict[path_folder], self.model_name, self.folder_name, f"{apk_name}.json")
        with open(report_path, "r") as f:
            report_data = json.load(f)

        report_behavior_list = []
        related_functions = []
        for behavior in report_data["present_behaviors"]:
            report_behavior_list.append({
                "behavior": behavior["behavior"],
                "confidence": behavior["confidence"],
                "evidence": behavior["evidence"],
            })
            related_functions.append(behavior["related_functions"])

        evidence_lines = []
        for behavior_info, function_names in zip(report_behavior_list, related_functions):
            evidence_lines.append(f"Behavior: {behavior_info['behavior']}\n")
            evidence_lines.append(f"  - Confidence: {behavior_info['confidence']}\n")
            evidence_lines.append(f"  - Evidence: {behavior_info['evidence']}\n")
            evidence_lines.append(f"Related Functions with their summaries:\n")
            for function_name in function_names:
                if function_name in context_summary_map:
                    evidence_lines.append(f"  - Function: {function_name}\n")
                    evidence_lines.append(f"    - Summary: {context_summary_map[function_name]['summary']}\n")
                    evidence_lines.append(f"    - Sensitivity Score: {context_summary_map[function_name]['sensitivity_score']}\n")
                    evidence_lines.append(f"    - Reasoning: {context_summary_map[function_name]['reasoning']}\n")
        evidence_block = "\n".join(evidence_lines)

        query = f"""
        Please evaluate whether the AI model's generated behavior matches the function it identified to support.

        **EVIDENCE (Behavior, Function Summaries, sorted by sensitivity):**
        ---
        {evidence_block}
        ---
        Please follow the schema in the system prompt and output only valid JSON.
        """
        return query

    def _build_src_query(self, apk_name, flag):
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")
        
        context_path = os.path.join(self.path_dict["context"], self.model_name, self.folder_name, f"{apk_name}.json")
        with open(context_path, "r") as f:
            context_data = json.load(f)
            
        
        report_path = os.path.join(self.path_dict[path_folder], self.model_name, self.folder_name, f"{apk_name}.json")
        with open(report_path, "r") as f:
            report_data = json.load(f)

        report_behavior_list = []
        related_functions = []
        for behavior in report_data["present_behaviors"]:
            report_behavior_list.append({
                "behavior": behavior["behavior"],
                "confidence": behavior["confidence"],
                "evidence": behavior["evidence"],
            })
            related_functions.append(behavior["related_functions"])

        evidence_lines = []
        for behavior_info, function_names in zip(report_behavior_list, related_functions):
            evidence_lines.append(f"Behavior: {behavior_info['behavior']}\n")
            evidence_lines.append(f"  - Confidence: {behavior_info['confidence']}\n")
            evidence_lines.append(f"  - Evidence: {behavior_info['evidence']}\n")
            evidence_lines.append(f"Related Functions with their summaries:\n")
            for function_name in function_names:
                if function_name in context_data:
                    evidence_lines.append(f"  - Function: {function_name}\n")
                    evidence_lines.append(f"    - Source Code: {context_data[function_name]['source']}\n")

                    # --- Caller Information ---
                    evidence_lines.append("### Context: Callers (Functions that call this function)")
                    if not context_data[function_name]['callers']:
                        # evidence_lines.append("- None provided.")
                        continue
                    else:
                        # Using a more readable list format
                        caller_details = []
                        for caller in context_data[function_name]['callers']:
                            caller_details.append(
                                f"- **Name:** `{caller.get('name', 'N/A')}`\n"
                                f"  - **Summary:** {caller.get('summary', 'No summary available.')}"
                            )
                        evidence_lines.append("\n".join(caller_details))
                    evidence_lines.append("")

                    # --- Callee Information ---
                    evidence_lines.append("### Context: Callees (Functions called by this function)")
                    if not context_data[function_name]['callees']:
                        # evidence_lines.append("- None provided.")
                        continue
                    else:
                        callee_details = []
                        for callee in context_data[function_name]['callees']:
                            callee_details.append(
                                f"- **Name:** `{callee.get('name', 'N/A')}`\n"
                                f"  - **Summary:** {callee.get('summary', 'No summary available.')}"
                            )
                        evidence_lines.append("\n".join(callee_details))
                    evidence_lines.append("")

        evidence_block = "\n".join(evidence_lines)

        query = f"""
        Please evaluate whether the AI model's generated behavior matches the function it identified to support.

        **EVIDENCE (Behavior, Function Source Code, Function Callers, Function Callees):**
        ---
        {evidence_block}
        ---
        Please follow the schema in the system prompt and output only valid JSON.
        """
        return query


    def evaluate_single_apk(self, apk_name, flag):
        query = self._build_query(apk_name, flag)
        # query = self._build_src_query(apk_name, flag)
        result, _ = utils.get_llm_client(model="gpt")(query, SYSTEM_PROMPT) # use gpt-5-mini

        try:
            behavior_scores = [b["support_score"] for b in result.get("behaviors", [])]
            overall_score = result.get("overall_score", None)

            final_result = {
                "apk": apk_name,
                "behaviors": result.get("behaviors", []),
                "overall_score": overall_score,
                "avg_score": sum(behavior_scores) / len(behavior_scores) if behavior_scores else 0.0
            }

            save_folder = os.path.join(self.path_dict["result"], "eas", f"{self.model_name}_{flag}", self.folder_name)
            os.makedirs(save_folder, exist_ok=True)
            with open(os.path.join(save_folder, f"{apk_name}.json"), "w") as f:
                json.dump(final_result, f, indent=4)
        except Exception as e:
            print(f"Error evaluating {apk_name}: {e}")
            return None

        return final_result
    
    def evaluate_multiple_apks(self, apk_list, flag):

        save_folder = os.path.join(self.path_dict["result"], "eas", f"{self.model_name}_{flag}", self.folder_name)
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")

        to_remove_apk_list = []
        for apk in apk_list:
            report_path = os.path.join(self.path_dict[path_folder], self.model_name, self.folder_name, f"{apk}.json")
            with open(report_path, "r") as f:
                report_data = json.load(f)
            if not report_data["is_malicious"]:
                to_remove_apk_list.append(apk)
            if os.path.exists(os.path.join(save_folder, f"{apk}.json")):
                to_remove_apk_list.append(apk)
        apk_list = [apk for apk in apk_list if apk not in to_remove_apk_list]

        results = []
        avg_scores = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_apk = {executor.submit(self.evaluate_single_apk, apk, flag): apk for apk in tqdm(apk_list, desc="Evaluating apks for evidence correctness")}
            for future in as_completed(future_to_apk):
                apk = future_to_apk[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"apk": apk, "error": str(e)})
                if future.result() is None:
                    print(f"Error evaluating {apk}, result: {future.result()}")
                avg_scores.append(future.result()["avg_score"])
        
        avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        
        return results, avg

    def get_avg_score(self, flag):
        save_folder = os.path.join(self.path_dict["result"], "eas", f"{self.model_name}_{flag}", self.folder_name)
        file_list = glob.glob(os.path.join(save_folder, "*.json"))
        results = []
        for file in tqdm(file_list, desc="Getting avg score"):
            with open(file, "r") as f:
                results.append(json.load(f))
        avg = sum([result["avg_score"] for result in results]) / len(results) if results else 0.0
        return avg
    
    def get_avg_score_all(self, flag):
        malradar_save_folder = os.path.join(self.path_dict["result"], "eas", f"{self.model_name}_{flag}", "archived")
        new_save_folder = os.path.join(self.path_dict["result"], "eas", f"{self.model_name}_{flag}", "latest")
        malradar_file_list = glob.glob(os.path.join(malradar_save_folder, "*.json"))
        new_file_list = glob.glob(os.path.join(new_save_folder, "*.json"))
        
        all_file_list = malradar_file_list + new_file_list
        all_results = []
        for file in tqdm(all_file_list, desc="Getting avg score"):
            with open(file, "r") as f:
                all_results.append(json.load(f))
        avg = sum([result["avg_score"] for result in all_results]) / len(all_results) if all_results else 0.0
        return avg

if __name__ == "__main__":
    path_dict = utils.get_path_dict()
    model_list = ["llama","coder","qwen", "deepseek","gpt","gemini","claude"]
    folder_list = ["archived","latest"]
    flag_list = ["context","no_context","meta"]
    result_dict = {}
    for model in model_list:
        for flag in flag_list:
            for folder in folder_list:
                apk_list = utils.check_file_exist(folder)
                evidence_correctness = EvidenceCorrectness(path_dict, model, folder)
                results, avg = evidence_correctness.evaluate_multiple_apks(apk_list, flag)
            avg = evidence_correctness.get_avg_score_all(flag)
            print(f"Average score: {avg} for {model}_{flag}_all")
            result_dict[f"{model}_{flag}_all"] = avg
    with open(os.path.join(path_dict["result"], "eas", f"avg_score_all.json"), "w") as f:
        json.dump(result_dict, f, indent=4)