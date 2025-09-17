import os
import json
from tqdm import tqdm
from collections import defaultdict
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import utils
import re




BEHAVIORS = [
    "Privacy Stealing", "SMS/CALL", "Remote Control", "Bank Stealing",
    "Ransom", "Abusing Accessibility", "Privilege Escalation",
    "Stealthy Escalation", "Ads", "Miner", "Tricky Behavior", "Premium Service",
]

MALICIOUS_TYPES = ["Spyware", "Ransomware", "Adware", "Banker", "Trojan", "Downloader", "Miner", "Scareware", "Rootkit", "Botnet"]


SYSTEM_PROMPT = f"""
You are an expert Android malware analyst. Your task is to conduct a detailed security audit of a given Android application, which is **already suspected to be malicious**. Your analysis will be based on a collection of function summaries. Your final output MUST be a conclusive report in JSON format.

**INPUT:**
You will receive a list of function summaries, sorted from most to least sensitive, which constitute the evidence for your analysis.

---
**TASK 1: Audit Framework (Confirming or Refuting Malicious Intent)**
You MUST use this framework to finalize your audit conclusion. The key differentiator is **intent**.
+ Crucial Rule for Interpretation: You must interpret "harm" broadly. Significant privacy invasion, severe user experience disruption (e.g., out-of-app ads), or excessive resource consumption are all valid forms of harm. 
The use of a legitimate SDK to perform these actions does not excuse the harm. An application whose most impactful or resource-intensive behaviors are dedicated to these harmful activities MUST be classified as malicious, with any other simple functionality considered a potential facade.

1.  **Assess to CONFIRM Malicious Intent:** Your primary goal is to find evidence that confirms the initial suspicion. Does the evidence point to deception, harm, non-consensual actions, or a clear violation of user trust for the attacker's gain? Do the actions strongly match any of the malicious `BEHAVIORS` defined below?
    -   If YES, your conclusion is **MALICIOUS**.

2.  **Assess to REFUTE Malicious Intent:** If the evidence for malice is not conclusive, you must consider if the suspicious activities have a legitimate purpose.
    -   **Crucial Test:** Would an average user understand *why* this permission or action is needed for the app to work as advertised? (e.g., a map app using location, a backup app reading SMS).
    -   If the actions align with a transparent and legitimate purpose that explains the initial suspicion, your conclusion is **BENIGN**.

---
**TASK 2: JSON Report Generation**
Based on your audit conclusion from TASK 1, generate the final JSON output.
**RULE:**
1. Read all the provided function summaries to build a complete picture of the application's capabilities.
2. Based on the combined evidence, decide whether to confirm or refute the initial suspicion of malice.

3.  **If you conclude the application is BENIGN (Initial Suspicion REFUTED)**:
    -   Your entire response MUST be a single JSON object with the following structure:
    {{
      "is_malicious": false,
      "present_behaviors": [],
      "summary": "..."
    }}
    -   In the `summary` field, you MUST explain *why* the initial suspicion was incorrect and what the legitimate purpose of the sensitive actions is.

4.  **If you conclude the application is MALICIOUS (Initial Suspicion CONFIRMED)**:
    -   You MUST generate a JSON object that strictly follows this structure.
    -   The `is_malicious` key MUST be `true` (boolean, not string).

    {{
      "is_malicious": true,
      "present_behaviors": [
        {{
          "behavior": "...",
          "confidence": "...",
          "evidence": "...",
          "related_functions": ['...', '...']
        }}
      ],
      "summary": "...",
      "type": "..."
    }}

**OUTPUT FORMAT DETAILS (MALICIOUS CASE):**
-   `is_malicious`: Must be the boolean value `true`.
-   `present_behaviors`: A list of objects. Each object identifies one distinct malicious behavior.
    -   `behavior`: MUST be one of the following predefined strings: {BEHAVIORS}. If no predefined behavior applies, simply do not include it. Do NOT substitute with the closest category or invent new behaviors.
    -   `confidence`: MUST be one of "low", "medium", or "high".
    -   `evidence`: A concise string (1-2 sentences) explaining *why* you chose that behavior, citing specific actions from the function summaries.
    -   `related_functions`: A list of the most relevant functions (at least one, usually 1–3) that provide the strongest evidence for this behavior.. 
-   `summary`: MUST synthesize all evidence into a clear, high-level explanation of the malware’s operation and goals. You MUST only use information supported by the function summaries; do NOT invent capabilities.
-   `type`: Choose the single most appropriate category from {MALICIOUS_TYPES}, prioritizing the core behavior. If multiple types apply, select the one that best represents the primary intent.


**CRITICAL INSTRUCTION:** Your final output must ONLY be the JSON object. Do not include any introductory text, explanations, or markdown formatting like ```json.
"""

class BehaviorExtractor:
    def __init__(self, path_dict, model_name, folder_name, gt_info_path, max_workers=48):
        self.path_dict = path_dict
        self.model_name = model_name
        self.folder_name = folder_name
        self.max_workers = max_workers
        
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

    def _build_query(self, apk_name, flag = "context", top_k=300):
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

        return query

    def get_behavior_single_apk(self, apk_name, flag = "context"):

        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        else:
            print(f"flag: {flag} is not valid")
            exit()
        
        query = self._build_query(apk_name, flag)
        result, final_user_message = utils.get_llm_client(model=model_name)(query, SYSTEM_PROMPT)
        if result is None:
            return None
        if self.folder_name != "benign":
            gt_type, gt_family = self.gt_info[apk_name]["type"], self.gt_info[apk_name]["family"]
        else:
            gt_type, gt_family = "benign", "benign"

        result["gt_type"] = gt_type
        result["gt_family"] = gt_family
        result["evidence_score"] = self._calculate_evidence_score(result, final_user_message)
        os.makedirs(os.path.join(self.path_dict[path_folder], f"{self.model_name}", self.folder_name), exist_ok=True)
        with open(os.path.join(self.path_dict[path_folder], f"{self.model_name}", self.folder_name, f"{apk_name}.json"), "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    def _calculate_evidence_score(self, result, final_user_message):
        input_functions = set(re.findall(r"Function:\s+([^\n]+)", final_user_message))
        report_functions = set()
        for b in result["present_behaviors"]:
            report_functions.update(b["related_functions"])

        if len(report_functions) == 0:
            return 0

        matches = report_functions.intersection(input_functions)
        match_ratio = len(matches) / len(report_functions)
        print(f"matches: {matches}")
        print(f"match_ratio: {match_ratio}")
        return match_ratio


    def get_behavior_all(self, apk_list, flag = "context"):
        assert flag in ["context", "no_context"]
        log_path = os.path.join(self.path_dict["info"], 'statistic', f"{self.model_name}_{self.folder_name}_{flag}.txt")
        with open(log_path, "w") as f:
            f.write(f"model: {self.model_name}\n")
            f.write(f"folder: {self.folder_name}\n")
        rejected_count = 0
        count= 0

        to_remove = []
        for apk_name in apk_list:
            behavior_path = os.path.join(self.path_dict["behavior"], self.model_name, self.folder_name, f"{apk_name}.json")
            if not os.path.exists(behavior_path):
                to_remove.append(apk_name)
        apk_list = [apk_name for apk_name in apk_list if apk_name not in to_remove]


        failed_apks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_apk = {
                executor.submit(self.get_behavior_single_apk, apk_name, flag): apk_name
                for apk_name in apk_list
            }

            for future in as_completed(future_to_apk):
                apk_name = future_to_apk[future]
                result = future.result()
                count += 1
                if result is None:
                    rejected_count += 1
                    failed_apks.append(apk_name)
                else:
                    print(f"processed {count} / {len(apk_list)} successfully")
            time.sleep(2)
        print(f"rejected count: {rejected_count} in {len(apk_list)}")

        with open(log_path, "a") as f:
            f.write(f"rejected count: {rejected_count} in {len(apk_list)}\n")
            f.write(f"failed apks: {failed_apks}\n")

        



if __name__ == "__main__":
    model_name = "llama"
    folder_name = "benign"
    path_dict = utils.get_path_dict()
    gt_info_path = os.path.join(path_dict["info"], "benign_sample_info.json")
    behavior_extractor = BehaviorExtractor(path_dict, model_name, folder_name, gt_info_path)
    apk_list = utils.check_file_exist(folder_name)
    behavior_extractor.get_behavior_all(apk_list, flag="context")


        