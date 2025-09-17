import os
import json
from tqdm import tqdm
from collections import defaultdict
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
import numpy as np


SYSTEM_PROMPT = (
    """
You are an expert cybersecurity analyst acting as an impartial judge. 
Your task is to evaluate the quality of an AI-generated malware analysis report against a ground truth (GT) report. 

You must output a single JSON object with the following fields:

- `insight_score` (0.0–1.0): How well does the report identify the malware’s core malicious objective. If the report does not explicitly identify the GT’s final malicious objective, its score must be meaningfully limited, even if other details are strong.   
- `insight_justification`: A short explanation of why this score was given.  

- `comprehensiveness_score` (0.0–1.0): How complete and accurate the report is in covering GT core and secondary behaviors. Missing a GT core behavior must significantly lower the score.  
- `comprehensiveness_justification`: A short explanation of why this score was given.  

- `evidence_quality_score` (0.0–1.0): This score measures how strong, specific, and contextual the supporting evidence is. Reports that demonstrate cross-function or sequential attack chains (showing how multiple functions interact to achieve malicious objectives) should generally score higher. Reports that only list isolated APIs, single functions, or unconnected technical details should generally score lower.
- `evidence_quality_justification`: A short explanation of why this score was given, explicitly stating whether an attack chain was demonstrated (yes/no).  

- `final_verdict`: One to two sentences summarizing the overall quality of the report.  
 

**Final Output (strict JSON format):**
```json
{
  "insight_score": <float 0.0–1.0>,
  "insight_justification": "<string>",
  "comprehensiveness_score": <float 0.0–1.0>,
  "comprehensiveness_justification": "<string>",
  "evidence_quality_score": <float 0.0–1.0>,
  "evidence_quality_justification": "<string>",
  "final_verdict": "<one or two sentences summary>"
}
```

"""
)

class ReportEvaluator:
    def __init__(self, path_dict, model_name, folder_name):
        self.path_dict = path_dict
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        self.path_dict = path_dict
        if folder_name == "archived":
            self.gt_info_path = os.path.join(path_dict["info"], "archived_sample_info.json")
        elif folder_name == "latest":
            self.gt_info_path = os.path.join(path_dict["info"], "latest_sample_info.json")
        else:
            raise ValueError(f"Invalid folder name: {folder_name}")
        self.folder_name = folder_name
        with open(self.gt_info_path, "r") as f:
            self.gt_info = json.load(f)
        
        self.max_workers_apk = 16


    def _build_query(self, apk_name, flag):
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")
        report_name = os.path.join(self.path_dict[path_folder], f"{self.model_name}", self.folder_name, f"{apk_name}.json")
        print(f"generated report path: {report_name} for apk {apk_name}")
        with open(report_name, "r") as f:
            report = json.load(f)

        folder_name = utils.make_doc_id(self.gt_info[apk_name]["url"])
        gt_report_path = os.path.join(self.path_dict["report"], folder_name, "analysis_result.json")
        print(f"gt_report_path: {gt_report_path}")
        with open(gt_report_path, "r") as f:
            gt_report = json.load(f)


        if not report["is_malicious"]:
            return None

        gt_type = self.gt_info[apk_name]["type"]
        gt_family = self.gt_info[apk_name]["family"]
        gt_behaviors_list = gt_report["present_behaviors"]
        gt_summary = gt_report["summary"]

        pred_type = report["type"]
        report_behavior_list = []
        for behavior in report["present_behaviors"]:
            report_behavior_list.append({
                "behavior": behavior["behavior"],
                "confidence": behavior["confidence"],
                "evidence": behavior["evidence"],
            })
        summary = report["summary"]


        user_prompt = f"""
        Please evaluate the AI model's report against the ground truth report based on the expert evaluation principles provided in the system prompt.

        **[Ground Truth Report]**

        **Ground Truth Malware Type:**
        {gt_type}

        **Ground Truth Malware Family:**
        {gt_family}

        **Ground Truth Malicious Behaviors:**
        {gt_behaviors_list}

        **Summary:**
        {gt_summary}

        **[AI Model's Report to Evaluate]**

        **Predicted Malware Type:**
        {pred_type}

        **Identified Behaviors:**
        {report_behavior_list}

        **Summary:**
        {summary}
        """
        return user_prompt

    def get_gt_report(self, apk_name):
        folder_name = utils.make_doc_id(self.gt_info[apk_name]["url"])
        gt_report_path = os.path.join(self.path_dict["report"], folder_name, "analysis_result.json")
        print(f"gt_report_path: {gt_report_path}")
        with open(gt_report_path, "r") as f:
            gt_report = json.load(f)
        return gt_report


    def evaluate_single_apk(self, apk_name, flag):
        user_prompt = self._build_query(apk_name, flag)
        if user_prompt is None:
            return None
        
        result, final_user_message = utils.get_llm_client(model="gpt")(user_prompt, self.system_prompt)
        result['apk_name'] = apk_name
        result['gt_type'] = self.gt_info[apk_name]["type"]
        result['gt_family'] = self.gt_info[apk_name]["family"]

        eas = self.calculate_sas(apk_name, flag)
        result["evidence_score"] = eas

        return result

    def calculate_score(self, apk_name, flag):
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        elif flag == "all_no_context":
            path_folder = "all_no_context_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")
        report_name = os.path.join(self.path_dict[path_folder], f"{self.model_name}", self.folder_name, f"{apk_name}.json")
        print(f"generated report path: {report_name}")
        with open(report_name, "r") as f:
            report = json.load(f)

        save_folder = os.path.join(self.path_dict["result"], "quality", f"{self.model_name}_{flag}", self.folder_name)
        
        if report["is_malicious"] == False:
            final_result = {}
            final_result["apk_name"] = apk_name
            final_result["model_name"] = self.model_name
            final_result["folder_name"] = self.folder_name
            final_result["insight_score"] = 0.0
            final_result["comprehensiveness_score"] = 0.0
            final_result["insight_justification"] = ""
            final_result["comprehensiveness_justification"] = ""
            final_result["evidence_quality_score"] = 0.0
            final_result["evidence_quality_justification"] = ""
            final_result["final_verdict"] = ""
            final_result["report_quality_score"] = 0.0
            final_result["evidence_score"] = 0.0
            with open(os.path.join(save_folder, f"{apk_name}.json"), "w") as f:
                json.dump(final_result, f, indent=4)
            return final_result

        judge_output = self.evaluate_single_apk(apk_name, flag)
        if judge_output is None:
            final_result = {}
            final_result["apk_name"] = apk_name
            final_result["model_name"] = self.model_name
            final_result["folder_name"] = self.folder_name
            final_result["insight_score"] = 0.0
            final_result["comprehensiveness_score"] = 0.0
            final_result["insight_justification"] = ""
            final_result["comprehensiveness_justification"] = ""
            final_result["evidence_quality_score"] = 0.0
            final_result["evidence_quality_justification"] = ""
            final_result["final_verdict"] = ""
            final_result["report_quality_score"] = 0.0
            final_result["evidence_score"] = 0.0
            with open(os.path.join(save_folder, f"{apk_name}.json"), "w") as f:
                json.dump(final_result, f, indent=4)
            return final_result
        
        insight_score = float(judge_output.get("insight_score", 0.0))
        comprehensiveness_score = float(judge_output.get("comprehensiveness_score", 0.0))
        insight_justification = judge_output["insight_justification"]
        comprehensiveness_justification = judge_output["comprehensiveness_justification"]
        evidence_quality_score = judge_output["evidence_quality_score"]
        evidence_quality_justification = judge_output["evidence_quality_justification"]
        final_verdict = judge_output["final_verdict"]
        evidence_score = judge_output["evidence_score"]

        weight_insight = 0.3
        weight_comprehensiveness = 0.4
        weight_evidence_quality = 0.3

        # Calculate the final weighted score
        report_quality_score = (weight_insight * insight_score) + \
                            (weight_comprehensiveness * comprehensiveness_score) + \
                            (weight_evidence_quality * evidence_quality_score)

        final_result = {}
        final_result["apk_name"] = apk_name
        final_result["model_name"] = self.model_name
        final_result["folder_name"] = self.folder_name
        final_result["insight_score"] = insight_score
        final_result["comprehensiveness_score"] = comprehensiveness_score
        final_result["insight_justification"] = insight_justification
        final_result["comprehensiveness_justification"] = comprehensiveness_justification
        final_result["evidence_quality_score"] = evidence_quality_score
        final_result["evidence_quality_justification"] = evidence_quality_justification
        final_result["final_verdict"] = final_verdict
        final_result["report_quality_score"] = report_quality_score
        final_result["evidence_score"] = evidence_score

        
        with open(os.path.join(save_folder, f"{apk_name}.json"), "w") as f:
            json.dump(final_result, f, indent=4)
        print(f"path of final_result: {os.path.join(save_folder, f'{apk_name}.json')}")

        return final_result

    def evaluate_all_apks(self, apk_list, flag):
        save_folder = os.path.join(self.path_dict["result"], "quality", f"{self.model_name}_{flag}", self.folder_name)

        to_remove = []
        for apk_name in apk_list:
            eval_report_path = os.path.join(save_folder, f"{apk_name}.json")
            if os.path.exists(eval_report_path):
                to_remove.append(apk_name)
        apk_list = [apk_name for apk_name in apk_list if apk_name not in to_remove]
        print(f"length of apk_list: {len(apk_list)}")
        print(f"length of to_remove: {len(to_remove)}")

        os.makedirs(save_folder, exist_ok=True)

        final_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = {ex.submit(self.calculate_score, apk_name, flag): apk_name for apk_name in apk_list}
            
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {self.model_name} in {self.folder_name} for {flag}"):
                final_results.append(fut.result())
                
        return final_results

    
    def calculate_sas(self, apk_name, flag):
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        elif flag == "all_no_context":
            path_folder = "all_no_context_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")
        report_name = os.path.join(self.path_dict[path_folder], self.model_name, self.folder_name, f"{apk_name}.json")
        with open(report_name, "r") as f:
            report = json.load(f)
        eas = report["evidence_score"]
        return eas

    def calculate_sas_folder(self, apk_name, flag, folder_name):
        if flag == "context":
            path_folder = "behavior"
        elif flag == "no_context":
            path_folder = "no_context_behavior"
        elif flag == "meta":
            path_folder = "meta_behavior"
        elif flag == "all_no_context":
            path_folder = "all_no_context_behavior"
        else:
            raise ValueError(f"Invalid flag: {flag}")
        report_name = os.path.join(self.path_dict[path_folder], self.model_name, folder_name, f"{apk_name}.json")
        with open(report_name, "r") as f:
            report = json.load(f)
        sas = report["evidence_score"]
        return sas

    def evaluate_sas_apks(self, apk_list, flag):
        
        final_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = [ex.submit(self.calculate_sas, apk_name, flag) for apk_name in tqdm(apk_list, desc=f"Evaluating SAS for {self.model_name} in {self.folder_name} for flag {flag}")]
            for fut in as_completed(futures):
                final_results.append(fut.result())
        sas = np.mean(final_results)
        return sas

    def evaluate_sas_all(self, flag):

        malradar_apk_list = utils.check_file_exist("malradar")
        new_apk_list = utils.check_file_exist("new")

        final_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = [ex.submit(self.calculate_sas_folder, apk_name, flag, "malradar") for apk_name in tqdm(malradar_apk_list, desc=f"Evaluating SAS for {self.model_name} in {self.folder_name} for flag {flag}")]
            for fut in as_completed(futures):
                final_results.append(fut.result())
        with ThreadPoolExecutor(max_workers=self.max_workers_apk) as ex:
            futures = [ex.submit(self.calculate_sas_folder, apk_name, flag, "new") for apk_name in tqdm(new_apk_list, desc=f"Evaluating SAS for {self.model_name} in {self.folder_name} for flag {flag}")]
            for fut in as_completed(futures):
                final_results.append(fut.result())
        sas = np.mean(final_results)
        return sas
        

    def statistic(self, flag):
        save_folder = os.path.join(self.path_dict["result"], "quality", f"{self.model_name}_{flag}", self.folder_name)
        file_paths = glob.glob(os.path.join(save_folder, "*.json"))

        to_remove = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_name = file_name.replace(".json", "")
            if file_name == "statistic":
                to_remove.append(file_path)
        file_paths = [file_path for file_path in file_paths if file_path not in to_remove]
        print(f"length of file_paths: {len(file_paths)}")
        print(f"length of to_remove: {len(to_remove)}")

        all_eq_scores = []
        all_insight_scores = []
        all_comprehensiveness_scores = []
        all_evidence_quality_scores = []
        for file_path in tqdm(file_paths, desc=f"Statistic RQ for {self.model_name} in {self.folder_name} for flag {flag}"):
            with open(file_path, "r") as f:
                data = json.load(f)
                all_eq_scores.append(data["report_quality_score"])
                all_insight_scores.append(data["insight_score"])
                all_comprehensiveness_scores.append(data["comprehensiveness_score"])
                all_evidence_quality_scores.append(data["evidence_quality_score"])
        results = {
            "model_name": self.model_name,
            "folder_name": self.folder_name,
            "flag": flag,
            "mean_rq": np.mean(all_eq_scores),
            "std_rq": np.std(all_eq_scores),
            "mean_insight_score": np.mean(all_insight_scores),
            "std_insight_score": np.std(all_insight_scores),
            "mean_comprehensiveness_score": np.mean(all_comprehensiveness_scores),
            "std_comprehensiveness_score": np.std(all_comprehensiveness_scores),
            "mean_evidence_quality_score": np.mean(all_evidence_quality_scores),
            "std_evidence_quality_score": np.std(all_evidence_quality_scores),
        }
        print(f"{self.model_name} in {self.folder_name} for flag {flag}: Mean RQ: {np.mean(all_eq_scores)}, Median RQ: {np.median(all_eq_scores)}, Max RQ: {np.max(all_eq_scores)}, Min RQ: {np.min(all_eq_scores)}")
        with open(os.path.join(save_folder, f"statistic.json"), "w") as f:
            json.dump(results, f, indent=4)
        return results
    


    def statistic_all(self, flag):
        archived_save_folder = os.path.join(self.path_dict["result"], "quality", f"{self.model_name}_{flag}", "archived")
        latest_save_folder = os.path.join(self.path_dict["result"], "quality", f"{self.model_name}_{flag}", "latest")
        archived_file_list = glob.glob(os.path.join(archived_save_folder, "*.json"))
        latest_file_list = glob.glob(os.path.join(latest_save_folder, "*.json"))
        all_file_list = archived_file_list + latest_file_list

        to_remove = []
        for file_path in all_file_list:
            file_name = os.path.basename(file_path)
            file_name = file_name.replace(".json", "")
            if file_name == "statistic":
                to_remove.append(file_path)
        all_file_list = [file_path for file_path in all_file_list if file_path not in to_remove]
        print(f"length of all_file_list: {len(all_file_list)}")
        print(f"length of to_remove: {len(to_remove)}")

        all_eq_scores = []
        all_insight_scores = []
        all_comprehensiveness_scores = []
        all_evidence_quality_scores = []
        for file in tqdm(all_file_list, desc=f"Statistic RQ for {self.model_name} in {self.folder_name} for flag {flag}"):
            with open(file, "r") as f:
                data = json.load(f)
                all_eq_scores.append(data["report_quality_score"])
                all_insight_scores.append(data["insight_score"])
                all_comprehensiveness_scores.append(data["comprehensiveness_score"])
                all_evidence_quality_scores.append(data["evidence_quality_score"])

        results = {
            "model_name": self.model_name,
            "folder_name": self.folder_name,
            "flag": flag,
            "mean_rq": np.mean(all_eq_scores),
            "std_rq": np.std(all_eq_scores),
            "mean_insight_score": np.mean(all_insight_scores),
            "std_insight_score": np.std(all_insight_scores),
            "mean_comprehensiveness_score": np.mean(all_comprehensiveness_scores),
            "std_comprehensiveness_score": np.std(all_comprehensiveness_scores),
            "mean_evidence_quality_score": np.mean(all_evidence_quality_scores),
            "std_evidence_quality_score": np.std(all_evidence_quality_scores),
        }
        print(f"{self.model_name} in {self.folder_name} for flag {flag}: Mean RQ: {np.mean(all_eq_scores)}, Median RQ: {np.median(all_eq_scores)}, Max RQ: {np.max(all_eq_scores)}, Min RQ: {np.min(all_eq_scores)}")
        save_folder = os.path.join(self.path_dict["result"], "quality", f"{self.model_name}_{flag}")
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, f"statistic_all.json"), "w") as f:
            json.dump(results, f, indent=4)
        return results

    def statistic_cross_report(self):
        report_quality_apks = defaultdict(list)

        model_list = ["qwen", "deepseek", "gemini", "claude","llama","gpt","coder"]
        folder_name = "archived"
        flag = "context"

        for model_name in model_list:
            print(f"Evaluating {model_name} in {folder_name} for {flag}")
            quality_folder = os.path.join(self.path_dict["result"], "quality", f"{model_name}_{flag}", folder_name)
            file_paths = glob.glob(os.path.join(quality_folder, "*.json"))
            for file_path in tqdm(file_paths, desc=f"Evaluating {model_name} in {folder_name} for {flag}"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if "apk_name" not in data:
                            print(f"Warning: Missing 'apk_name' in {file_path}")
                            continue
                        if "report_quality_score" not in data:
                            print(f"Warning: Missing 'report_quality_score' in {file_path} for {data.get('apk_name', 'unknown')}")
                            continue
                        report_quality_apks[data["apk_name"]].append(data["report_quality_score"])
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        results = {}
        for apk_name, report_quality_scores in report_quality_apks.items():
            if not report_quality_scores:  
                print(f"Warning: No scores available for APK {apk_name}")
                continue  
            
            results[apk_name] = np.mean(report_quality_scores)
        
        if not results:
            print("Error: No valid results to save")
            return {}
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        output_path = os.path.join(self.path_dict["result"], "quality", f"cross_report_quality_{flag}_{folder_name}.json")
        
        with open(output_path, "w") as f:
            json.dump(sorted_results, f, indent=4)
        
        print(f"Successfully saved results to {output_path}")
        return sorted_results
        

if __name__ == "__main__":
    model_name = "llama"
    
    flag = "context"
    path_dict = utils.get_path_dict()
    folder_name = "archived"
    archived_report_evaluator = ReportEvaluator(path_dict, model_name, folder_name)
    apk_list = utils.check_file_exist(folder_name)
    archived_report_evaluator.evaluate_all_apks(apk_list, flag)

    folder_name = "latest"
    latest_report_evaluator = ReportEvaluator(path_dict, model_name, folder_name)
    apk_list = utils.check_file_exist(folder_name)
    latest_report_evaluator.evaluate_all_apks(apk_list, flag)
    print(f"--------------------------------------------------")

    statistic_all_result = latest_report_evaluator.statistic_all(flag)
    sas_all_result = latest_report_evaluator.evaluate_sas_all(flag)
    print(f"Model name: {model_name}, Flag: {flag}")
    print(f"SAS result: {sas_all_result}")
    print(f"Statistic result: {statistic_all_result}")
   

