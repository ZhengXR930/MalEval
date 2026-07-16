import os
import json
from tqdm import tqdm
from collections import defaultdict
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from prompts import EAS_SYSTEM_PROMPT as SYSTEM_PROMPT


def get_report_root_key(flag: str) -> str:
    flag_to_key = {
        "context": "behavior",
        "no_context": "no_context_behavior",
        "meta": "meta_behavior",
        "topk_removal_context": "topk_removal_behavior",
        "topk_removal_no_context": "topk_removal_no_context_behavior",
        "topk_removal_meta": "topk_removal_meta_behavior",
        "raw_code": "raw_code_behavior",
    }
    if flag not in flag_to_key:
        raise ValueError(f"Invalid flag: {flag}")
    return flag_to_key[flag]


def parse_args():
    parser = argparse.ArgumentParser(description="Evidence Attribution Score (EAS) evaluation.")
    parser.add_argument(
        "--model",
        default="deepseek",
        choices=["gemini", "gpt", "claude", "qwen", "deepseek", "llama", "coder"],
        help="Model name for the behavior reports.",
    )
    parser.add_argument(
        "--flag",
        default="context",
        choices=[
            "context",
            "no_context",
            "meta",
            "topk_removal_context",
            "topk_removal_no_context",
            "topk_removal_meta",
            "raw_code",
        ],
        help="Report setting flag (context/no_context/meta/topk_removal_*/raw_code).",
    )
    parser.add_argument(
        "--split",
        default="new",
        choices=["malradar", "new", "benign"],
        help="Dataset split folder (malradar/new/benign).",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini",
        choices=["gemini", "gpt", "gpt-5", "claude", "qwen", "deepseek", "llama", "coder"],
        help="LLM judge model used by utils.get_llm_client().",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Thread pool size for evaluation.",
    )
    return parser.parse_args()


class EvidenceCorrectness:
    def __init__(self, path_dict, model_name, folder_name, judge_model="gemini", max_workers=24):
        self.path_dict = path_dict
        self.model_name = model_name
        self.folder_name = folder_name
        self.judge_model = judge_model
        self.max_workers = max_workers


    def _build_query(self, apk_name, flag):
        path_folder = get_report_root_key(flag)
        
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

        # Build evidence ID to atom_evidence mapping
        evidence_map = {}
        for evidence in report_data.get("atom_evidence", []):
            evidence_map[evidence["id"]] = evidence

        evidence_lines = []
        for behavior in report_data.get("behaviors", []):
            behavior_label = behavior.get("label", "Unknown")
            behavior_rationale = behavior.get("rationale", "")
            supporting_evidence_ids = behavior.get("supporting_evidence_ids", [])
            
            evidence_lines.append(f"Behavior: {behavior_label}\n")
            evidence_lines.append(f"  - Rationale: {behavior_rationale}\n")
            
            # Get supporting evidence and their functions
            evidence_lines.append(f"  - Supporting Evidence:\n")
            all_support_functions = []
            for ev_id in supporting_evidence_ids:
                if ev_id in evidence_map:
                    ev = evidence_map[ev_id]
                    ev_triplet = ev.get("evidence", {})
                    evidence_lines.append(f"    - [{ev_id}] Action: {ev_triplet.get('action')}, Asset: {ev_triplet.get('asset')}, Target: {ev_triplet.get('target')}\n")
                    evidence_lines.append(f"      Raw Text: {ev.get('raw_text', '')}\n")
                    evidence_lines.append(f"      Explanation: {ev.get('explanation', '')}\n")
                    # Collect support functions
                    support_funcs = ev.get("support_functions", [])
                    if isinstance(support_funcs, list):
                        all_support_functions.extend(support_funcs)
            
            # Add function summaries
            evidence_lines.append(f"  - Related Functions with their summaries:\n")
            for function_name in all_support_functions:
                if function_name in context_summary_map:
                    evidence_lines.append(f"  - Function: {function_name}\n")
                    evidence_lines.append(f"    - Summary: {context_summary_map[function_name]['summary']}\n")
                    evidence_lines.append(f"    - Sensitivity Score: {context_summary_map[function_name]['sensitivity_score']}\n")
                    evidence_lines.append(f"    - Reasoning: {context_summary_map[function_name]['reasoning']}\n")
        evidence_block = "\n".join(evidence_lines)

        query = f"""
        Please evaluate whether the AI model's generated behavior matches the function it identified to support.

        **EVIDENCE (Behavior, Atomic Evidence, Function Summaries, sorted by sensitivity):**
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
                        continue
                    else:
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
        result, _ = utils.get_llm_client(model=self.judge_model)(query, SYSTEM_PROMPT)

        try:
            behavior_scores = [b["support_score"] for b in result.get("behaviors", [])]
            overall_score = result.get("overall_score", None)

            final_result = {
                "apk": apk_name,
                "behaviors": result.get("behaviors", []),
                "overall_score": overall_score,
                "avg_score": sum(behavior_scores) / len(behavior_scores) if behavior_scores else 0.0
            }

            save_folder = os.path.join(self.path_dict["result"], f"eas_{self.judge_model}", f"{self.model_name}_{flag}", self.folder_name)
            os.makedirs(save_folder, exist_ok=True)
            with open(os.path.join(save_folder, f"{apk_name}.json"), "w") as f:
                json.dump(final_result, f, indent=4)
        except Exception as e:
            print(f"Error evaluating {apk_name}: {e}")
            return None

        return final_result
    
    def evaluate_multiple_apks(self, apk_list, flag):

        save_folder = os.path.join(self.path_dict["result"], f"eas_{self.judge_model}", f"{self.model_name}_{flag}", self.folder_name)
        path_folder = get_report_root_key(flag)

        to_remove_apk_list = []
        for apk in apk_list:
            report_path = os.path.join(self.path_dict[path_folder], self.model_name, self.folder_name, f"{apk}.json")
            try:
                with open(report_path, "r") as f:
                    report_data = json.load(f)
                is_malicious = report_data.get("verdict") == "malware"
                if not is_malicious:
                    to_remove_apk_list.append(apk)
            except FileNotFoundError:
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
                    continue
                avg_scores.append(future.result()["avg_score"])
        
        avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        
        return results, avg

    def get_avg_score(self, flag):
        save_folder = os.path.join(self.path_dict["result"], f"eas_{self.judge_model}", f"{self.model_name}_{flag}", self.folder_name)
        file_list = glob.glob(os.path.join(save_folder, "*.json"))
        results = []
        for file in tqdm(file_list, desc="Getting avg score"):
            with open(file, "r") as f:
                results.append(json.load(f))
        avg = sum([result["avg_score"] for result in results]) / len(results) if results else 0.0
        return avg
    
    def get_avg_score_all(self, flag):
        malradar_save_folder = os.path.join(self.path_dict["result"], f"eas_{self.judge_model}", f"{self.model_name}_{flag}", "malradar")
        new_save_folder = os.path.join(self.path_dict["result"], f"eas_{self.judge_model}", f"{self.model_name}_{flag}", "new")
        malradar_file_list = glob.glob(os.path.join(malradar_save_folder, "*.json"))
        new_file_list = glob.glob(os.path.join(new_save_folder, "*.json"))
        
        all_file_list = malradar_file_list + new_file_list
        all_results = []
        for file in tqdm(all_file_list, desc="Getting avg score"):
            with open(file, "r") as f:
                all_results.append(json.load(f))
        avg = sum([result["avg_score"] for result in all_results]) / len(all_results) if all_results else 0.0
        return avg

def compute_eas_std_across_evaluators(path_dict, model_name, flag, folder_list=["malradar", "new"]):
    """
    Compute the standard deviation of avg_score across three evaluators (GPT, DeepSeek, Gemini)
    for each report and save the combined results.
    """
    evaluator_folders = {
        "gpt": "eas",        # GPT results in eas/
        "deepseek": "eas_deepseek",
        "gemini": "eas_gemini"
    }
    
    combined_results = []
    
    for folder in folder_list:
        all_apk_names = set()
        for evaluator, eas_folder in evaluator_folders.items():
            folder_path = os.path.join(path_dict["result"], eas_folder, f"{model_name}_{flag}", folder)
            if os.path.exists(folder_path):
                files = glob.glob(os.path.join(folder_path, "*.json"))
                for f in files:
                    apk_name = os.path.basename(f).replace(".json", "")
                    all_apk_names.add(apk_name)
        
        # For each APK, collect avg_score from all evaluators
        for apk_name in tqdm(all_apk_names, desc=f"Processing {model_name}_{flag}/{folder}"):
            scores = {}
            
            for evaluator, eas_folder in evaluator_folders.items():
                file_path = os.path.join(
                    path_dict["result"], eas_folder, f"{model_name}_{flag}", folder, f"{apk_name}.json"
                )
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        scores[evaluator] = data.get("avg_score", None)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            
            # Only compute if we have at least 2 evaluators
            valid_scores = [s for s in scores.values() if s is not None]
            if len(valid_scores) >= 2:
                std = float(np.std(valid_scores, ddof=0))
                mean = float(np.mean(valid_scores))
                
                result = {
                    "apk": apk_name,
                    "folder": folder,
                    "scores": scores,
                    "mean": mean,
                    "std": std,
                    "num_evaluators": len(valid_scores)
                }
                combined_results.append(result)
    
    # Compute overall statistics
    all_stds = [r["std"] for r in combined_results]
    all_means = [r["mean"] for r in combined_results]
    
    overall_stats = {
        "model": model_name,
        "flag": flag,
        "total_reports": len(combined_results),
        "mean_of_means": float(np.mean(all_means)) if all_means else 0.0,
        "mean_std": float(np.mean(all_stds)) if all_stds else 0.0,
        "std_of_stds": float(np.std(all_stds)) if all_stds else 0.0,
        "max_std": float(max(all_stds)) if all_stds else 0.0,
        "min_std": float(min(all_stds)) if all_stds else 0.0,
    }
    
    # Save results
    save_folder = os.path.join(path_dict["result"], "eas_combined")
    os.makedirs(save_folder, exist_ok=True)
    
    with open(os.path.join(save_folder, f"{model_name}_{flag}_combined.json"), "w") as f:
        json.dump({
            "overall_stats": overall_stats,
            "reports": combined_results
        }, f, indent=4)
    
    return overall_stats, combined_results


def compute_all_eas_std(path_dict):
    """
    Compute EAS std for all models and flags and save summary.
    """
    model_list = ["llama", "coder", "qwen", "deepseek", "gpt", "gemini", "claude"]
    flag_list = ["context", "no_context", "meta", "topk_removal_context", "topk_removal_no_context", "topk_removal_meta"]
    
    summary = {}
    
    for model in model_list:
        for flag in flag_list:
            try:
                overall_stats, _ = compute_eas_std_across_evaluators(path_dict, model, flag)
                key = f"{model}_{flag}"
                summary[key] = {
                    "mean_score": overall_stats["mean_of_means"],
                    "mean_std": overall_stats["mean_std"],
                    "total_reports": overall_stats["total_reports"]
                }
                print(f"Processed {key}: mean={overall_stats['mean_of_means']:.4f}, mean_std={overall_stats['mean_std']:.4f}")
            except Exception as e:
                print(f"Error processing {model}_{flag}: {e}")
    
    # Save summary
    save_folder = os.path.join(path_dict["result"], "eas_combined")
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "summary_std.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary saved to {os.path.join(save_folder, 'summary_std.json')}")
    return summary


if __name__ == "__main__":
    args = parse_args()
    path_dict = utils.get_path_dict()

    # NOTE: EAS depends on `context_summary` (function summaries). For raw_code/slicing
    # reports, you can still evaluate, but it requires that the report's `support_functions`
    # refer to entries present in `context_summary`.
    apk_list = utils.check_file_exist(args.split)
    evidence_correctness = EvidenceCorrectness(
        path_dict, args.model, args.split, judge_model=args.judge_model, max_workers=args.max_workers
    )
    results, avg = evidence_correctness.evaluate_multiple_apks(apk_list, args.flag)
    print(f"Average score: {avg} for {args.model}_{args.flag}_{args.split}")
