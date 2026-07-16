"""
Report Quality Evaluation Script using LLM-as-a-Judge (GPT-5)

This script evaluates the quality of LLM-generated malware analysis reports by using
GPT-5 as a judge to compare complete generated reports against Ground Truth reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import REPORT_QUALITY_SYSTEM_PROMPT
from utils import get_path_dict, get_sample_info_path, resolve_report_analysis_path


path_dict = get_path_dict()
report_quality_root = Path(path_dict["result"]).parent / "report_quality_"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def call_openai_judge(prompt: str, *, judge_model: str = "gpt-5", api_key: Optional[str] = None) -> str:
    """
    Call the judge model to get report-quality judgment.
    
    Args:
        prompt: The prompt for GPT-5
        judge_model: Judge model name
        api_key: OpenAI API key (if None, uses environment variable OPENAI_API_KEY)
    
    Returns:
        Response text from the judge model
    """
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": REPORT_QUALITY_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=12000  # GPT-5 uses max_completion_tokens instead of max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to call GPT-5 API: {e}")


def build_llm_report_for_judging(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the substantive generated-report fields and drop run metadata."""
    return {
        "verdict": llm_data.get("verdict"),
        "atom_evidence": llm_data.get("atom_evidence", []),
        "behaviors": llm_data.get("behaviors", []),
        "summary": llm_data.get("summary"),
    }


def build_gt_report_for_judging(gt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the full GT report structure relevant for quality comparison."""
    return {
        "atom_evidence": gt_data.get("atom_evidence", []),
        "behaviors": gt_data.get("behaviors", []),
        "malware_summary": gt_data.get("malware_summary"),
    }


def create_evaluation_prompt(llm_report: Dict[str, Any], gt_report: Dict[str, Any]) -> str:
    """Create a comprehensive prompt for the judge to evaluate report quality."""
    llm_report_str = json.dumps(llm_report, indent=2, ensure_ascii=False)
    gt_report_str = json.dumps(gt_report, indent=2, ensure_ascii=False)

    prompt = f"""
## Ground Truth Report (complete)
{gt_report_str}

---

## AI-Generated Report To Evaluate (complete)
{llm_report_str}

---

Evaluate the AI-generated report against the GT report as a complete report.
Use the evidence, behaviors, verdict/type, and final summary together.
Provide your assessment in the specified JSON format.
"""
    return prompt


def parse_llm_response(response: str) -> Dict[str, Any]:
    response = response.strip()

    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON object found in GPT-5 response")
    
    json_str = response[start_idx:end_idx + 1]
    
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        json_str = json_str.replace(',\n}', '\n}').replace(',\n]', '\n]')
        try:
            result = json.loads(json_str)
            return result
        except:
            raise ValueError(f"Failed to parse GPT-5 JSON response: {e}. Response: {response[:500]}")


def resolve_report_dir_key(report_type: str, flag: str, mode: str) -> str:
    if report_type == "behavior":
        if mode == "topk_removal":
            return "topk_removal_behavior" if flag == "context" else "topk_removal_no_context_behavior"
        return "behavior" if flag == "context" else "no_context_behavior"
    if report_type == "meta_behavior":
        if flag != "context":
            raise ValueError("meta_behavior does not support --flag no_context.")
        return "topk_removal_meta_behavior" if mode == "topk_removal" else "meta_behavior"
    if report_type == "raw_code_behavior":
        return "raw_code_behavior"
    raise ValueError(f"Unsupported report_type: {report_type}")
def resolve_output_subdir(report_type: str, flag: str, mode: str) -> str:
    if report_type == "behavior":
        if mode == "topk_removal":
            return "topk_removal" if flag == "context" else "no_context_removal"
        return "context" if flag == "context" else "no_context"
    if report_type == "meta_behavior":
        return "meta_topk_removal" if mode == "topk_removal" else "meta_context"
    if report_type == "raw_code_behavior":
        return "raw_code"
    raise ValueError(f"Unsupported report_type: {report_type}")


def load_info_data(folder_name: str) -> Dict[str, Dict[str, Any]]:
    info_path = get_sample_info_path(folder_name)
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_report_quality(
    *,
    sha256: str,
    report_model: str,
    folder_name: str,
    report_root: Path,
    gt_root: Path,
    info_data: Dict[str, Dict[str, Any]],
    judge_model: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one generated report against the matching GT analysis.json."""
    if sha256 not in info_data:
        raise ValueError(f"SHA256 {sha256} not found in sample info")

    url = info_data[sha256].get("url")
    if not url or not isinstance(url, str):
        raise ValueError(f"URL not found for {sha256} in sample info")

    gt_path, doc_id = resolve_report_analysis_path(gt_root, url)

    llm_path = report_root / report_model / folder_name / f"{sha256}.json"

    if not llm_path.exists():
        raise FileNotFoundError(f"LLM report not found: {llm_path}")

    if gt_path is None:
        raise FileNotFoundError(f"GT report not found for URL {url}. Expected doc id: {doc_id}")

    with llm_path.open("r", encoding="utf-8") as f:
        llm_data = json.load(f)

    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    llm_report = build_llm_report_for_judging(llm_data)
    gt_report = build_gt_report_for_judging(gt_data)
    prompt = create_evaluation_prompt(llm_report=llm_report, gt_report=gt_report)

    response = call_openai_judge(prompt, judge_model=judge_model, api_key=api_key)
    evaluation_result = parse_llm_response(response)

    overall_score = clamp01(
        safe_float(evaluation_result.get("overall_quality_score"), 0.0)
    )
    confidence_score = clamp01(
        safe_float(
            evaluation_result.get(
                "confidence_score",
                evaluation_result.get("confidence", 0.5),
            ),
            0.5,
        )
    )
    explanation = str(
        evaluation_result.get(
            "explanation",
            evaluation_result.get("reasoning", "No explanation provided."),
        )
    )
    uncertainty_explanation = str(
        evaluation_result.get(
            "uncertainty_explanation",
            evaluation_result.get(
                "confidence_explanation",
                "No uncertainty explanation provided.",
            ),
        )
    )
    behavior_details = evaluation_result.get("behavior_details")
    if not isinstance(behavior_details, dict):
        behavior_details = {
            "well_supported": evaluation_result.get("aligned_strengths", []),
            "missing_from_llm": evaluation_result.get("missing_or_misaligned_core_points", []),
            "speculative_or_unsupported": evaluation_result.get("unsupported_or_weak_claims", []),
            "weak_or_keyword_based": [],
        }

    results = {
        "apk_name": sha256,
        "report_model": report_model,
        "judge_model": judge_model,
        "report_quality_score": round(overall_score, 4),
        "confidence_score": round(confidence_score, 4),
        "explanation": explanation,
        "uncertainty_explanation": uncertainty_explanation,
        "behavior_details": behavior_details,
        "doc_id": doc_id,
        "url": url,
        "gt_type": info_data[sha256].get("type"),
        "gt_family": info_data[sha256].get("family"),
        "llm_path": str(llm_path),
        "gt_path": str(gt_path),
        "raw_response": response,
    }

    return results


def process_single_sha(
    *,
    sha: str,
    report_model: str,
    folder_name: str,
    report_root: Path,
    gt_root: Path,
    info_data: Dict[str, Dict[str, Any]],
    output_root: Path,
    judge_model: str,
    force: bool,
    api_key: Optional[str] = None,
) -> Optional[Tuple[str, str, str, Dict[str, float]]]:
    """Evaluate one SHA and persist the judge output."""
    output_file = output_root / f"{sha}.json"
    if output_file.exists() and not force:
        return None

    family_name = str(info_data[sha].get("family", ""))
    type_name = str(info_data[sha].get("type", ""))

    try:
        results = evaluate_report_quality(
            sha256=sha,
            report_model=report_model,
            folder_name=folder_name,
            report_root=report_root,
            gt_root=gt_root,
            info_data=info_data,
            judge_model=judge_model,
            api_key=api_key,
        )

        output_root.mkdir(parents=True, exist_ok=True)
        print(f"Saving RQ result for APK {sha} to {output_file}")
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return (sha, type_name, family_name, {
            "report_quality_score": results["report_quality_score"],
        })
    except Exception as e:
        print(f"Error processing {sha}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated malware reports against GT analysis.json with an LLM judge.")
    parser.add_argument("--model", default="deepseek", choices=["claude", "gpt", "qwen", "deepseek", "llama", "coder", "gemini"])
    parser.add_argument("--judge-model", default="gpt-5", help="Judge model name for OpenAI chat completions")
    parser.add_argument("--folder", default="new", choices=["malradar", "new"])
    parser.add_argument("--report-type", default="behavior", choices=["behavior", "meta_behavior", "raw_code_behavior"])
    parser.add_argument("--flag", default="context", choices=["context", "no_context"])
    parser.add_argument("--mode", default="normal", choices=["normal", "topk_removal"])
    parser.add_argument("--apk", help="Evaluate a single APK sha256")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-run judge evaluation even if output file already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_dir_key = resolve_report_dir_key(args.report_type, args.flag, args.mode)
    output_subdir = resolve_output_subdir(args.report_type, args.flag, args.mode)
    report_root = Path(path_dict[report_dir_key])
    gt_root = Path(path_dict["reports"])
    info_data = load_info_data(args.folder)

    output_root = (
        report_quality_root
        / output_subdir
        / args.model
        / args.folder
    )

    if args.apk:
        process_single_sha(
            sha=args.apk,
            report_model=args.model,
            folder_name=args.folder,
            report_root=report_root,
            gt_root=gt_root,
            info_data=info_data,
            output_root=output_root,
            judge_model=args.judge_model,
            force=args.force,
            api_key=api_key,
        )
        return

    sha_list = sorted(info_data.keys())
    print(f"Evaluating report quality for {len(sha_list)} APKs")

    report_score_results = defaultdict(lambda: defaultdict(dict))
    to_process = []
    for sha in sha_list:
        output_file = output_root / f"{sha}.json"
        if output_file.exists() and not args.force:
            continue
        to_process.append(sha)

    print(f"Processing {len(to_process)} APKs")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_sha = {
            executor.submit(
                process_single_sha,
                sha=sha,
                report_model=args.model,
                folder_name=args.folder,
                report_root=report_root,
                gt_root=gt_root,
                info_data=info_data,
                output_root=output_root,
                judge_model=args.judge_model,
                force=args.force,
                api_key=api_key,
            ): sha
            for sha in to_process
        }
        
        for future in tqdm(as_completed(future_to_sha), total=len(future_to_sha), desc="report_quality", unit="apk"):
            result = future.result()
            if result is not None:
                sha, type_name, family_name, scores = result
                report_score_results[type_name][family_name][sha] = scores

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "report_score_results.json").open("w", encoding="utf-8") as f:
        json.dump(report_score_results, f, indent=4)


if __name__ == "__main__":
    main()
