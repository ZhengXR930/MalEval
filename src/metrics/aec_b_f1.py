import json
import numpy as np
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
from utils import resolve_report_analysis_path


EQUIVALENCE_CLASSES = {
    "action": [
        {"CAPTURE", "MONITOR"},
        {"INSTALL", "INJECT"},
        {"PREVENT", "HIDE"}
    ],
    "asset": [
        {"SMS", "CALL_LOGS"},
        {"CREDENTIALS", "SENSITIVE_DATA"},
        {"CODE", "PAYLOAD", "APP"},
        {"ROOT_PRIVILEGES", "ADMIN_PRIVILEGES"}
    ],
    "target": [
        {"USER_INTERFACE", "BROWSER"},
        {"SYSTEM_SETTINGS", "DEVICE_ADMIN"}
    ]
}

METRIC_KEYS = [
    "aec_score", "ae_f1", "ae_precision", "ae_recall",
    "ae_f1_1to1", "ae_precision_1to1", "ae_recall_1to1",
    "b_f1", "b_precision", "b_recall"
]


def get_component_score(v1, v2, category):
    """
    Calculate similarity score for a component with semantic understanding.
    
    Scoring:
    - Exact match: 1.0 (perfect alignment)
    - Semantic match (Action): 0.9 (correct intent, different terminology)
    - Semantic match (Asset/Target): 1.0 (same category, equivalent meaning)
    - No match: 0.0
    """
    def normalize(val):
        if val is None or (isinstance(val, str) and val.lower() == "null"):
            return "NONE"
        return str(val).upper()
    
    v1_str = normalize(v1)
    v2_str = normalize(v2)
    
    if v1_str == v2_str:
        return 1.0
    
    groups = EQUIVALENCE_CLASSES.get(category, [])
    for group in groups:
        if v1_str in group and v2_str in group:
            return 0.9 if category == "action" else 1.0
    
    if category == "action":
        v1_groups = [group for group in groups if v1_str in group]
        v2_groups = [group for group in groups if v2_str in group]
        
        for v1_group in v1_groups:
            for v2_group in v2_groups:
                common_actions = v1_group.intersection(v2_group) - {v1_str, v2_str}
                if common_actions:
                    return 0.9
    
    return 0.0

def get_component_match_type(v1, v2, category):
    """
    Determine the match type for a component.
    Returns: ('exact', 1.0) or ('semantic', 1.0) or ('no_match', 0.0)
    """
    def normalize(val):
        if val is None or (isinstance(val, str) and val.lower() == "null"):
            return "NONE"
        return str(val).upper()
    
    v1_str = normalize(v1)
    v2_str = normalize(v2)
    
    if v1_str == v2_str:
        return 'exact', 1.0
    
    groups = EQUIVALENCE_CLASSES.get(category, [])
    for group in groups:
        if v1_str in group and v2_str in group:
            return 'semantic', 1.0
    
    if category == "action":
        v1_groups = [group for group in groups if v1_str in group]
        v2_groups = [group for group in groups if v2_str in group]
        
        for v1_group in v1_groups:
            for v2_group in v2_groups:
                common_actions = v1_group.intersection(v2_group) - {v1_str, v2_str}
                if common_actions:
                    return 'semantic', 1.0
    
    return 'no_match', 0.0

def calculate_triplet_match(t1, t2):
    """
    Calculate match score for two triplets using WEIGHTED SEMANTIC MATCHING.
    
    Returns:
        Float [0.0, 1.0]: Weighted similarity score
    """
    s_act = get_component_score(t1.get('action'), t2.get('action'), 'action')
    
    if s_act == 0:
        return 0.0
    
    s_ass = get_component_score(t1.get('asset'), t2.get('asset'), 'asset')
    s_tar = get_component_score(t1.get('target'), t2.get('target'), 'target')
    
    return (s_act * 0.6) + (s_ass * 0.3) + (s_tar * 0.1)

def get_triplet_match_details(t1, t2):
    act_type, act_score = get_component_match_type(t1.get('action'), t2.get('action'), 'action')
    ass_type, ass_score = get_component_match_type(t1.get('asset'), t2.get('asset'), 'asset')
    tar_type, tar_score = get_component_match_type(t1.get('target'), t2.get('target'), 'target')
    
    overall_score = 0 if act_score==0 else 0.7*act_score + 0.2*ass_score + 0.1*tar_score
    
    return {
        'overall_score': round(overall_score, 4),
        'action': {'value_llm': t1.get('action'), 'value_gt': t2.get('action'), 'match_type': act_type, 'score': round(act_score, 4)},
        'asset': {'value_llm': t1.get('asset'), 'value_gt': t2.get('asset'), 'match_type': ass_type, 'score': round(ass_score, 4)},
        'target': {'value_llm': t1.get('target'), 'value_gt': t2.get('target'), 'match_type': tar_type, 'score': round(tar_score, 4)}
    }

# --- Core Evaluation Functions ---

def evaluate_atom_evidence(llm_ev, gt_ev, tp_threshold=0.0):
    """
    Calculate Atomic Evidence Coverage (AEC) using weighted semantic matching.
    """
    llm_triplets = [e['evidence'] for e in llm_ev]
    gt_triplets = [e['evidence'] for e in gt_ev]
    
    if not gt_triplets:
        extra_llm_triplets = [{'index': idx, 'triplet': llm_triplets[idx]} for idx in range(len(llm_triplets))] if llm_triplets else []
        return {
            'aec_score': 0.0,
            'total_llm': len(llm_triplets),
            'total_gt': 0,
            'total_matched': 0,
            'ae_precision': 0.0,
            'ae_recall': 0.0,
            'ae_f1': 0.0,
            'ae_precision_1to1': 0.0,
            'ae_recall_1to1': 0.0,
            'ae_f1_1to1': 0.0,
            'total_matched_1to1': 0,
            'missing_gt_triplets': [],
            'extra_llm_triplets': extra_llm_triplets
        }
    
    if not llm_triplets:
        missing_gt_triplets = [{'index': idx, 'triplet': gt_triplets[idx]} for idx in range(len(gt_triplets))]
        return {
            'aec_score': 0.0,
            'total_llm': 0,
            'total_gt': len(gt_triplets),
            'total_matched': 0,
            'ae_precision': 0.0,
            'ae_recall': 0.0,
            'ae_f1': 0.0,
            'ae_precision_1to1': 0.0,
            'ae_recall_1to1': 0.0,
            'ae_f1_1to1': 0.0,
            'total_matched_1to1': 0,
            'missing_gt_triplets': missing_gt_triplets,
            'extra_llm_triplets': []
        }

    # Calculate similarity matrix
    scores = np.zeros((len(llm_triplets), len(gt_triplets)))
    for i, t_l in enumerate(llm_triplets):
        for j, t_g in enumerate(gt_triplets):
            scores[i, j] = calculate_triplet_match(t_l, t_g)
    
    # For each GT triplet, find its best matching LLM triplet (no restrictions on reuse)
    total_similarity_sum = 0.0
    matched_pairs = []
    matched_gt_indices = set() 
    used_llm_indices = set()    
    
    for j, t_g in enumerate(gt_triplets):
        best_score = 0.0
        best_llm_idx = -1
        
        for i, t_l in enumerate(llm_triplets):
            score = scores[i, j]
            if score > best_score:
                best_score = score
                best_llm_idx = i
        
        total_similarity_sum += best_score
        
        if best_llm_idx >= 0 and best_score >= tp_threshold:
            matched_gt_indices.add(j)
            used_llm_indices.add(best_llm_idx)
            details = get_triplet_match_details(llm_triplets[best_llm_idx], gt_triplets[j])
            matched_pairs.append({
                'llm_index': best_llm_idx,
                'gt_index': j,
                'score': round(best_score, 4),
                'details': details
            })
    
    aec_score = total_similarity_sum / len(gt_triplets)
    
    tp = len(matched_gt_indices)
    fp = len(llm_triplets) - len(used_llm_indices)
    fn = len(gt_triplets) - len(matched_gt_indices)
    
    missing_gt_indices = sorted(set(range(len(gt_triplets))) - matched_gt_indices)
    extra_llm_indices = sorted(set(range(len(llm_triplets))) - used_llm_indices)
    
    missing_gt_triplets = [{'index': idx, 'triplet': gt_triplets[idx]} for idx in missing_gt_indices]
    extra_llm_triplets = [{'index': idx, 'triplet': llm_triplets[idx]} for idx in extra_llm_indices]
    
    ae_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ae_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ae_f1 = 2 * ae_precision * ae_recall / (ae_precision + ae_recall) if (ae_precision + ae_recall) > 0 else 0.0

    candidates = []
    for i in range(len(llm_triplets)):
        for j in range(len(gt_triplets)):
            sc = float(scores[i, j])
            if sc >= tp_threshold:
                candidates.append((sc, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_llm_1to1 = set()
    used_gt_1to1 = set()
    matched_pairs_1to1 = []

    for sc, i, j in candidates:
        if i in used_llm_1to1 or j in used_gt_1to1:
            continue
        used_llm_1to1.add(i)
        used_gt_1to1.add(j)
        details = get_triplet_match_details(llm_triplets[i], gt_triplets[j])
        matched_pairs_1to1.append(
            {
                "llm_index": i,
                "gt_index": j,
                "score": round(sc, 4),
                "details": details,
            }
        )

    tp_1to1 = len(used_gt_1to1)
    fp_1to1 = len(llm_triplets) - len(used_llm_1to1)
    fn_1to1 = len(gt_triplets) - tp_1to1

    ae_precision_1to1 = tp_1to1 / (tp_1to1 + fp_1to1) if (tp_1to1 + fp_1to1) > 0 else 0.0
    ae_recall_1to1 = tp_1to1 / (tp_1to1 + fn_1to1) if (tp_1to1 + fn_1to1) > 0 else 0.0
    ae_f1_1to1 = (
        2 * ae_precision_1to1 * ae_recall_1to1 / (ae_precision_1to1 + ae_recall_1to1)
        if (ae_precision_1to1 + ae_recall_1to1) > 0
        else 0.0
    )

    return {
        'aec_score': round(aec_score, 4),
        'total_similarity_sum': round(total_similarity_sum, 4),
        'total_matched': len(matched_pairs),
        'total_llm': len(llm_triplets),
        'total_gt': len(gt_triplets),
        'matched_pairs': matched_pairs,
        'ae_precision': round(ae_precision, 4),
        'ae_recall': round(ae_recall, 4),
        'ae_f1': round(ae_f1, 4),
        'ae_precision_1to1': round(ae_precision_1to1, 4),
        'ae_recall_1to1': round(ae_recall_1to1, 4),
        'ae_f1_1to1': round(ae_f1_1to1, 4),
        'total_matched_1to1': len(matched_pairs_1to1),
        'matched_pairs_1to1': matched_pairs_1to1,
        'missing_gt_triplets': missing_gt_triplets,
        'extra_llm_triplets': extra_llm_triplets
    }

def evaluate_behavior_labels(llm_behaviors, gt_behaviors):
    """
    Compare behavior labels between LLM and GT reports.
    Also identifies missing behavior labels and extra behavior labels.
    """
    llm_labels = {b['label'] for b in llm_behaviors}
    gt_labels = {b['label'] for b in gt_behaviors}
    
    matched = llm_labels.intersection(gt_labels)
    unmatched_gt = gt_labels - llm_labels  
    unmatched_llm = llm_labels - gt_labels  
    
    tp = len(matched)
    fp = len(unmatched_llm)
    fn = len(unmatched_gt)
    
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b_f1 = 2 * b_precision * b_recall / (b_precision + b_recall) if (b_precision + b_recall) > 0 else 0.0
    
    return {
        'b_precision': round(b_precision, 4),
        'b_recall': round(b_recall, 4),
        'b_f1': round(b_f1, 4),
        'matched_labels': sorted(list(matched)),
        'missing_labels': sorted(list(unmatched_gt)),  
        'extra_labels': sorted(list(unmatched_llm)),  
        'total_gt_labels': len(gt_labels),
        'total_llm_labels': len(llm_labels),
        'total_matched': len(matched)
    }

def run_evaluation(gt_path, llm_path, model_name=None, tp_threshold=0.0):
    """
    Main evaluation function that loads JSON reports and computes all metrics.
    
    Returns:
        dict containing all evaluation results
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"LLM report file not found: {llm_path}")
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(llm_path, 'r') as f:
        llm_data = json.load(f)
    
    if model_name is None:
        parts = llm_path.split('/')
        for i, part in enumerate(parts):
            if part == 'behavior_reports' and i + 1 < len(parts):
                model_name = parts[i + 1]
                break
        if model_name is None:
            model_name = 'unknown'
    
    # Run evaluations
    atom_results = evaluate_atom_evidence(llm_data['atom_evidence'], gt_data['atom_evidence'], tp_threshold=tp_threshold)
    behavior_results = evaluate_behavior_labels(llm_data['behaviors'], gt_data['behaviors'])
    
    results = {
        'gt_path': gt_path,
        'llm_path': llm_path,
        'model_name': model_name,
        'atom_evidence': atom_results,
        'behavior_labels': behavior_results
    }
    
    return results

def process_samples(sha_list, info_data, gt_folder, llm_root, model_name, folder_name, tp_threshold):
    per_sample = {}
    skipped = 0
    for sha in sha_list:
        if sha not in info_data:
            skipped += 1
            continue
        gt_path, _ = resolve_report_analysis_path(gt_folder, info_data[sha]["url"])
        llm_path = os.path.join(llm_root, model_name, folder_name, f"{sha}.json")
        if gt_path is None or not os.path.exists(llm_path):
            skipped += 1
            continue
        try:
            results = run_evaluation(gt_path, llm_path, model_name, tp_threshold=tp_threshold)
            atom = results["atom_evidence"]
            beh = results["behavior_labels"]
            per_sample[sha] = {
                "aec_score": atom["aec_score"],
                "ae_f1": atom["ae_f1"],
                "ae_precision": atom["ae_precision"],
                "ae_recall": atom["ae_recall"],
                "ae_f1_1to1": atom["ae_f1_1to1"],
                "ae_precision_1to1": atom["ae_precision_1to1"],
                "ae_recall_1to1": atom["ae_recall_1to1"],
                "b_f1": beh["b_f1"],
                "b_precision": beh["b_precision"],
                "b_recall": beh["b_recall"],
            }
        except Exception:
            skipped += 1
    return per_sample, skipped

def compute_averages(per_sample_metrics):
    if not per_sample_metrics:
        return {f"avg_{k}": 0.0 for k in METRIC_KEYS}
    n = len(per_sample_metrics)
    return {f"avg_{k}": sum(m[k] for m in per_sample_metrics.values()) / n for k in METRIC_KEYS}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AEC/AE-F1/Behavior-F1 metrics.")
    parser.add_argument(
        "--model",
        default="deepseek",
        choices=["gemini", "gpt", "claude", "qwen", "deepseek", "llama", "coder"],
        help="Model name to evaluate.",
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
            "slicing",
        ],
        help="Behavior report setting to evaluate.",
    )
    parser.add_argument(
        "--tp-threshold",
        type=float,
        default=0.8,
        help="Triplet match threshold for TP counting.",
    )
    parser.add_argument(
        "--output-subdir",
        default="task2&3",
        help="Subdirectory under result/ to save metric outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_dict = utils.get_path_dict()
    gt_folder = path_dict["reports"]
    task_root = os.path.join(path_dict["result"], args.output_subdir)
    tp_threshold = args.tp_threshold

    with open(os.path.join(path_dict["info"], "archived_sample_info.json"), "r") as f:
        archived_info = json.load(f)
    with open(os.path.join(path_dict["info"], "latest_sample_info.json"), "r") as f:
        latest_info = json.load(f)

    behavior_root_map = {
        "context": path_dict["behavior"], "no_context": path_dict["no_context_behavior"],
        "meta": path_dict["meta_behavior"], "topk_removal_context": path_dict["topk_removal_behavior"],
        "topk_removal_no_context": path_dict["topk_removal_no_context_behavior"],
        "topk_removal_meta": path_dict["topk_removal_meta_behavior"],
        "raw_code": path_dict["raw_code_behavior"],
    }

    model_name = args.model
    flag = args.flag
    print(f"Evaluating AEC for model={model_name}, flag={flag}")
    llm_root = behavior_root_map[flag]

    archived_metrics, archived_skip = process_samples(
        list(archived_info.keys()), archived_info, gt_folder, llm_root, model_name, "malradar", tp_threshold)
    latest_metrics, latest_skip = process_samples(
        list(latest_info.keys()), latest_info, gt_folder, llm_root, model_name, "new", tp_threshold)

    per_sample_metrics = {**archived_metrics, **latest_metrics}
    skipped_count = archived_skip + latest_skip
    processed_count = len(per_sample_metrics)

    if processed_count == 0:
        print(f"Error: No files processed for model={model_name}, flag={flag}")
        sys.exit(1)

    overall = compute_averages(per_sample_metrics)

    out_dir = os.path.join(task_root, flag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{model_name}_{flag}.json"), "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "flag": flag,
                "tp_threshold": tp_threshold,
                "processed": processed_count,
                "skipped": skipped_count,
                "overall": overall,
                "per_sample": per_sample_metrics,
            },
            f,
            indent=4,
        )
    
    per_sample_dir = os.path.join(task_root, flag, model_name)
    os.makedirs(per_sample_dir, exist_ok=True)
    for sha, metrics in per_sample_metrics.items():
        with open(os.path.join(per_sample_dir, f"{sha}.json"), "w") as f:
            json.dump({"sha": sha, "model_name": model_name, **metrics}, f, indent=4)
            
