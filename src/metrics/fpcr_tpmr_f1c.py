import json
import os
import numpy as np
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

class_mapping = {
    'trojan': 0,
    'banker': 1,
    'rootkit': 2,
    'spyware': 3,
    'adware': 4,
    'ransomware': 5
}


def get_behavior_folder(path_dict, flag):
    flag_to_key = {
        "context": "behavior",
        "no_context": "no_context_behavior",
        "meta": "meta_behavior",
        "raw_code": "raw_code_behavior",
    }
    try:
        return path_dict[flag_to_key[flag]]
    except KeyError as exc:
        raise ValueError(f"Invalid flag: {flag}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FPCR/TPMR/F1_c metrics.")
    parser.add_argument(
        "--model",
        default="deepseek",
        choices=["gemini", "gpt", "claude", "qwen", "deepseek", "llama", "coder"],
        help="Model name to evaluate.",
    )
    parser.add_argument(
        "--flag",
        default="context",
        choices=["context", "no_context", "meta", "raw_code"],
        help="Behavior report setting to evaluate.",
    )
    parser.add_argument(
        "--output-subdir",
        default="task4",
        help="Subdirectory under result/ to save metric outputs.",
    )
    return parser.parse_args()

class ClassificationEvaluator:
    def __init__(self, path_dict, model_name):
        self.model_name = model_name

        self.path_dict = path_dict
        self.save_folder = os.path.join(self.path_dict["info"], "fpcr_tpmr_f1c")
        # Record samples whose reports are unusable (e.g., missing verdict)
        self.failed_sha256 = []

    def load_behavior_report(self, apk_name, flag, folder_name):
        behavior_folder = get_behavior_folder(self.path_dict, flag)

        behavior_report_path = os.path.join(behavior_folder, self.model_name, folder_name, f"{apk_name}.json")
        with open(behavior_report_path, "r") as f:
            behavior_report = json.load(f)

        verdict = behavior_report.get("verdict", None)
        if verdict is None:
            # Record failure and signal caller to skip this sample
            self.failed_sha256.append(apk_name)
            return None

        y_pred = 1 if verdict == "malware" else 0

        return y_pred

    

    def apk_filter(self, apk_list, flag, folder_name):
        behavior_folder = get_behavior_folder(self.path_dict, flag)

        to_remove = []
        for apk_name in apk_list:
            behavior_report_path = os.path.join(behavior_folder, self.model_name, folder_name, f"{apk_name}.json")
            with open(behavior_report_path, "r") as f:
                behavior_report = json.load(f)
            verdict = behavior_report.get("verdict")
            if verdict == "benign":
                to_remove.append(apk_name)
        print(f"Number of APKs to remove: {len(to_remove)}")
        apk_list = [apk_name for apk_name in apk_list if apk_name not in to_remove]
        return apk_list

    def evaluate_tpmr(self, apk_list, flag, folder_name):

        y_preds = []
        for apk_name in tqdm(apk_list, desc="Evaluating TPMR"):
            y_pred = self.load_behavior_report(apk_name, flag, folder_name)
            if y_pred is None:
                continue
            y_preds.append(y_pred)

        
        y_true = [1] * len(y_preds)
        tpmr = np.mean(np.array(y_preds) == np.array(y_true))
        return tpmr
    
    def evaluate_tpmr_all(self, flag):
        """
        TPMR over all malware samples (archived + new).
        Malware behavior reports live in `malradar/` and `new/` under the behavior folder.
        Malware SHA256 lists are loaded from info JSONs, not call_chain.
        """
        # Load archive and new malware SHA256s from info files
        archived_info_path = os.path.join(self.path_dict["info"], "archived_sample_info.json")
        latest_info_path = os.path.join(self.path_dict["info"], "latest_sample_info.json")

        with open(archived_info_path, "r") as f:
            archived_info = json.load(f)
        with open(latest_info_path, "r") as f:
            latest_info = json.load(f)

        malradar_apk_list = list(archived_info.keys())
        new_apk_list = list(latest_info.keys())

        y_preds = []

        for apk_name in tqdm(malradar_apk_list, desc="Evaluating TPMR (malradar)"):
            try:
                y_pred = self.load_behavior_report(apk_name, flag, "malradar")
            except FileNotFoundError:
                continue
            if y_pred is None:
                continue
            y_preds.append(y_pred)

        for apk_name in tqdm(new_apk_list, desc="Evaluating TPMR (new)"):
            try:
                y_pred = self.load_behavior_report(apk_name, flag, "new")
            except FileNotFoundError:
                continue
            if y_pred is None:
                continue
            y_preds.append(y_pred)

        y_true = [1] * len(y_preds)
        tpmr = np.mean(np.array(y_preds) == np.array(y_true))
        return tpmr

    def evaluate_fpcr(self, apk_list, flag, folder_name):
        
        assert folder_name in ["benign"]

        y_preds = []
        for apk_name in tqdm(apk_list, desc="Evaluating FPCR"):
            y_pred = self.load_behavior_report(apk_name, flag, folder_name)
            if y_pred is None:
                continue
            y_preds.append(y_pred)
        
        y_true = [0] * len(y_preds)
        fpcr = np.mean(np.array(y_preds) == np.array(y_true))

        return fpcr

    def evaluate_fpcr_all(self, flag):
        """
        FPCR over all benign samples.
        Benign behavior reports live in `benign/` under the behavior folder.
        """
        behavior_folder = get_behavior_folder(self.path_dict, flag)

        benign_reports_folder = os.path.join(behavior_folder, self.model_name, "benign")
        if not os.path.exists(benign_reports_folder):
            return 0.0

        all_apk_list = [
            os.path.splitext(fn)[0]
            for fn in os.listdir(benign_reports_folder)
            if fn.endswith(".json")
        ]

        y_preds = []
        for apk_name in tqdm(all_apk_list, desc="Evaluating FPCR (benign)"):
            y_pred = self.load_behavior_report(apk_name, flag, "benign")
            if y_pred is None:
                continue
            y_preds.append(y_pred)

        y_true = [0] * len(y_preds)
        fpcr = np.mean(np.array(y_preds) == np.array(y_true))
        return fpcr

    def evaluate_accuracy(self, flag: str):
        apk_list_archived = utils.check_file_exist("archived")
        apk_list_latest = utils.check_file_exist("latest")
        apk_list_benign = utils.check_file_exist("benign")
        
        y_pred = []
        for apk_name in tqdm(apk_list_archived, desc="Evaluating accuracy"):
            pred = self.load_behavior_report(apk_name, flag, "malradar")
            if pred is not None:
                y_pred.append(pred)

        for apk_name in tqdm(apk_list_latest, desc="Evaluating accuracy"):
            pred = self.load_behavior_report(apk_name, flag, "new")
            if pred is not None:
                y_pred.append(pred)
        
        for apk_name in tqdm(apk_list_benign, desc="Evaluating accuracy"):
            pred = self.load_behavior_report(apk_name, flag, "benign")
            if pred is not None:
                y_pred.append(pred)
        
        y_true = [1] * len(apk_list_archived) + [1] * len(apk_list_latest) + [0] * len(apk_list_benign)
        accuracy = np.mean(np.array(y_pred) == np.array(y_true))
        f1 = f1_score(np.array(y_true), np.array(y_pred))
        precision = precision_score(np.array(y_true), np.array(y_pred))
        recall = recall_score(np.array(y_true), np.array(y_pred))
        return accuracy, f1, precision, recall
    
    def evaluate_category_all(self, flag):
        """
        F1_c over all malware samples (archived + new).
        Uses `summary.type_name` as the predicted type and `gt_type` as ground truth.
        """
        archived_info_path = os.path.join(self.path_dict["info"], "archived_sample_info.json")
        latest_info_path = os.path.join(self.path_dict["info"], "latest_sample_info.json")

        with open(archived_info_path, "r") as f:
            archived_info = json.load(f)
        with open(latest_info_path, "r") as f:
            latest_info = json.load(f)

        malradar_apk_list = list(archived_info.keys())
        new_apk_list = list(latest_info.keys())

        behavior_folder = get_behavior_folder(self.path_dict, flag)

        y_pred_types = []
        y_gt_types = []

        for apk_name in tqdm(malradar_apk_list, desc="Evaluating type classification (malradar)"):
            behavior_report_path = os.path.join(behavior_folder, self.model_name, "malradar", f"{apk_name}.json")
            try:
                with open(behavior_report_path, "r") as f:
                    behavior_report = json.load(f)
            except FileNotFoundError:
                continue

            # Skip invalid verdicts: only use samples where model predicts malware
            if behavior_report.get("verdict") != "malware":
                continue

            summary = behavior_report.get("summary") or {}
            pred_type_name = summary.get("type_name")
            if not pred_type_name:
                continue

            pred_type = pred_type_name.lower()
            gt_type = behavior_report["gt_type"].lower()
            if pred_type not in [x.lower() for x in class_mapping.keys()]:
                pred_type = 6
            else:
                pred_type = class_mapping[pred_type]
            y_pred_types.append(pred_type)
            y_gt_types.append(class_mapping[gt_type])

        for apk_name in tqdm(new_apk_list, desc="Evaluating type classification (new)"):
            behavior_report_path = os.path.join(behavior_folder, self.model_name, "new", f"{apk_name}.json")
            try:
                with open(behavior_report_path, "r") as f:
                    behavior_report = json.load(f)
            except FileNotFoundError:
                continue

            # Skip invalid verdicts: only use samples where model predicts malware
            if behavior_report.get("verdict") != "malware":
                continue

            summary = behavior_report.get("summary") or {}
            pred_type_name = summary.get("type_name")
            if not pred_type_name:
                continue

            pred_type = pred_type_name.lower()
            gt_type = behavior_report["gt_type"].lower()
            if pred_type not in [x.lower() for x in class_mapping.keys()]:
                pred_type = 6
            else:
                pred_type = class_mapping[pred_type]
            y_pred_types.append(pred_type)
            y_gt_types.append(class_mapping[gt_type])
            
        accuracy = np.mean(np.array(y_pred_types) == np.array(y_gt_types))
        f1 = f1_score(np.array(y_gt_types), np.array(y_pred_types), average="macro", zero_division=0)
        precision = precision_score(np.array(y_gt_types), np.array(y_pred_types), average="macro", zero_division=0)
        recall = recall_score(np.array(y_gt_types), np.array(y_pred_types), average="macro", zero_division=0)
        return accuracy, f1, precision, recall

        
    def statistic(self, folder_name: str):
        if folder_name == "archived":
            json_name= "archived_sample_info.json"
        elif folder_name == "latest":
            json_name= "latest_sample_info.json"
        else:
            print(f"Invalid folder name: {folder_name}")
            exit()
        
        gt_info_path = os.path.join(self.path_dict["info"], json_name)
        with open(gt_info_path, "r") as f:
            gt_info = json.load(f)
        
        families = set([x["family"] for x in gt_info.values()])
        types = set([x["type"] for x in gt_info.values()])
        print(f"families: {families}, len: {len(families)}")
        print(f"types: {types}, len: {len(types)}")


if __name__ == "__main__":
    args = parse_args()
    path_dict = utils.get_path_dict()

    model_name = args.model
    flag = args.flag
    print(f"Evaluating {model_name} with flag={flag} on full datasets")
    cls_evaluator = ClassificationEvaluator(path_dict, model_name)

    tpmr = cls_evaluator.evaluate_tpmr_all(flag)
    fpcr = cls_evaluator.evaluate_fpcr_all(flag)
    accuracy, f1, precision, recall = cls_evaluator.evaluate_category_all(flag)

    result_folder = os.path.join(path_dict["result"], args.output_subdir, flag)
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f"{model_name}_{flag}.json"), "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "flag": flag,
                "fpcr": fpcr,
                "tpmr": tpmr,
                "type_accuracy": accuracy,
                "type_f1": f1,
                "type_precision": precision,
                "type_recall": recall,
                "num_failed_samples": len(cls_evaluator.failed_sha256),
            },
            f,
            indent=4,
        )

    if cls_evaluator.failed_sha256:
        with open(os.path.join(result_folder, f"{model_name}_{flag}_failed.json"), "w") as f_fail:
            json.dump(
                {
                    "model_name": model_name,
                    "flag": flag,
                    "num_failed_samples": len(cls_evaluator.failed_sha256),
                    "failed_sha256": cls_evaluator.failed_sha256,
                },
                f_fail,
                indent=4,
            )
