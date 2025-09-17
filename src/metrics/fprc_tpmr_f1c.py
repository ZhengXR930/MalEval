import json
import os
import utils
import numpy as np
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

class ClassificationEvaluator:
    def __init__(self, path_dict, model_name):
        self.model_name = model_name

        self.path_dict = path_dict
        self.save_folder = os.path.join(self.path_dict["info"], "fprc_tpmr_f1c")

    def load_behavior_report(self, apk_name, flag, folder_name):
        if flag == "context":
            behavior_folder = self.path_dict["behavior"]
        elif flag == "no_context":
            behavior_folder = self.path_dict["no_context_behavior"]
        elif flag == "meta":
            behavior_folder = self.path_dict["meta_behavior"]
        else:
            raise ValueError(f"Invalid flag: {flag}")

        behavior_report_path = os.path.join(behavior_folder, self.model_name, folder_name, f"{apk_name}.json")
        with open(behavior_report_path, "r") as f:
            behavior_report = json.load(f)
        
        y_pred = 1 if behavior_report["is_malicious"] else 0
        
        return y_pred

    

    def apk_filter(self, apk_list, flag, folder_name):
        if flag == "context":
            behavior_folder = self.path_dict["behavior"]
        elif flag == "no_context":
            behavior_folder = self.path_dict["no_context_behavior"]
        elif flag == "meta":
            behavior_folder = self.path_dict["meta_behavior"]
        else:
            raise ValueError(f"Invalid flag: {flag}")

        to_remove = []
        for apk_name in apk_list:
            behavior_report_path = os.path.join(behavior_folder, self.model_name, folder_name, f"{apk_name}.json")
            with open(behavior_report_path, "r") as f:
                behavior_report = json.load(f)
            if "is_malicious" not in behavior_report:
                print(f"is_malicious not in {behavior_report_path}")
            if behavior_report["is_malicious"] == False:
                to_remove.append(apk_name)
        print(f"Number of APKs to remove: {len(to_remove)}")
        apk_list = [apk_name for apk_name in apk_list if apk_name not in to_remove]
        return apk_list

    def evaluate_tpmr(self, apk_list, flag, folder_name):

        assert folder_name in ["archived", "latest"]
        y_preds = []
        for apk_name in tqdm(apk_list, desc="Evaluating TPMR"):
            y_pred = self.load_behavior_report(apk_name, flag, folder_name)
            y_preds.append(y_pred)

        
        y_true = [1] * len(apk_list)
        tpir = np.mean(np.array(y_preds) != np.array(y_true))
        return tpir
    
    def evaluate_tpmr_all(self, flag):
        archived_apk_list = utils.check_file_exist("archived")
        latest_apk_list = utils.check_file_exist("latest")
        y_preds = []
        for apk_name in tqdm(archived_apk_list, desc="Evaluating TPMR"):
            y_pred = self.load_behavior_report(apk_name, flag, "malradar")
            y_preds.append(y_pred)
            
        for apk_name in tqdm(latest_apk_list, desc="Evaluating TPMR"):
            y_pred = self.load_behavior_report(apk_name, flag, "new")
            y_preds.append(y_pred)

        y_true = [1] * len(archived_apk_list) + [1] * len(latest_apk_list)
        tpir = np.mean(np.array(y_preds) != np.array(y_true))
        return tpir

    def evaluate_fpcr(self, apk_list, flag, folder_name):
        
        assert folder_name in ["benign"]

        y_preds = []
        for apk_name in tqdm(apk_list, desc="Evaluating FPCR"):
            y_pred = self.load_behavior_report(apk_name, flag, folder_name)
            y_preds.append(y_pred)
        
        y_true = [0] * len(apk_list)
        fpcr = np.mean(np.array(y_preds) != np.array(y_true))

        return fpcr

    def evaluate_accuracy(self, flag: str):
        apk_list_archived = utils.check_file_exist("archived")
        apk_list_latest = utils.check_file_exist("latest")
        apk_list_benign = utils.check_file_exist("benign")
        
        y_pred = []
        for apk_name in tqdm(apk_list_archived, desc="Evaluating accuracy"):
            y_pred.append(self.load_behavior_report(apk_name, flag, "malradar")[0])

        for apk_name in tqdm(apk_list_latest, desc="Evaluating accuracy"):
            y_pred.append(self.load_behavior_report(apk_name, flag, "new")[0])
        
        for apk_name in tqdm(apk_list_benign, desc="Evaluating accuracy"):
            y_pred.append(self.load_behavior_report(apk_name, flag, "benign")[0])
        
        y_true = [1] * len(apk_list_archived) + [1] * len(apk_list_latest) + [0] * len(apk_list_benign)
        accuracy = np.mean(np.array(y_pred) == np.array(y_true))
        f1 = f1_score(np.array(y_true), np.array(y_pred))
        precision = precision_score(np.array(y_true), np.array(y_pred))
        recall = recall_score(np.array(y_true), np.array(y_pred))
        return accuracy, f1, precision, recall
    
    def evaluate_category_classification(self, apk_list, flag, folder_name):

        assert folder_name in ["archived", "latest"]

        apk_list = self.apk_filter(apk_list, flag, folder_name)

        if flag == "context":
            behavior_folder = self.path_dict["behavior"]
        elif flag == "no_context":
            behavior_folder = self.path_dict["no_context_behavior"]
        elif flag == "meta":
            behavior_folder = self.path_dict["meta_behavior"]
        else:
            raise ValueError(f"Invalid flag: {flag}")

        y_pred_types = []
        y_gt_types = []
        for apk_name in tqdm(apk_list, desc="Evaluating category classification"):
            behavior_report_path = os.path.join(behavior_folder, self.model_name, folder_name, f"{apk_name}.json")
            with open(behavior_report_path, "r") as f:
                behavior_report = json.load(f)

            pred_type = behavior_report["type"].lower()
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

    def evaluate_category_all(self,flag):

        archived_apk_list = utils.check_file_exist("archived")
        latest_apk_list = utils.check_file_exist("latest")

        archived_apk_list = self.apk_filter(archived_apk_list, flag, "archived")
        latest_apk_list = self.apk_filter(latest_apk_list, flag, "latest")


        if flag == "context":
            behavior_folder = self.path_dict["behavior"]
        elif flag == "no_context":
            behavior_folder = self.path_dict["no_context_behavior"]
        elif flag == "meta":
            behavior_folder = self.path_dict["meta_behavior"]
        else:
            raise ValueError(f"Invalid flag: {flag}")

        y_pred_types = []
        y_gt_types = []

        for apk_name in tqdm(archived_apk_list, desc="Evaluating type classification"):
            behavior_report_path = os.path.join(behavior_folder, self.model_name, "archived", f"{apk_name}.json")
            with open(behavior_report_path, "r") as f:
                behavior_report = json.load(f)

            pred_type = behavior_report["type"].lower()
            gt_type = behavior_report["gt_type"].lower()
            if pred_type not in [x.lower() for x in class_mapping.keys()]:
                pred_type = 6
            else:
                pred_type = class_mapping[pred_type]
            y_pred_types.append(pred_type)
            y_gt_types.append(class_mapping[gt_type])

        for apk_name in tqdm(latest_apk_list, desc="Evaluating type classification"):
            behavior_report_path = os.path.join(behavior_folder, self.model_name, "latest", f"{apk_name}.json")
            with open(behavior_report_path, "r") as f:
                behavior_report = json.load(f)

            pred_type = behavior_report["type"].lower()
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
        types = set([x["category"] for x in gt_info.values()])
        print(f"families: {families}, len: {len(families)}")
        print(f"types: {types}, len: {len(types)}")


if __name__ == "__main__":
    path_dict = utils.get_path_dict()

    model_list = ["gemini","gpt","claude","qwen","deepseek","llama","coder"]
    folder_name_list = ["archived", "latest", "benign"]
    flag_list = ["no_context", "context", "meta"]
    for model_name in model_list:
        for flag in flag_list:
            for folder_name in folder_name_list:
                apk_list = utils.check_file_exist(folder_name)
                result_folder = os.path.join(path_dict["result"], "fprc_tpmr_f1c", flag)
                print(f"Evaluating {model_name} in {folder_name} with {flag}")
                cls_evaluator = ClassificationEvaluator(path_dict, model_name)
                if folder_name == "benign":
                    fpcr = cls_evaluator.evaluate_fpcr(apk_list, flag, folder_name)
                    accuracy, f1, precision, recall = 0, 0, 0, 0
                    tpmr = 0
                else:
                    fpcr = 0
                    accuracy, f1, precision, recall = cls_evaluator.evaluate_category_classification(apk_list, flag, folder_name)
                    tpmr = cls_evaluator.evaluate_tpmr(apk_list, flag, folder_name)
                
                os.makedirs(result_folder, exist_ok=True)
                with open(os.path.join(result_folder, f"{model_name}_{folder_name}_{flag}.json"), "w") as f:
                    json.dump({
                        "model_name": model_name,
                        "folder_name": folder_name,
                        "flag": flag,
                        "fpcr": fpcr, 
                        "tpmr": tpmr,
                        "type_accuracy": accuracy,
                        "type_f1": f1,
                        "type_precision": precision,
                        "type_recall": recall}, f, indent=4)

    

        
