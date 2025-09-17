import json
import re
from typing import Set, Dict, List
import os
import utils
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math


class FunctionLevelAPIExtractor:
    def __init__(self, path_dict,model_name,folder_name,sensitive_apis_list, flag):
        self.model_name = model_name
        self.folder_name = folder_name
        self.sensitive_apis = sensitive_apis_list
        self.normalized_sensitive_apis = set()
        self.sensitive_api_lookup = {}
        self.flag = flag
        self._build_sensitive_api_lookup()
        self.path_dict = path_dict
        self.save_folder = os.path.join(self.path_dict["result"], "csr_score",self.flag, self.model_name, self.folder_name)
        os.makedirs(self.save_folder, exist_ok=True)
        self.max_workers = 24

    def load_extracted_sensitive_apis(self, apk_name: str):
        meta_info_path = os.path.join(self.path_dict["meta_info"], self.folder_name, f"{apk_name}.json")
        with open(meta_info_path, "r") as f:
            meta_info = json.load(f)
        sensitive_apis = meta_info["sensitive_apis"]
        return sensitive_apis
    
    def _build_sensitive_api_lookup(self):
        """Build normalized lookup tables (same as your existing logic)"""
        for api in self.sensitive_apis:
            normalized = api.replace(' ', '')
            self.normalized_sensitive_apis.add(normalized)
            
            if '->' in normalized:
                base = normalized.split('(')[0]
                if base not in self.sensitive_api_lookup:
                    self.sensitive_api_lookup[base] = set()
                self.sensitive_api_lookup[base].add(normalized)
    
    def extract_apis_from_function_code(self, function_code: str) -> Set[str]:
        """Extract API calls from decompiled function source code"""
        apis = set()
        
        method_pattern = r'([a-zA-Z_$][a-zA-Z0-9_$.]*?)\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        matches = re.findall(method_pattern, function_code)
        
        for class_part, method_name in matches:
            if '.' in class_part:
                class_name = f"L{class_part.replace('.', '/')}"
                if not class_name.endswith(';'):
                    class_name += ';'
            else:
                # Handle simple class names
                class_name = f"L{class_part};"
            
            api_signature = f"{class_name}->{method_name}"
            apis.add(api_signature)
        
        constructor_pattern = r'new\s+([a-zA-Z_$][a-zA-Z0-9_$.]*)\s*\('
        constructor_matches = re.findall(constructor_pattern, function_code)
        
        for class_name in constructor_matches:
            if '.' in class_name:
                normalized_class = f"L{class_name.replace('.', '/')};"
            else:
                normalized_class = f"L{class_name};"
            
            constructor_signature = f"{normalized_class}-><init>"
            apis.add(constructor_signature)
        
        static_pattern = r'([A-Z][a-zA-Z0-9_$.]*?)\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        static_matches = re.findall(static_pattern, function_code)
        
        for class_part, method_name in static_matches:
            if '.' in class_part:
                class_name = f"L{class_part.replace('.', '/')}"
            else:
                class_name = f"L{class_part}"
            
            if not class_name.endswith(';'):
                class_name += ';'
            
            static_signature = f"{class_name}->{method_name}"
            apis.add(static_signature)
        
        android_patterns = [
            r'getSystemService\s*\(',
            r'startActivity\s*\(',
            r'sendBroadcast\s*\(',
            r'getContentResolver\s*\(',
            r'openFileOutput\s*\(',
            r'openFileInput\s*\(',
        ]
        
        for pattern in android_patterns:
            if re.search(pattern, function_code):
                # Add common Android API signatures
                apis.add("Landroid/content/Context;->getSystemService")
                apis.add("Landroid/app/Activity;->startActivity")
                apis.add("Landroid/content/Context;->sendBroadcast")
                apis.add("Landroid/content/Context;->getContentResolver")
                apis.add("Landroid/content/Context;->openFileOutput")
                apis.add("Landroid/content/Context;->openFileInput")
        
        return apis
    
    def find_sensitive_apis_in_function(self, function_apis: Set[str]) -> Set[str]:
        """Find sensitive APIs from extracted function APIs"""
        if not self.normalized_sensitive_apis:
            return set()
        
        found_sensitive = set()
        
        for extracted_api in function_apis:
            normalized = extracted_api.replace(' ', '')
            
            # Exact match
            if normalized in self.normalized_sensitive_apis:
                found_sensitive.add(normalized)
                continue
            
            # Base method matching
            if '->' in normalized:
                base = normalized.split('(')[0]
                if base in self.sensitive_api_lookup:
                    found_sensitive.update(self.sensitive_api_lookup[base])
        
        return found_sensitive
    
    def analyze_all_functions(self, apk_name: str) -> Dict[str, Set[str]]:
        """Process JSONL file and extract sensitive APIs for each function"""
        results = {}
        json_file_path = os.path.join(self.path_dict["reachable_func"], self.folder_name, f"{apk_name}.json")

        gt_sensitive_apis = self.load_extracted_sensitive_apis(apk_name)
        gt_sensitive_apis = set(gt_sensitive_apis)
        print(f"Number of GT sensitive APIs: {len(gt_sensitive_apis)}")
        
        with open(json_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    function_data = json.loads(line.strip())
                    
                    for function_signature, source_code in function_data.items():
                        # Extract APIs from this function's source code
                        function_apis = self.extract_apis_from_function_code(source_code)
                        
                        # Find sensitive APIs
                        sensitive_apis = self.find_sensitive_apis_in_function(function_apis)
                        
                        if sensitive_apis:  # Only store functions with sensitive APIs
                            results[function_signature] = sensitive_apis
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing function: {str(e)}")
                    continue

        match_count = 0
        for function_signature, sensitive_apis in results.items():
            if sensitive_apis.intersection(gt_sensitive_apis):
                match_count += 1
        # print(f"Number of matched sensitive APIs: {match_count} in {len(gt_sensitive_apis)} GT sensitive APIs, Extracted sensitive APIs: {len(results)}")

        return results, match_count, len(results), len(gt_sensitive_apis)

    def _get_sensitive_functions(self, apk_name: str):
        json_file_path = os.path.join(self.path_dict["reachable_func"], self.folder_name, f"{apk_name}.json")
        with open(json_file_path, 'r', encoding='utf-8') as file:
            func_src = json.load(file)
        
        context_folder = os.path.join(self.path_dict["context_summary"], self.model_name, self.folder_name)
        context_file_path = os.path.join(context_folder, f"{apk_name}.jsonl")
        samples = []
        with open(context_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        # print(f"Number of samples: {len(samples)}")

        function_names, sensitivity_scores, selected_functions = [], [], []
        for sample in samples:
            function_names.append(sample["function"])
            sensitivity_scores.append(sample["sensitivity_score"])
        
        sensitivity_scores = np.array(sensitivity_scores)  # Convert to numpy array
        ranked_indices = np.argsort(sensitivity_scores)[::-1]
        mask_indices = sensitivity_scores[ranked_indices] >= 7
        selected_indices = ranked_indices[mask_indices]

        srcs = [func_src[function_names[i]] for i in selected_indices]
        function_names = [function_names[i] for i in selected_indices]

        for function_name, src in zip(function_names, srcs):
            selected_functions.append({function_name: src})
        
        # print(f"Number of selected functions: {len(selected_functions)}")
        return selected_functions, func_src

    def analyze_specific_functions(self, apk_name: str) -> Dict[str, Set[str]]:
        """Analyze a specific list of functions"""
        results = {}
        gt_sensitive_apis = self.load_extracted_sensitive_apis(apk_name)
        gt_sensitive_apis = set(gt_sensitive_apis)
        # print(f"Number of GT sensitive APIs: {len(gt_sensitive_apis)}")

        # sensitive functions are sensitive by 5 score or higher
        selected_functions, func_src = self._get_sensitive_functions(apk_name)

        for function_data in selected_functions:
            for function_signature, source_code in function_data.items():
                function_apis = self.extract_apis_from_function_code(source_code)
                sensitive_apis = self.find_sensitive_apis_in_function(function_apis)
                
                if sensitive_apis:
                    results[function_signature] = sensitive_apis

        match_count = 0
        for function_signature, sensitive_apis in results.items():
            if sensitive_apis.intersection(gt_sensitive_apis):
                match_count += 1
        # print(f"Number of matched sensitive APIs: {match_count} in {len(gt_sensitive_apis)} GT sensitive APIs, Extracted sensitive APIs: {len(results)}")

        stats_info = {
            "matched_sensitive_apis": match_count,
            "extracted_sensitive_apis": len(results),
            "gt_sensitive_apis": len(gt_sensitive_apis),
            "function_number": len(func_src),
            "selected_function_number": len(selected_functions),
        }

        csr = self._calculate_csr_score(stats_info)
        stats_info["csr"] = csr


        return results, stats_info

    def _calculate_csr_score(self, stats_info):
        coverage = stats_info["matched_sensitive_apis"] / (stats_info["gt_sensitive_apis"] if stats_info["gt_sensitive_apis"] > 0 else 1)
        selected_ratio = stats_info["selected_function_number"] / stats_info["function_number"]

        csr = coverage / selected_ratio if selected_ratio > 0 else 0

        return csr

    def analyze_all_apks(self,apk_list):
        
        print(f"Number of APKs: {len(apk_list)} needed to be analyzed")
        df_result = []
        csr_scores = []
        state_info_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.analyze_specific_functions, apk_name) for apk_name in tqdm(apk_list, desc=f"Analyzing all APKs for csr score, {self.model_name} in {self.folder_name}")]
            for future in as_completed(futures):
                results, stats_info = future.result()
                csr_scores.append(stats_info["csr"])
                state_info_list.append(stats_info)

        final_info = {"mean_csr": np.mean(csr_scores), "median_csr": np.median(csr_scores), "max_csr": np.max(csr_scores), "min_csr": np.min(csr_scores)}

        df_result = pd.DataFrame(state_info_list)
        df_result.to_csv(os.path.join(self.save_folder, f"{self.model_name}_{self.folder_name}_sensitive_info.csv"), index=False)
        with open(os.path.join(self.save_folder, f"{self.model_name}_{self.folder_name}_csr_final.json"), "w") as f:
            json.dump(final_info, f, indent=4)

def statistic(path_dict, model_name, flag):

    df_malradar = pd.read_csv(os.path.join(path_dict["result"], "csr_score", flag, model_name, "archived", f"{model_name}_archived_sensitive_info.csv"))
    df_new = pd.read_csv(os.path.join(path_dict["result"], "csr_score", flag, model_name, "latest", f"{model_name}_latest_sensitive_info.csv"))

    malradar_csr = df_malradar["csr"].tolist()
    new_csr = df_new["csr"].tolist()

    all_csr = malradar_csr + new_csr
    print(f"Number of CSR: {len(all_csr)} in {model_name} with {flag}")
    print(f"Mean CSR: {np.mean(all_csr)}")
    print(f"Median CSR: {np.median(all_csr)}")
    print(f"Max CSR: {np.max(all_csr)}")
    print(f"Min CSR: {np.min(all_csr)}")
    save_folder = os.path.join(path_dict["result"], "csr_score", flag, model_name)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f"{model_name}_{flag}_csr_statistic.json"), "w") as f:
        json.dump(all_csr, f, indent=4)
    return all_csr
        


if __name__ == "__main__":
    path_dict = utils.get_path_dict()
    sensitive_api_path = os.path.join(path_dict["info"], "sensitive_api.txt")
    with open(sensitive_api_path, "r") as f:
        sensitive_apis_list = f.read().splitlines()

    print(f"Number of Sensitive APIs: {len(sensitive_apis_list)}")

    model_name = "llama"
    flag = "context"
    folder_name = "archived"
    assert folder_name in ["archived", "latest"]
    apk_list = utils.check_file_exist(folder_name)
    extractor = FunctionLevelAPIExtractor(path_dict, model_name, folder_name, sensitive_apis_list, flag)
    extractor.analyze_all_apks(apk_list)
    
    # evaluate csr score for all archived and latest
    statistic(path_dict, model_name, flag)