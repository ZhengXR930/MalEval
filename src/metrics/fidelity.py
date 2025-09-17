import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from collections import defaultdict
import joblib  
import utils
from collections import Counter
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

class_mapping = {
    'trojan': 0,
    'banker': 1,
    'rootkit': 2,
    'spyware': 3,
    'adware': 4,
    'ransomware': 5
}

class Fidelity:
    def __init__(self, path_dict: dict, model_name: str, flag: str):
        self.path_dict = path_dict
        if flag == "context":
            self.context_summary_folder = self.path_dict["context_summary"]
        elif flag == "no_context":
            self.context_summary_folder = self.path_dict["no_context_summary"]
        else:
            raise ValueError(f"Invalid flag: {flag}")
        self.model_name = model_name
        
        # Create result directories
        result_dir = os.path.join(self.path_dict["result"], 'fidelity',self.model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        self.result_csv_path = None
        self.result_detail_csv_path = None
        
        self.save_model_path = os.path.join(self.path_dict["classifier"], f"{self.model_name}_classifier.pkl")
        self.vectorizer_path = os.path.join(self.path_dict["classifier"], f"{self.model_name}_vectorizer.pkl")
        self.scaler_path = os.path.join(self.path_dict["classifier"], f"{self.model_name}_scaler.pkl")
        self.result_folder = os.path.join(self.path_dict["result"], "fidelity", self.model_name)
        
        # Cache for loaded models
        self._classifier = None
        self._vectorizer = None

        self._detailed_results_saved = False

        self.train_apk_names = set()
        self.test_apk_names = set()

        self.folder_mapping = {
            "archived": 0,
            "latest": 1
        }
        
        self.reverse_folder_mapping = {v: k for k, v in self.folder_mapping.items()}

    def _get_sample_info_path(self, folder):
        if folder == "archived":
            return os.path.join(self.path_dict["info"], "archived_sample_info.json")
        else:  # malpedia or new
            return os.path.join(self.path_dict["info"], "latest_sample_info.json")

    def load_one_samples(self, folder, apk_name):
        """Load samples for a single APK with enhanced error handling"""
        assert folder in ["archived", "latest"], f"Invalid folder name: {folder}"
        context_summary_path = os.path.join(self.context_summary_folder, self.model_name, folder, f"{apk_name}.jsonl")
        
        samples = []
        try:
            with open(context_summary_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample)
        except FileNotFoundError:
            print(f"Warning: File not found {context_summary_path}")
            return [], None, [], None

        sample_info_path = self._get_sample_info_path(folder)
        try:
            with open(sample_info_path, 'r') as f:
                sample_info = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Sample info not found {sample_info_path}")
            return [], None, [], None

        if apk_name not in sample_info:
            print(f"Warning: APK {apk_name} not found in sample info")
            return [], None, [], None

        type_label = class_mapping[sample_info[apk_name]['type']]
        
        text_feature = []
        sensitivity_score = []
        for sample in samples:
            if 'function' in sample and 'summary' in sample and 'sensitivity_score' in sample:
                text_feature.append(f"{sample['function']} {sample['summary']}")
                sensitivity_score.append(sample['sensitivity_score'])
        
        folder_source = self.folder_mapping[folder]
        return text_feature, type_label, sensitivity_score, folder_source

    def _load_all_apk_data(self, folder):
        folder_path = os.path.join(self.context_summary_folder, self.model_name, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found {folder_path}")
            return {}
            
        apk_files = [x for x in os.listdir(folder_path) if x.endswith('.jsonl')]
        apk_data = {}
        
        for apk_file in tqdm(apk_files, desc=f"Loading all APKs from {folder}"):
            apk_name = os.path.basename(apk_file).replace(".jsonl", "")
            text_feature, type_label, sensitivity_score, source = self.load_one_samples(folder, apk_name)
            
            if text_feature:  # Only include APKs with valid data
                apk_data[apk_name] = {
                    'text_features': text_feature,
                    'type_label': type_label,
                    'sensitivity_scores': sensitivity_score,
                    'source': source
                }
        
        return apk_data
    
    def _load_all_apk_data_from_all_folders(self):

        malradar_apk_data = self._load_all_apk_data("archived")
        new_apk_data = self._load_all_apk_data("latest")
        
        all_apk_data = {**malradar_apk_data, **new_apk_data}
        return all_apk_data


    def _load_models(self):
        if self._classifier is None:
            self._classifier = joblib.load(self.save_model_path)
        if self._vectorizer is None:
            self._vectorizer = joblib.load(self.vectorizer_path)
        return self._classifier, self._vectorizer

    def _cal_fnr_fpr(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        
        fnr_list, fpr_list = {}, {}

        for label in unique_labels:
            class_name = list(class_mapping.keys())[label]
            
            TP = cm[label, label] if label < cm.shape[0] and label < cm.shape[1] else 0
            FN = cm[label, :].sum() - TP if label < cm.shape[0] else 0
            FP = cm[:, label].sum() - TP if label < cm.shape[1] else 0
            TN = cm.sum() - (TP + FN + FP)

            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
            
            print(f"class {class_name}: FNR: {float(fnr):.4f}, FPR: {float(fpr):.4f}")

            fnr_list[class_name] = fnr
            fpr_list[class_name] = fpr

        return fnr_list, fpr_list, np.mean(list(fnr_list.values())), np.mean(list(fpr_list.values()))

    def _combine_text_features(self, text_features):
        if not text_features:
            return ""
        
        return ' '.join(text_features)

    def classify_model(self, test_size=0.2, random_state=42):
        cls_result = {}
        
        print(f"Loading all APK data from archived and latest...")
        apk_data = self._load_all_apk_data_from_all_folders()
        
        if not apk_data:
            print("Warning: No APK data loaded")
            return cls_result, [], []
        
        print(f'Total APKs loaded: {len(apk_data)}')
        
        apk_names = list(apk_data.keys())
        apk_labels = [apk_data[apk]['type_label'] for apk in apk_names]
        
        print(f'APK label distribution: {np.unique(apk_labels, return_counts=True)}')
        
        train_apk_names, test_apk_names = train_test_split(
            apk_names, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=apk_labels
        )
        
        self.train_apk_names = set(train_apk_names)
        self.test_apk_names = set(test_apk_names)
        
        print(f"Training APKs: {len(self.train_apk_names)}")
        print(f"Test APKs: {len(self.test_apk_names)}")
        
        train_text_concatenated = []
        train_type_labels = []
        test_text_concatenated = []
        test_type_labels = []
        
        for apk_name in train_apk_names:
            apk_info = apk_data[apk_name]
            combined_text = self._combine_text_features(apk_info['text_features'])
            train_text_concatenated.append(combined_text)
            train_type_labels.append(apk_info['type_label'])
        
        for apk_name in test_apk_names:
            apk_info = apk_data[apk_name]
            combined_text = self._combine_text_features(apk_info['text_features'])
            test_text_concatenated.append(combined_text)
            test_type_labels.append(apk_info['type_label'])
        
        print(f"Training samples: {len(train_text_concatenated)}")
        print(f"Test samples: {len(test_text_concatenated)}")

        train_score_features = []
        for apk_name in train_apk_names:
            scores = apk_data[apk_name]['sensitivity_scores']
            if scores:
                train_score_features.append([np.mean(scores), np.max(scores)])
            else:
                train_score_features.append([0, 0])

        test_score_features = []
        for apk_name in test_apk_names:
            scores = apk_data[apk_name]['sensitivity_scores']
            if scores:
                test_score_features.append([np.mean(scores), np.max(scores)])
            else:
                test_score_features.append([0, 0])
        
        if os.path.exists(self.vectorizer_path):
            print(f"Loading existing vectorizer: {self.vectorizer_path}")
            vectorizer = joblib.load(self.vectorizer_path)
            X_train = vectorizer.transform(train_text_concatenated)
            X_test = vectorizer.transform(test_text_concatenated)
        else:
            print("Creating new vectorizer on training data only")
            vectorizer = TfidfVectorizer(
                analyzer="word", 
                ngram_range=(1,2), 
                max_features=10000,
                norm=None,
                sublinear_tf=True
            )
            X_train = vectorizer.fit_transform(train_text_concatenated)
            X_test = vectorizer.transform(test_text_concatenated)

        scaler = StandardScaler()
        train_score_features = scaler.fit_transform(train_score_features)
        test_score_features = scaler.transform(test_score_features)

        joblib.dump(scaler, self.scaler_path)

        X_train_final = np.hstack((X_train.toarray(), train_score_features))
        X_test_final = np.hstack((X_test.toarray(), test_score_features))
                
        print(f"Training feature dimension: {X_train_final.shape}")
        print(f"Test feature dimension: {X_test_final.shape}")

        X_train_resampled, y_train_resampled = SMOTE(
            sampling_strategy='auto', 
            random_state=42
        ).fit_resample(X_train_final, train_type_labels)
        
        print(f"After SMOTE - Training samples: {X_train_resampled.shape[0]}")
        print(f"After SMOTE - Label distribution: {np.unique(y_train_resampled, return_counts=True)}")

        if os.path.exists(self.save_model_path):
            print(f"Loading existing model: {self.save_model_path}")
            classifier = joblib.load(self.save_model_path)
        else:
            print("Training new model")
            classifier = XGBClassifier(
                n_estimators=100,
                max_depth=4,          
                learning_rate=0.2,    
                subsample=0.8,        
                colsample_bytree=0.8, 
                reg_alpha=0.1,        
                reg_lambda=0.5,       
                random_state=42,
                n_jobs=-1
            )

            classifier.fit(X_train_resampled, y_train_resampled)

        y_pred_test = classifier.predict(X_test_final)
        y_pred_proba_test = classifier.predict_proba(X_test_final)
        
        f1 = f1_score(test_type_labels, y_pred_test, average='macro')
        precision = precision_score(test_type_labels, y_pred_test, average='macro')
        recall = recall_score(test_type_labels, y_pred_test, average='macro')
        accuracy = accuracy_score(test_type_labels, y_pred_test)

        print(f"\n{'='*60}")
        print(f"TEST SET EVALUATION (APK-level split, no data leakage)")
        print(f"{'='*60}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 score: {f1:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        
        max_probs = y_pred_proba_test.max(axis=1)
        print(f"\nConfidence Distribution on Test Set:")
        print(f"  Mean confidence: {max_probs.mean():.4f}")
        print(f"  Std confidence: {max_probs.std():.4f}")
        print(f"  Min confidence: {max_probs.min():.4f}")
        print(f"  Max confidence: {max_probs.max():.4f}")
        print(f"  Samples with >0.99 confidence: {(max_probs > 0.99).sum()}/{len(max_probs)}")
        print(f"  Samples with >0.95 confidence: {(max_probs > 0.95).sum()}/{len(max_probs)}")
        
        fnr_list, fpr_list, macro_avg_fnr, macro_avg_fpr = self._cal_fnr_fpr(test_type_labels, y_pred_test)

        joblib.dump(classifier, self.save_model_path)
        joblib.dump(vectorizer, self.vectorizer_path)

        self.result_detail_csv_path = os.path.join(
            self.result_folder, 
            f"{self.model_name}_classifier.csv"
        )
        
        if not os.path.exists(self.result_detail_csv_path):
            with open(self.result_detail_csv_path, 'w') as f:
                f.write("model,class,f1,precision,recall,fnr,fpr\n")

            f1_list = f1_score(test_type_labels, y_pred_test, average=None)
            precision_list = precision_score(test_type_labels, y_pred_test, average=None)
            recall_list = recall_score(test_type_labels, y_pred_test, average=None)
            
            with open(self.result_detail_csv_path, 'a') as f:
                for j, class_name in enumerate(class_mapping.keys()):
                    if j < len(f1_list) and class_name in fpr_list:
                        f.write(f"{self.model_name},{class_name},"
                                f"{f1_list[j]:.4f},"
                                f"{precision_list[j]:.4f},"
                                f"{recall_list[j]:.4f},"
                                f"{fnr_list[class_name]:.4f},"
                                f"{fpr_list[class_name]:.4f}\n")
            
            print(f"Detailed test results saved to: {self.result_detail_csv_path}")

        cls_result.update({
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "macro_avg_fnr": macro_avg_fnr,
            "macro_avg_fpr": macro_avg_fpr
        })

        return cls_result, list(test_apk_names), apk_data

    
    def get_fidelity(self, test_apk_names: list, apk_data: dict, top_k: int = 10):
        """
        FIXED: Calculate fidelity with improved logic and proper data leakage prevention
        
        Args:
            test_apk_names: List of test APK names
            folder: Folder name
            apk_data: Pre-loaded APK data
            top_k: Percentage of top functions to remove
        """
        classifier, vectorizer = self._load_models()
        
        detailed_result_csv_path = os.path.join(
            self.result_folder, 
            f"{self.model_name}_topk_{top_k}_fidelity_detail_result.csv"
        )

        if not os.path.exists(detailed_result_csv_path):
            with open(detailed_result_csv_path, 'w') as f:
                f.write("model,apk_name,sample_idx,n_funcs,removed,true_class,orig_pred_class,reduced_pred_class,"
                    "orig_prob,reduced_prob,decision_change,fidelity,fidelity_correct\n")

        print(f"Calculating fidelity for {len(test_apk_names)} TEST APKs only (top-{top_k}%)")
        print("Ensuring NO training data contamination...")

        if os.path.exists(self.scaler_path):
            scaler = joblib.load(self.scaler_path)
        else:
            print("Warning: Scaler not found. Creating a new one.")
            scaler = StandardScaler()
        
        fidelity_scores = []
        decision_change_scores = []
        correct_fidelity_scores = []  
        detailed_info = []
        skipped_count = 0

        for sample_idx, apk_name in enumerate(tqdm(test_apk_names, desc=f"Fidelity (top-{top_k}%)")):
            if apk_name in self.train_apk_names:
                print(f"ERROR: {apk_name} was in training set! Skipping to prevent data leakage.")
                skipped_count += 1
                continue
            
            if apk_name not in apk_data:
                print(f"Warning: {apk_name} not found in APK data")
                continue
                
            apk_info = apk_data[apk_name]
            text_features = apk_info['text_features']
            sensitivity_scores = apk_info['sensitivity_scores']
            type_label = apk_info['type_label']
            
            if not text_features:
                print(f"Warning: No text features for {apk_name}")
                fidelity_scores.append(0.0)
                continue

            original_apk_text = self._combine_text_features(text_features)
            
            if not original_apk_text.strip():
                print(f"Warning: Empty text for {apk_name}")
                fidelity_scores.append(0.0)
                continue
                
            original_vector = vectorizer.transform([original_apk_text])

            score_features = [[np.mean(sensitivity_scores), np.max(sensitivity_scores)]]
            score_features_scaled = scaler.transform(score_features)

            original_features = np.hstack((original_vector.toarray(), score_features_scaled))
            original_pred_proba = classifier.predict_proba(original_features)[0]
            predicted_class_index = np.argmax(original_pred_proba)
            original_prob = original_pred_proba[predicted_class_index]

            is_correct_prediction = (predicted_class_index == type_label)

            if len(sensitivity_scores) != len(text_features):
                print(f"Warning: Mismatch between sensitivity scores and text features for {apk_name}")
                continue
                
            ranked_indices = np.argsort(np.array(sensitivity_scores))[::-1]  # High to low
            num_functions_to_remove = max(10, int(top_k/100 * len(ranked_indices)))
            remaining_indices = ranked_indices[num_functions_to_remove:]
            
            if len(remaining_indices) > 0:
                remaining_functions = [text_features[i] for i in remaining_indices]
                remaining_scores = [sensitivity_scores[i] for i in remaining_indices]
                reduced_apk_text = self._combine_text_features(remaining_functions)
                
                if reduced_apk_text.strip():
                    reduced_vector = vectorizer.transform([reduced_apk_text])

                    reduced_score_features = [[np.mean(remaining_scores), np.max(remaining_scores)]]
                    reduced_score_features_scaled = scaler.transform(reduced_score_features)

                    reduced_features = np.hstack((reduced_vector.toarray(), reduced_score_features_scaled))

                    reduced_pred_proba = classifier.predict_proba(reduced_features)[0]
                    reduced_class_index = np.argmax(reduced_pred_proba)

                    reduced_prob = reduced_pred_proba[predicted_class_index]
                else:
                    reduced_class_index = np.argmax([1.0/len(classifier.classes_)] * len(classifier.classes_))
                    reduced_prob = 1.0 / len(classifier.classes_)
            else:
                reduced_class_index = np.argmax([1.0/len(classifier.classes_)] * len(classifier.classes_))
                reduced_prob = 1.0 / len(classifier.classes_)

            if original_prob > 1e-9:
                fidelity = abs(original_prob - reduced_prob) / (original_prob)
            else:
                fidelity = 0.0

            decision_change = 1.0 if predicted_class_index != reduced_class_index else 0.0

            fidelity = min(1.0, max(0.0, fidelity))
            
            fidelity_scores.append(fidelity)
            decision_change_scores.append(decision_change)
            
            if is_correct_prediction:
                correct_fidelity_scores.append(fidelity)

            correct_fidelity_str = f"{fidelity:.4f}" if is_correct_prediction else "NA"

            with open(detailed_result_csv_path, 'a') as f:
                f.write(f"{self.model_name},{apk_name},{sample_idx},{len(text_features)},"
                        f"{num_functions_to_remove},{type_label},{predicted_class_index},{reduced_class_index},"
                        f"{original_prob:.4f},{reduced_prob:.4f},{decision_change:.1f},"
                        f"{fidelity:.4f},{correct_fidelity_str}\n")

            if sample_idx < 5:
                detailed_info.append({
                    'apk_name': apk_name, 
                    'sample_idx': sample_idx, 
                    'n_funcs': len(text_features), 
                    'removed': num_functions_to_remove, 
                    'true_class': type_label,
                    'orig_pred_class': predicted_class_index,
                    'reduced_pred_class': reduced_class_index,
                    'orig_prob': original_prob, 
                    'reduced_prob': reduced_prob, 
                    'decision_change': decision_change,
                    'fidelity': fidelity,
                    'correct_fidelity': fidelity if is_correct_prediction else 'NA'
                })

        if skipped_count > 0:
            print(f"WARNING: Skipped {skipped_count} APKs due to training set contamination!")
        
        print("\n--- Fidelity Calculation Debug Info (First 5 Test Samples) ---")
        for info in detailed_info:
            print(f"APK {info['apk_name']} ({info['n_funcs']} funcs, removed {info['removed']}): "
                f"True={info['true_class']}, Pred: {info['orig_pred_class']} -> {info['reduced_pred_class']}, "
                f"Prob: {info['orig_prob']:.4f} -> {info['reduced_prob']:.4f}, "
                f"DecisionChange: {info['decision_change']}, Fidelity: {info['fidelity']:.4f}")

        if fidelity_scores:
            mean_fidelity = np.mean(fidelity_scores)
            std_fidelity = np.std(fidelity_scores)
            
            decision_change_rate = np.mean(decision_change_scores)

            if correct_fidelity_scores:
                mean_correct_fidelity = np.mean(correct_fidelity_scores)
                std_correct_fidelity = np.std(correct_fidelity_scores)
                correct_count = len(correct_fidelity_scores)
            else:
                mean_correct_fidelity = None
                std_correct_fidelity = None
                correct_count = 0
            
            print(f"\nTEST SET Fidelity Results (top-{top_k}%):")
            print(f"  Traditional Fidelity: {mean_fidelity:.4f} (+/- {std_fidelity:.4f})")
            print(f"  Decision Change Rate: {decision_change_rate:.4f}")
            
            if correct_count > 0:
                print(f"  Correct Prediction Fidelity: {mean_correct_fidelity:.4f} (+/- {std_correct_fidelity:.4f})")
            else:
                print(f"  Correct Prediction Fidelity: N/A (no correct predictions)")
                
            print(f"  Correct Predictions: {correct_count}/{len(fidelity_scores)}")
            print(f"  Valid test samples: {len(fidelity_scores)}")
        else:
            print("ERROR: No valid fidelity scores calculated!")

        return {
            'traditional_fidelity': fidelity_scores,
            'decision_change': decision_change_scores,
            'correct_fidelity': correct_fidelity_scores
        }

    def fidelity_evaluation(self, top_k_list: list = [1, 5, 10]):

        cls_result, test_apk_names, apk_data = self.classify_model(
            test_size=0.2, random_state=42
        )
        
        print(f"\nTraining completed with APK-level split!")
        print(f"  Training APKs: {len(self.train_apk_names)}")
        print(f"  Test APKs: {len(self.test_apk_names)}")
        print("Now calculating fidelity ONLY on test set...")

        self.result_csv_path = os.path.join(
            self.result_folder, 
            f"{self.model_name}_fidelity_result.csv"
        )

        if not os.path.exists(self.result_csv_path):
            with open(self.result_csv_path, 'w') as f:
                f.write("model_name,top_k,mean_fidelity,std_fidelity,"
                    "macro_avg_fnr,macro_avg_fpr,f1,precision,recall,test_samples,train_samples\n")
        
        all_results = {}
        for top_k in top_k_list:
            print(f"\n{'='*60}")
            print(f"Evaluating fidelity with top-{top_k}% function removal (TEST SET ONLY)")
            print(f"{'='*60}")
            
            result = self.get_fidelity(test_apk_names, apk_data, top_k)

            fidelity_scores = result['traditional_fidelity']
            
            if not fidelity_scores:
                print(f"Skipping top-{top_k}% due to no valid fidelity scores.")
                continue

            mean_fidelity = np.mean(fidelity_scores)
            std_fidelity = np.std(fidelity_scores)
            
            # Save results
            with open(self.result_csv_path, 'a') as f:
                f.write(f"{self.model_name},{top_k},{mean_fidelity:.4f},{std_fidelity:.4f},"
                        f"{cls_result['macro_avg_fnr']:.4f},{cls_result['macro_avg_fpr']:.4f},"
                        f"{cls_result['f1']:.4f},{cls_result['precision']:.4f},{cls_result['recall']:.4f},"
                        f"{len(self.test_apk_names)},{len(self.train_apk_names)}\n")
            
            all_results[top_k] = {
                'fidelity_scores': fidelity_scores,
                'mean_fidelity': mean_fidelity,
                'std_fidelity': std_fidelity
            }
        
        print(f"\nFidelity evaluation complete!")
        print(f"Results saved to: {self.result_csv_path}")
        print(f"Test samples used: {len(self.test_apk_names)}")
        print(f"Training samples: {len(self.train_apk_names)}")
        
        return {
            'classification_results': cls_result,
            'fidelity_results': all_results,
            'test_apk_names': list(self.test_apk_names),
            'train_apk_names': list(self.train_apk_names)
        }

if __name__ == "__main__":
    path_dict = utils.get_path_dict()
    model_name = "llama"
    flag = "context"

    fidelity = Fidelity(path_dict, model_name, flag)
    results = fidelity.fidelity_evaluation(top_k_list=[10])
    print(results)