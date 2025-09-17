import json
import os
import sys
from typing import Set, List, Tuple, Dict, Any
from androguard.misc import AnalyzeAPK
from tqdm import tqdm
import utils
from loguru import logger
logger.remove()
import pandas as pd
import ast
import gc

class MetaInfoExtractor:
    def __init__(self, path_dict, flag='malradar', sensitive_api_file: str = None, sensitive_perm_file: str = None):
        self.path_dict = path_dict
        self.flag = flag
        self.sensitive_apis = set()
        self.sensitive_perms = set()
        self.output_folder = self.path_dict["meta_info"]
        
        # Pre-compute normalized sensitive APIs for faster matching
        self.normalized_sensitive_apis = set()
        self.sensitive_api_lookup = {}  # For O(1) base method lookup
        
        if sensitive_api_file:
            self.load_sensitive_api_list(sensitive_api_file)

        if sensitive_perm_file:
            self.load_sensitive_perm_list(sensitive_perm_file)

        if flag == 'malradar':   
            self.apk_folder = self.path_dict["apk"]
        elif flag == 'benign':
            self.apk_folder = self.path_dict["benign_apk"]
        elif flag == 'new':
            self.apk_folder = self.path_dict["new_apk"]
        else:
            raise ValueError(f"Invalid flag: {flag}")

    def load_sensitive_api_list(self, file_path: str):
        """Load sensitive API list from external text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.sensitive_apis.add(line)
                        # Pre-normalize for faster matching
                        normalized = line.replace(' ', '')
                        self.normalized_sensitive_apis.add(normalized)
                        
                        # Create base method lookup for O(1) access
                        base = normalized.split('(')[0] if '(' in normalized else normalized
                        if base not in self.sensitive_api_lookup:
                            self.sensitive_api_lookup[base] = []
                        self.sensitive_api_lookup[base].append(normalized)
            
            print(f"Loaded {len(self.sensitive_apis)} sensitive APIs from {file_path}")
            
        except FileNotFoundError:
            print(f"Error: Sensitive API file not found: {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading sensitive API file: {str(e)}")
            sys.exit(1)

    def load_sensitive_perm_list(self, file_path: str):
        """Load sensitive permissions list from external text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.sensitive_perms.add(line)
            
            print(f"Loaded {len(self.sensitive_perms)} sensitive permissions from {file_path}")
            
        except FileNotFoundError:
            print(f"Error: Sensitive permissions file not found: {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading sensitive permissions file: {str(e)}")
            sys.exit(1)

    def analyze_apk(self, apk_path: str):
        """Analyze APK file using Androguard"""
        try:
            print(f"Analyzing APK: {apk_path}")
            apk, dex_files, analysis = AnalyzeAPK(apk_path)
            return apk, dex_files, analysis
        except Exception as e:
            print(f"Error analyzing APK {apk_path}: {str(e)}")
            return None, None, None

    def extract_package_info(self, apk) -> Dict[str, str]:
        """Extract package name and application name"""
        try:
            package_info = {
                'package_name': apk.get_package() or 'N/A',
                'application_name': apk.get_app_name() or 'N/A',
                'version_name': apk.get_androidversion_name() or 'N/A',
                'version_code': str(apk.get_androidversion_code()) or 'N/A'
            }
            return package_info
        except Exception as e:
            print(f"Error extracting package info: {str(e)}")
            return {
                'package_name': 'N/A',
                'application_name': 'N/A', 
                'version_name': 'N/A',
                'version_code': 'N/A'
            }
    
    def extract_activities(self, apk) -> Dict[str, List[str]]:
        """Extract all activities"""
        try:
            activities = {
                'all_activities': apk.get_activities(),
                'main_activity': apk.get_main_activity() or 'N/A',
                'exported_activities': []
            }
            
            # Try to identify exported activities
            try:
                for activity in activities['all_activities']:
                    try:
                        intent_filters = apk.get_intent_filters('activity', activity)
                        if intent_filters:
                            activities['exported_activities'].append(activity)
                    except:
                        continue
            except:
                pass
                
            return activities
        except Exception as e:
            print(f"Error extracting activities: {str(e)}")
            return {
                'all_activities': [],
                'main_activity': 'N/A',
                'exported_activities': []
            }

    def extract_services(self, apk) -> Dict[str, List[str]]:
        """Extract all services"""
        try:
            services = {
                'all_services': apk.get_services(),
                'exported_services': []
            }
            
            # Try to identify exported services
            try:
                for service in services['all_services']:
                    try:
                        intent_filters = apk.get_intent_filters('service', service)
                        if intent_filters:
                            services['exported_services'].append(service)
                    except:
                        continue
            except:
                pass
                
            return services
        except Exception as e:
            print(f"Error extracting services: {str(e)}")
            return {
                'all_services': [],
                'exported_services': []
            }

    def extract_receivers(self, apk) -> Dict[str, List[str]]:
        """Extract all broadcast receivers"""
        try:
            receivers = {
                'all_receivers': apk.get_receivers(),
                'exported_receivers': []
            }
            
            # Try to identify exported receivers
            try:
                for receiver in receivers['all_receivers']:
                    try:
                        intent_filters = apk.get_intent_filters('receiver', receiver)
                        if intent_filters:
                            receivers['exported_receivers'].append(receiver)
                    except:
                        continue
            except:
                pass
                
            return receivers
        except Exception as e:
            print(f"Error extracting receivers: {str(e)}")
            return {
                'all_receivers': [],
                'exported_receivers': []
            }

    def extract_certificate_info(self, apk) -> Dict[str, Any]:
        """Extract certificate information"""
        try:
            cert_info = {
                'is_signed': apk.is_signed(),
                'certificates': []
            }
            
            if apk.is_signed():
                try:
                    certificates = apk.get_certificates()
                    for i, cert in enumerate(certificates):
                        try:
                            cert_data = {
                                'certificate_index': i + 1,
                                'subject': str(cert.subject) if hasattr(cert, 'subject') else 'N/A',
                                'issuer': str(cert.issuer) if hasattr(cert, 'issuer') else 'N/A',
                                'serial_number': str(cert.serial_number) if hasattr(cert, 'serial_number') else 'N/A',
                                'not_valid_before': str(cert.not_valid_before) if hasattr(cert, 'not_valid_before') else 'N/A',
                                'not_valid_after': str(cert.not_valid_after) if hasattr(cert, 'not_valid_after') else 'N/A',
                                'signature_algorithm': 'N/A'
                            }
                            
                            # Try to get signature algorithm
                            try:
                                if hasattr(cert, 'signature_algorithm_oid'):
                                    cert_data['signature_algorithm'] = str(cert.signature_algorithm_oid._name)
                                elif hasattr(cert, 'signature_hash_algorithm'):
                                    cert_data['signature_algorithm'] = str(cert.signature_hash_algorithm.name)
                            except:
                                pass
                                
                            cert_info['certificates'].append(cert_data)
                        except Exception as e:
                            cert_info['certificates'].append({
                                'certificate_index': i + 1,
                                'error': f'Could not parse certificate: {str(e)}'
                            })
                except Exception as e:
                    cert_info['certificates'] = [{'error': f'Could not extract certificates: {str(e)}'}]
            
            return cert_info
            
        except Exception as e:
            print(f"Error extracting certificate info: {str(e)}")
            return {
                'is_signed': False,
                'certificates': [{'error': f'Certificate extraction failed: {str(e)}'}]
            }

    def normalize_permission_format(self, permission: str) -> str:
        cleaned = permission.strip()
        
        # Remove Permission: prefix
        if cleaned.startswith('Permission:'):
            cleaned = cleaned[11:]  # Remove 'Permission:'
        
        # Handle Manifest.permission.* -> android.permission.*
        if cleaned.startswith('Manifest.permission.'):
            return cleaned.replace('Manifest.permission.', 'android.permission.')
        
        # Handle android.Manifest.permission.* -> android.permission.*
        if cleaned.startswith('android.Manifest.permission.'):
            return cleaned.replace('android.Manifest.permission.', 'android.permission.')
        
        # Handle RoleManager.ROLE_* -> android.app.role.*
        if cleaned.startswith('RoleManager.ROLE_'):
            role_name = cleaned.replace('RoleManager.ROLE_', '')
            return f"android.app.role.{role_name}"
        
        # Handle special cases without dots (method-like permissions)
        method_permissions = {
            'getWindowToken': 'android.accessibilityservice.getWindowToken',
            'temporaryEnableAccessibilityStateUntilKeyguardRemoved': 'android.accessibilityservice.temporaryEnableAccessibilityStateUntilKeyguardRemoved',
            'READ_PRIVILEGED_PHONE_STATE': 'android.permission.READ_PRIVILEGED_PHONE_STATE'
        }
        
        if cleaned in method_permissions:
            return method_permissions[cleaned]
        
        # If already in correct format or vendor permission, return as-is
        return cleaned

    def extract_permissions(self, apk) -> Dict[str, List[str]]:
        """Extract permissions with optimized normalization"""
        try:
            all_permissions = apk.get_permissions()
            
            permissions_info = {
                'all_permissions': all_permissions,
                'sensitive_permissions': [],
                'normal_permissions': []
            }
            
            # Create normalized sensitive permissions lookup (one-time operation)
            normalized_sensitive = set()
            for perm in self.sensitive_perms:
                normalized_perm = self.normalize_permission_format(perm)
                normalized_sensitive.add(normalized_perm)
            
            # Categorize permissions
            for perm in all_permissions:
                if perm in normalized_sensitive:
                    permissions_info['sensitive_permissions'].append(perm)
                else:
                    permissions_info['normal_permissions'].append(perm)
            
            return permissions_info
            
        except Exception as e:
            print(f"Error extracting permissions: {str(e)}")
            return {
                'all_permissions': [],
                'sensitive_permissions': [],
                'normal_permissions': []
            }

    def extract_all_apis_optimized(self, analysis) -> Set[str]:
        """Optimized method to extract all APIs in one pass"""
        all_apis = set()
        
        try:
            for class_analysis in analysis.get_classes():
                class_name = class_analysis.name
                
                # Normalize class name once
                if not class_name.startswith('L'):
                    class_name = f'L{class_name}'
                if not class_name.endswith(';'):
                    class_name = f'{class_name};'
                
                for method_analysis in class_analysis.get_methods():
                    method = method_analysis.method
                    method_name = method.get_name()
                    
                    # Add both full and simple signatures
                    try:
                        descriptor = method.get_descriptor()
                        full_signature = f"{class_name}->{method_name}{descriptor}"
                        all_apis.add(full_signature)
                    except:
                        pass
                    
                    # Always add simple signature
                    simple_signature = f"{class_name}->{method_name}"
                    all_apis.add(simple_signature)
                    
                    # Process cross-references in the same loop
                    try:
                        for ref_class, ref_method, offset in method_analysis.get_xref_to():
                            try:
                                if hasattr(ref_method, 'get_class_name'):
                                    ref_class_name = ref_method.get_class_name()
                                    ref_method_name = ref_method.get_name()
                                    
                                    # Normalize reference class name
                                    if not ref_class_name.startswith('L'):
                                        ref_class_name = f'L{ref_class_name}'
                                    if not ref_class_name.endswith(';'):
                                        ref_class_name = f'{ref_class_name};'
                                    
                                    # Add reference signatures
                                    try:
                                        if hasattr(ref_method, 'get_descriptor'):
                                            ref_descriptor = ref_method.get_descriptor()
                                            ref_full_signature = f"{ref_class_name}->{ref_method_name}{ref_descriptor}"
                                            all_apis.add(ref_full_signature)
                                    except:
                                        pass
                                    
                                    ref_simple_signature = f"{ref_class_name}->{ref_method_name}"
                                    all_apis.add(ref_simple_signature)
                            except:
                                continue
                    except:
                        continue
                        
        except Exception as e:
            print(f"Error in optimized extraction: {str(e)}")
            
        return all_apis

    def find_sensitive_apis_optimized(self, all_apis: Set[str]) -> Set[str]:
        """Highly optimized sensitive API matching using pre-built lookup"""
        if not self.normalized_sensitive_apis:
            return set()
        
        found_sensitive = set()
        
        # Fast matching using pre-built lookup
        for extracted_api in all_apis:
            normalized = extracted_api.replace(' ', '')
            
            # Exact match first (fastest)
            if normalized in self.normalized_sensitive_apis:
                found_sensitive.add(normalized)
                continue
            
            # Base method matching using O(1) lookup
            if '->' in normalized:
                base = normalized.split('(')[0]
                if base in self.sensitive_api_lookup:
                    found_sensitive.update(self.sensitive_api_lookup[base])
        
        return found_sensitive

    def analyze_single_apk(self, apk_name: str) -> Dict[str, Any]:
        """Analyze a single APK file and extract all information"""
        apk_path = os.path.join(self.apk_folder, apk_name + ".apk")
        
        result = {
            'file_path': apk_name,
            'package_info': {},
            'activities': {},
            'services': {},
            'receivers': {},
            'certificate_info': {},
            'permissions': {},
            'sensitive_apis': [],
            'sensitive_apis_count': 0
        }
        
        apk, dex_files, analysis = self.analyze_apk(apk_path)
        
        if not apk or not analysis:
            result['error'] = 'Failed to analyze APK'
            return result
        
        try:
            # Extract essential meta information
            result['package_info'] = self.extract_package_info(apk)
            result['activities'] = self.extract_activities(apk)
            result['services'] = self.extract_services(apk)
            result['receivers'] = self.extract_receivers(apk)
            result['certificate_info'] = self.extract_certificate_info(apk)
            result['permissions'] = self.extract_permissions(apk)
            
            # Use optimized API extraction
            all_apis = self.extract_all_apis_optimized(analysis)
            sensitive_apis = self.find_sensitive_apis_optimized(all_apis)
            
            result['sensitive_apis'] = sorted(list(sensitive_apis))
            result['sensitive_apis_count'] = len(sensitive_apis)
            
            print(f"  -> Sensitive APIs: {len(sensitive_apis)}, Permissions: {len(result['permissions']['sensitive_permissions'])}")
            
        finally:
            # Explicitly clean up memory
            del apk, dex_files, analysis
            gc.collect()
        
        self.save_results_json(result, apk_name)
        return result

    def analyze_directory_batched(self, apk_list: List[str], batch_size: int = 5) -> Dict[str, Dict[str, Any]]:
        """Process APKs in batches to manage memory usage and avoid threading issues"""
        results = {}
        total_apks = len(apk_list)
        
        if not apk_list:
            print(f"No APK files found in {self.apk_folder}")
            return results
        
        # Calculate batch information
        total_batches = (total_apks + batch_size - 1) // batch_size
        
        print(f"Processing {total_apks} APKs in {total_batches} batches (batch size: {batch_size})")
        print("=" * 60)
        
        for i in range(0, total_apks, batch_size):
            batch = apk_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"\nBatch {batch_num}/{total_batches}: Processing {len(batch)} APKs")
            print("-" * 40)
            
            # Process each APK in the current batch
            for j, apk_name in enumerate(batch, 1):
                try:
                    print(f"  [{j}/{len(batch)}] Processing: {apk_name}")
                    result = self.analyze_single_apk(apk_name)
                    results[apk_name] = result
                    
                except Exception as e:
                    error_msg = f"Error processing {apk_name}: {str(e)}"
                    print(f"  [ERROR] {error_msg}")
                    results[apk_name] = {'error': error_msg}
            
            # Force garbage collection between batches to free memory
            print(f"  Batch {batch_num} completed. Cleaning memory...")
            gc.collect()
            
            # Progress summary
            completed = min(i + batch_size, total_apks)
            print(f"  Progress: {completed}/{total_apks} APKs completed ({completed/total_apks*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print(f"All {total_apks} APKs processed successfully!")
        
        return results

    def analyze_directory(self, apk_list: List[str]) -> Dict[str, Dict[str, Any]]:
        for apk in apk_list:
            if os.path.exists(os.path.join(self.output_folder, self.flag, f"{apk}.json")):
                apk_list.remove(apk)
                print(f"apk {apk} already processed")

        """Main analysis method - uses batched processing for reliability"""
        if len(apk_list) <= 3:
            # For very small lists, process sequentially without batching
            print(f"Processing {len(apk_list)} APKs sequentially (small dataset)")
            results = {}
            for i, apk_name in enumerate(apk_list, 1):
                print(f"\nProcessing ({i}/{len(apk_list)}): {apk_name}")
                try:
                    result = self.analyze_single_apk(apk_name)
                    results[apk_name] = result
                except Exception as e:
                    print(f"Error processing {apk_name}: {str(e)}")
                    results[apk_name] = {'error': str(e)}
            return results
        else:
            # Use batched processing for larger datasets
            return self.analyze_directory_batched(apk_list, batch_size=5)

    def save_results_json(self, results: Dict[str, Any], apk_name: str):
        """Save results in JSON format"""
        try:
            output_file = os.path.join(self.output_folder, self.flag, f"{apk_name}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"Error saving JSON results for {apk_name}: {str(e)}")


if __name__ == "__main__":
    path_dict = utils.get_path_dict()
    folder_name = flag = "malradar"

    apk_list = utils.check_file_exist(folder_name=folder_name)
    print(f"apk: {len(apk_list)} in folder_name: {folder_name}")

    sensitive_perm_file = os.path.join(path_dict["info"], "permission.txt")
    sensitive_api_file = os.path.join(path_dict["info"], "sensitive_api.txt")

    print("Initializing MetaInfoExtractor...")
    metainfo_extractor = MetaInfoExtractor(
        path_dict, 
        flag=flag, 
        sensitive_api_file=sensitive_api_file,
        sensitive_perm_file=sensitive_perm_file, 
    )
    
    print(f"Starting analysis of {len(apk_list)} APKs...")
    results = metainfo_extractor.analyze_directory(apk_list)
    
    print(f"\nAnalysis complete! Processed {len(results)} APKs.")
    print(f"Results saved to: {os.path.join(path_dict['meta_info'], flag)}")