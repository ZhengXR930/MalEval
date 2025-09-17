import os
import json
from androguard.misc import AnalyzeAPK
from androguard.decompiler import decompile
from collections import deque
from androguard.core.analysis.analysis import MethodAnalysis
from tqdm import tqdm
from loguru import logger
logger.remove()
import time
import sys
sys.stdout.flush()
from collections import defaultdict
import networkx as nx
import utils
from get_entrypoint import EntrypointExtractor


class ReachableFuncExtractor:
    def __init__(self, path_dict, folder_name):
        self.path_dict = path_dict
        self.folder_name = folder_name
        self.decompile_folder = path_dict["call_chain"]
        self.entrypoint_folder = path_dict["entrypoint"]
        self.all_func_folder = path_dict["reachable_func"]
        if not os.path.exists(self.decompile_folder):
            os.makedirs(self.decompile_folder)
        if not os.path.exists(self.entrypoint_folder):
            os.makedirs(self.entrypoint_folder)
        self.entrypoint_extractor = EntrypointExtractor(path_dict, folder_name)

        if self.folder_name == "archived":
            self.apk_folder = path_dict["archived_apk"]
        elif self.folder_name == "latest":
            self.apk_folder = path_dict["latest_apk"]
        elif self.folder_name == "benign":
            self.apk_folder = path_dict["benign_apk"]
        else:
            print("wrong folder name")

    def _extract_method_obj(self, x):
        if isinstance(x, MethodAnalysis):
            return x
        if isinstance(x, tuple) and isinstance(x[1], MethodAnalysis):
            return x[1]
        return None

    def _load_entrypoints(self, entrypoint_file):
        entrypoints = []
        with open(entrypoint_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "->" not in line:
                    continue
                class_part, method_part = line.split(" -> ")
                class_name = class_part.strip()  # e.g., Lcom/xxx/MyActivity;
                method_name = method_part.strip()  # e.g., onCreate
                entrypoints.append((class_name, method_name))
        return entrypoints


    def _safe_decompile(self, method, method_source_cache):
        if method.full_name in method_source_cache:
            return method_source_cache[method.full_name]
        try:
            dv = decompile.DvMethod(method)
            dv.process()
            src = dv.get_source()
        except Exception as e:
            src = f"/* Failed to decompile: {e} */"
        method_source_cache[method.full_name] = src
        return src

    def _index_dx_methods(self, dx):
        class_method_map = defaultdict(list)
        for m in dx.get_methods():
            class_method_map[(m.class_name, m.name)].append(m)
        return class_method_map

    def processing(self, apk_name, entrypoints):
        apk_path = os.path.join(self.apk_folder, f"{apk_name}.apk")
        os.makedirs(os.path.join(self.decompile_folder, self.folder_name), exist_ok=True)
        reachable_func_path = os.path.join(self.decompile_folder, self.folder_name, f"{apk_name}.jsonl")
        if os.path.exists(reachable_func_path):
            print(f"reachable func already exists: {reachable_func_path}")
            return {}
        
        try:
            a, d, dx = AnalyzeAPK(apk_path) 
        except Exception as e:
            print(f"Error processing {apk_path}: {e}")
            return {}

        method_source_cache = {}
        global_func_seen = set()
        entrypoint_func_records = []
        class_method_map = self._index_dx_methods(dx)
        for i, (class_name, method_name) in tqdm(enumerate(entrypoints), total=len(entrypoints)):
            matched_methods = class_method_map[(class_name, method_name)]

            if not matched_methods:
                print(f"[Warning] Entrypoint not found: {class_name} -> {method_name}")
                continue

            for method in matched_methods:
                ep_name = f"{class_name}->{method_name}".replace("/", "_").replace(";", "")
                
                all_funcs = {}
                visited = set()
                queue = deque([method])
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)

                    if current.full_name in global_func_seen:
                        continue
                    global_func_seen.add(current.full_name)

                    # Attempt to get source code
                    try:
                        src = self._safe_decompile(current, method_source_cache)
                    except Exception as e:
                        src = f"/* Failed to decompile: {e} */"
                    
                    method_obj = self._extract_method_obj(current)
                    caller_set = {}
                    for ref in method_obj.get_xref_from():
                        try:
                            caller = ref[1]
                            if caller.full_name not in caller_set:
                                try:
                                    caller_src = self._safe_decompile(caller, method_source_cache)
                                except Exception as e:
                                    caller_src = f"/* Failed to decompile caller: {e} */"
                                caller_set[caller.full_name] = caller_src
                        except Exception:
                            pass

                    callee_set = {}
                    for ref in method_obj.get_xref_to():
                        try:
                            callee = ref[1]
                            if callee not in visited:
                                queue.append(callee)
                            if callee.full_name not in callee_set:
                                try:
                                    callee_src = self._safe_decompile(callee, method_source_cache)
                                except Exception as e:
                                    callee_src = f"/* Failed to decompile callee: {e} */"
                                callee_set[callee.full_name] = callee_src
                        except Exception:
                            pass
                    
                    if "/* Failed to decompile" in src:
                        continue
                    
                    all_funcs[method_obj.full_name] = {
                        "func": method_obj.full_name,
                        "source": src,
                        "callers": [{"name": n, "source": s} for n, s in caller_set.items()],
                        "callees": [{"name": n, "source": s} for n, s in callee_set.items()]
                    }

                for func_data in all_funcs.values():
                    func_data["entrypoint"] = ep_name
                    entrypoint_func_records.append(func_data)

        out_path = os.path.join(self.decompile_folder, self.folder_name, f"{apk_name}.jsonl")
        with open(out_path, "w") as f:
            for func_data in entrypoint_func_records:
                f.write(json.dumps(func_data, ensure_ascii=False) + "\n")

        return entrypoint_func_records

    def process_single_apk(self, apk_name):
        try:
            entrypoint_path = os.path.join(self.entrypoint_folder, self.folder_name, f"{apk_name}.txt")
            if not os.path.exists(entrypoint_path):
                entrypoints = self.entrypoint_extractor.get_entrypoints(os.path.join(self.apk_folder, f"{apk_name}.apk"))
                with open(entrypoint_path, "w") as f:
                    for cls, mtd in entrypoints:
                        f.write(f"{cls} -> {mtd}\n")
            else:
                with open(entrypoint_path, "r") as f:
                    entrypoints = [line.strip().split(" -> ") for line in f]
            entrypoints = self._load_entrypoints(entrypoint_path)
            start_time = time.time()
            all_funcs = self.processing(apk_name, entrypoints)
            print(f"number of reachable functions: {len(all_funcs)}")
            end_time = time.time()
            print(f"time taken: {end_time - start_time} seconds")
            
            return True
        except Exception as e:
            print(f"Error processing {apk_name}: {e}")
            return False
           
        
    def get_reachable_func(self):
        success_apks = []
        failed_apks = []
        apk_list = [x.replace(".apk", "") for x in os.listdir(self.apk_folder)]
        for apk_name in tqdm(apk_list):
            flag = self.process_single_apk(apk_name)
            if not flag:
                print(f"Error processing {apk_name}")
                failed_apks.append(apk_name)
            else:
                success_apks.append(apk_name)

        print(f"number of success apks: {len(success_apks)}")
        print(f"number of failed apks: {len(failed_apks)}")

        return success_apks, failed_apks

    def _select_fallback(self, threshold=15):
        selected_apk_list = []
        apk_list = [os.path.basename(apk_path).replace(".apk", "") for apk_path in self.apk_folder]
        apk_list = set(apk_list)
        call_chain_file_list = [os.path.join(self.decompile_folder, self.folder_name, f"{x}.jsonl") for x in apk_list]
        function_count = {}
        for call_chain_file in tqdm(call_chain_file_list):
            base_name = os.path.basename(call_chain_file).replace(".jsonl", "")
            func_samples = []
            with open(call_chain_file, "r") as f:
                for line in f:
                    func_data = json.loads(line)
                    func_samples.append(func_data)
            function_count[base_name] = len(func_samples)
        function_count = sorted(function_count.items(), key=lambda x: x[1], reverse=True)
        select_func = [x[0] for x in function_count if x[1] < threshold]
        print(f"number of selected fallback func: {len(select_func)} in apk_list: {len(apk_list)}")
        for func in select_func:
            selected_apk_list.append(func.split('/')[-1].replace(".jsonl", ""))
        return selected_apk_list

    def get_all_func_for_fallback(self):
        """
        Get all methods (no entrypoints case).
        For each method, output itself, its 1-hop callers and callees.
        """
        fallback_apk_list = self._select_fallback()
        apk_paths = [
            f"{self.apk_folder}/{x}.apk"
            for x in fallback_apk_list
            if not os.path.exists(os.path.join(self.decompile_folder, self.folder_name, f"{x}.jsonl"))
        ]

        for apk_path in tqdm(apk_paths):
            try:
                a, d, dx = AnalyzeAPK(apk_path)
            except Exception as e:
                print(f"[!] Failed to analyze {apk_path}: {e}")
                continue

            cg: nx.DiGraph = dx.get_call_graph()
            nodes = list(cg.nodes())
            print(f"processing {apk_path}, number of methods: {len(nodes)}")

            method_source_cache = {}
            func_records = []

            for n in nodes:
                method = dx.get_method(n)
                if method is None:
                    continue

                # try decompile
                try:
                    src = self._safe_decompile(method, method_source_cache)
                except Exception as e:
                    src = f"/* Failed to decompile: {e} */"

                if "/* Failed to decompile" in src:
                    continue

                method_obj = self._extract_method_obj(method)

                # collect callers (1-hop)
                caller_set = {}
                for ref in method_obj.get_xref_from():
                    try:
                        caller = ref[1]
                        if caller.full_name not in caller_set:
                            try:
                                caller_src = self._safe_decompile(caller, method_source_cache)
                            except Exception as e:
                                caller_src = f"/* Failed to decompile caller: {e} */"
                            caller_set[caller.full_name] = caller_src
                    except Exception:
                        pass

                # collect callees (1-hop)
                callee_set = {}
                for ref in method_obj.get_xref_to():
                    try:
                        callee = ref[1]
                        if callee.full_name not in callee_set:
                            try:
                                callee_src = self._safe_decompile(callee, method_source_cache)
                            except Exception as e:
                                callee_src = f"/* Failed to decompile callee: {e} */"
                            callee_set[callee.full_name] = callee_src
                    except Exception:
                        pass

                func_records.append({
                    "func": method_obj.full_name,
                    "source": src,
                    "callers": [{"name": n, "source": s} for n, s in caller_set.items()],
                    "callees": [{"name": n, "source": s} for n, s in callee_set.items()],
                    "entrypoint": None 
                })

            # save result
            apk_name = os.path.basename(apk_path).replace(".apk", "")
            out_path = os.path.join(self.decompile_folder, self.folder_name, f"{apk_name}.jsonl")
            with open(out_path, "w") as f:
                for func_data in func_records:
                    f.write(json.dumps(func_data, ensure_ascii=False) + "\n")

            print(f"[+] Saved {len(func_records)} funcs to {out_path}")

    def statistic(self, apk_list):
        reachable_func_folder = os.path.join(self.decompile_folder, self.folder_name)
        all_func_folder = os.path.join(self.all_func_folder, self.folder_name)
        reachable_func_count_all = 0
        all_func_count_all = 0
        for apk_name in tqdm(apk_list):
            reachable_func_path = os.path.join(reachable_func_folder, f"{apk_name}.jsonl")
            all_func_path = os.path.join(all_func_folder, f"{apk_name}.jsonl")
            with open(reachable_func_path, "r") as f:
                reachable_func_count = len(f.readlines())
            with open(all_func_path, "r") as f:
                all_func_count = len(f.readlines())
            reachable_func_count_all += reachable_func_count
            all_func_count_all += all_func_count
        print(f"total reachable_func_count: {reachable_func_count_all} / {len(apk_list)} = {reachable_func_count_all / len(apk_list)}")
        print(f"total all_func_count: {all_func_count_all} / {len(apk_list)} = {all_func_count_all / len(apk_list)}")
        print(f"redution rate: {reachable_func_count_all / all_func_count_all}")

    
if __name__ == "__main__":
    path_dict = utils.get_path_dict()
    folder_name = "benign"

    extractor = ReachableFuncExtractor(path_dict, folder_name)
    extractor.get_reachable_func()

    