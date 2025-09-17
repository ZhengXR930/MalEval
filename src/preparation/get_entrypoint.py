from androguard.misc import AnalyzeAPK
import networkx as nx
from androguard.core.analysis.analysis import ExternalClass, ExternalMethod

from androguard.core.apk import APK
import os
from loguru import logger
logger.remove()
from tqdm import tqdm
import utils



class EntrypointExtractor:
    def __init__(self, path_dict, folder_name):
        self.path_dict = path_dict
        self.folder_name = folder_name
        self.entrypoint_folder = os.path.join(path_dict["entrypoint"], self.folder_name)
        if not os.path.exists(self.entrypoint_folder):
            os.makedirs(self.entrypoint_folder)

    def _get_all_superclasses(self,class_analysis, dx):
        superclasses = set()
        visited = set()
        current = class_analysis

        while current:
            vm_class = current.get_vm_class()

            if isinstance(vm_class, ExternalClass):
                break

            try:
                superclass_name = vm_class.get_superclassname()
            except Exception:
                break

            if not superclass_name or superclass_name in visited:
                break
            if superclass_name == "Ljava/lang/Object;":
                break

            superclasses.add(superclass_name)
            visited.add(superclass_name)

            current = dx.get_class_analysis(superclass_name)

        return superclasses
    

    def _get_subsignature(self, method):
        return f"{method.name}{method.descriptor}"


    def _get_all_interfaces(self, class_analysis, dx):
        interfaces = set()
        to_visit = list(class_analysis.implements)

        while to_visit:
            iface_name = to_visit.pop()
            if iface_name in interfaces:
                continue
            interfaces.add(iface_name)

            iface_analysis = dx.get_class_analysis(iface_name)
            if iface_analysis:
                to_visit.extend(iface_analysis.implements)

        return interfaces

    def _is_user_class(self,class_name, pkg_name) -> bool:
        user_class_prefix = "L" + pkg_name.replace(".", "/") + "/"
        return class_name.startswith(user_class_prefix)

    def _is_override_framework_method(self, method, dx, pkg_name):
        method_subsig = self._get_subsignature(method)

        class_analysis = dx.get_class_analysis(method.class_name)
        if not class_analysis:
            return False

        super_classes = self._get_all_superclasses(class_analysis, dx)

        for super_class_name in super_classes:
            if self._is_user_class(super_class_name, pkg_name):
                continue

            super_class_analysis = dx.get_class_analysis(super_class_name)
            if not super_class_analysis:
                continue

            for m in super_class_analysis.get_methods():
                
                if self._get_subsignature(m) == method_subsig:
                    return True
        
        interfaces = self._get_all_interfaces(class_analysis, dx)
        for interface_name in interfaces:
            if self._is_user_class(interface_name, pkg_name):
                continue

            interface_analysis = dx.get_class_analysis(interface_name)
            if not interface_analysis:
                continue

            for m in interface_analysis.get_methods():
                if self._get_subsignature(m) == method_subsig:
                    return True

        return False

    def _get_manifest_entrypoints(self, apk_path, dx):
        entrypoints = []

        a = APK(apk_path)

        receivers = a.get_receivers()
        services = a.get_services()
        providers = a.get_providers()


        receiver_methods = {"onReceive"}
        service_methods = {"onCreate", "onStartCommand", "onBind"}
        provider_methods = {"onCreate"}

        def fqcn_to_descriptor(name):
            return "L" + name.replace(".", "/") + ";"

        for cls in receivers:
            descriptor = fqcn_to_descriptor(cls)
            analysis = dx.get_class_analysis(descriptor)
            if analysis:
                for m in analysis.get_methods():
                    if m.name in receiver_methods:
                        entrypoints.append((descriptor, m.name))

        for cls in services:
            descriptor = fqcn_to_descriptor(cls)
            analysis = dx.get_class_analysis(descriptor)
            if analysis:
                for m in analysis.get_methods():
                    if m.name in service_methods:
                        entrypoints.append((descriptor, m.name))

        for cls in providers:
            descriptor = fqcn_to_descriptor(cls)
            analysis = dx.get_class_analysis(descriptor)
            if analysis:
                for m in analysis.get_methods():
                    if m.name in provider_methods:
                        entrypoints.append((descriptor, m.name))

        return entrypoints

    def _is_user_defined_class(self, class_name, dx) -> bool:
        cls = dx.get_class_analysis(class_name)
        if cls is None:
            return False 
        if cls.is_external():
            return False
        if cls.is_android_api():
            return False
        unwanted_prefixes = (
            "Landroid/support/",
            "Landroidx/",
            "Landroid/",
            "Lcom/google/",
            "Lorg/apache/",
            "Ldalvik/",
            "Ljava/",
            "Ljavax/",
            "Lkotlin/",
        )
        if class_name.startswith(unwanted_prefixes):
            return False
        return True

    def get_entrypoints(self, apk_path):
        try:
            a, d, dx = AnalyzeAPK(apk_path)
        except Exception as e:
            if "0x3e" in str(e):
                print(f"[!] Skipping APK {apk_path} due to illegal opcode 0x3e")
                return []
            else:
                print(f"[!] Failed to analyze {apk_path}: {e}")
                return []
        cg: nx.DiGraph = dx.get_call_graph()
        pkg_name = a.get_package()

        entry_candidates = [n for n in cg.nodes()]

        entrypoints = []

        for n in entry_candidates:
            method = dx.get_method(n)
            if method is None:
                continue
                
            class_name = method.class_name  
            method_name = method.name

            if method.name == "<init>":
                continue

            if not self._is_user_defined_class(class_name, dx):
                continue

            if isinstance(method, ExternalMethod):
                continue

            if self._is_override_framework_method(method, dx, pkg_name):
                entrypoints.append((class_name, method_name))

        manifest_eps = self._get_manifest_entrypoints(apk_path, dx)
        entrypoints.extend(manifest_eps)
        entrypoints = list(set(entrypoints))
        
        return entrypoints
    
    def get_all_entrypoints(self):
        if self.folder_name == "malradar":
            apk_folder = path_dict["archived_apk"]
        elif self.folder_name == "new":
            apk_folder = path_dict["latest_apk"]
        elif self.folder_name == "benign":
            apk_folder = path_dict["benign_apk"]
        else:
            print("wrong folder name")

        entrypoint_length = []
        processed_count = 0
        for apk in tqdm(os.listdir(apk_folder)):
            apk_name = os.path.basename(apk).replace(".apk", "")
            if os.path.exists(os.path.join(self.entrypoint_folder, f"{apk_name}.txt")):
                continue
            apk_path = os.path.join(apk_folder, apk)
            entrypoints = self.get_entrypoints(apk_path)
            with open(os.path.join(self.entrypoint_folder, f"{apk_name}.txt"), "w") as f:
                for cls, mtd in entrypoints:
                    f.write(f"{cls} -> {mtd}\n")
            entrypoint_length.append(len(entrypoints))
            processed_count += 1

        print(f"average number of entrypoints: {sum(entrypoint_length) / len(entrypoint_length)}")
        print(f"max number of entrypoints: {max(entrypoint_length)}")
        print(f"min number of entrypoints: {min(entrypoint_length)}")
        print(f"processed count: {processed_count} in number of apks: {len(os.listdir(apk_folder))}")

    def _get_rank_entrypoints(self):
        entrypoint_folder = os.path.join(self.path_dict["entrypoint"], self.folder_name)
        
        if not os.path.exists(entrypoint_folder):
            print(f"Error: Entrypoint folder not found: {entrypoint_folder}")
            return {}
        
        entrypoint_counts = {}
        
        for filename in os.listdir(entrypoint_folder):
            filepath = os.path.join(entrypoint_folder, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
                        line_count = sum(1 for line in f if line.strip())
                    
                    # Extract APK name from filename (remove extension)
                    apk_name = os.path.splitext(filename)[0]
                    entrypoint_counts[apk_name] = line_count
                    
                except:
                    apk_name = os.path.splitext(filename)[0]
                    entrypoint_counts[apk_name] = 0
        
        entrypoint_counts = dict(sorted(entrypoint_counts.items(), key=lambda x: x[1], reverse=True))
        
        print(f"Processed {len(entrypoint_counts)} APK entrypoint files")
        if entrypoint_counts:
            for entrypoint in entrypoint_counts:
                print(f"entrypoint: {entrypoint}, count: {entrypoint_counts[entrypoint]}")
        
        return entrypoint_counts



if __name__ == "__main__":
    path_dict = utils.get_path_dict()
    folder_name = "benign"
    entrypoint_extractor = EntrypointExtractor(path_dict, folder_name)
    entrypoint_extractor.get_all_entrypoints()
    