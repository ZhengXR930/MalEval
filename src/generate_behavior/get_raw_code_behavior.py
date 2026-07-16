import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_behavior.ablation_behavior_utils import (
    get_llm_client,
    compute_sas_score,
    count_tokens,
    estimate_total_tokens,
    get_external_paths,
    get_indexed_apk_list,
    get_model_budget,
    load_gt_info,
    rank_apks_by_call_chain_size,
    write_statistics,
)
from prompts.ablation_behavior_prompt import RAW_CODE_BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING


class RawCodeBehaviorExtractor:
    def __init__(
        self,
        *,
        model_name: str = "deepseek",
        folder_name: str = "new",
        max_workers: int = 4,
        force: bool = False,
    ) -> None:
        self.model_name = model_name
        self.folder_name = folder_name
        self.max_workers = max_workers
        self.force = force

        self.paths = get_external_paths()
        self.gt_info = load_gt_info(folder_name)
        self.output_root = self.paths["raw_code_behavior"] / model_name / folder_name
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_function_name(name: str) -> str:
        return " ".join(str(name).split())

    @staticmethod
    def _extract_function_names_from_block(block: str) -> Set[str]:
        return {
            RawCodeBehaviorExtractor._normalize_function_name(match)
            for match in re.findall(r"^Function:\s+(.+)$", block, flags=re.MULTILINE)
        }

    @staticmethod
    def _clean_source(source: Optional[str]) -> str:
        if not source:
            return ""
        cleaned = str(source).strip()
        if not cleaned:
            return ""
        if "Failed to decompile" in cleaned:
            return ""
        return cleaned

    def _load_call_chain_map(self, apk_name: str) -> Dict[str, Dict[str, Any]]:
        call_chain_path = self.paths["call_chain"] / self.folder_name / f"{apk_name}.jsonl"
        if not call_chain_path.exists():
            raise FileNotFoundError(f"Call-chain file not found: {call_chain_path}")

        call_chain_map: Dict[str, Dict[str, Any]] = {}
        with call_chain_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                func_name = self._normalize_function_name(item.get("func", ""))
                if func_name:
                    call_chain_map[func_name] = item
        return call_chain_map

    def _load_ranked_context_summaries(self, apk_name: str) -> List[Dict[str, Any]]:
        summary_path = self.paths["context_summary"] / self.model_name / self.folder_name / f"{apk_name}.jsonl"
        if not summary_path.exists():
            raise FileNotFoundError(f"Context summary file not found: {summary_path}")

        items: List[Dict[str, Any]] = []
        with summary_path.open("r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        items.sort(key=lambda x: x.get("sensitivity_score", 0), reverse=True)
        return items

    def _append_function_block(
        self,
        *,
        lines: List[str],
        seen_functions: Set[str],
        local_seen: Set[str],
        function_name: str,
        source: str,
        role: str,
        allow_seen: bool = False,
    ) -> bool:
        norm_name = self._normalize_function_name(function_name)
        cleaned_source = self._clean_source(source)
        if not norm_name or not cleaned_source:
            return False
        if norm_name in local_seen:
            return False
        if not allow_seen and norm_name in seen_functions:
            return False

        local_seen.add(norm_name)
        lines.append(f"{role}")
        lines.append(f"Function: {norm_name}")
        lines.append("Code:")
        lines.append(cleaned_source)
        return True

    def _format_primary_function_block(
        self,
        *,
        rank: int,
        summary_item: Dict[str, Any],
        call_chain_item: Dict[str, Any],
        seen_functions: Set[str],
    ) -> Tuple[str, Set[str]]:
        primary_name = self._normalize_function_name(summary_item["function"])
        local_seen: Set[str] = set()
        lines = [
            f"### Ranked Sensitive Function {rank}",
            f"Primary Function: {primary_name}",
            f"Sensitivity Score: {summary_item.get('sensitivity_score', 0)}",
            "Below is raw decompiled code for the primary function and its one-hop caller/callee context.",
            "",
        ]

        self._append_function_block(
            lines=lines,
            seen_functions=seen_functions,
            local_seen=local_seen,
            function_name=primary_name,
            source=call_chain_item.get("source", ""),
            role="Primary Decompiled Function",
            allow_seen=True,
        )

        callers = call_chain_item.get("callers", []) or []
        if callers:
            lines.append("")
            lines.append("One-hop Callers:")
            for caller in callers:
                self._append_function_block(
                    lines=lines,
                    seen_functions=seen_functions,
                    local_seen=local_seen,
                    function_name=caller.get("name", ""),
                    source=caller.get("source", ""),
                    role="Caller Function",
                )

        callees = call_chain_item.get("callees", []) or []
        if callees:
            lines.append("")
            lines.append("One-hop Callees:")
            for callee in callees:
                self._append_function_block(
                    lines=lines,
                    seen_functions=seen_functions,
                    local_seen=local_seen,
                    function_name=callee.get("name", ""),
                    source=callee.get("source", ""),
                    role="Callee Function",
                )

        block = "\n".join(lines).strip()
        return block, local_seen

    def _truncate_block_to_fit(
        self,
        *,
        base_query: str,
        block: str,
        available_user_tokens: int,
    ) -> Optional[str]:
        lines = block.splitlines()
        if not lines:
            return None

        truncation_note = "[BLOCK TRUNCATED TO FIT TOKEN BUDGET]"
        low, high = 1, len(lines)
        best_block: Optional[str] = None

        while low <= high:
            mid = (low + high) // 2
            candidate_lines = lines[:mid]
            candidate_block = "\n".join(candidate_lines).strip()
            if mid < len(lines):
                candidate_block = f"{candidate_block}\n{truncation_note}"

            candidate_query = f"{base_query}\n\n{candidate_block}".strip()
            if count_tokens(candidate_query, self.model_name) <= available_user_tokens:
                best_block = candidate_block
                low = mid + 1
            else:
                high = mid - 1

        return best_block

    def _build_minimal_primary_block_to_fit(
        self,
        *,
        rank: int,
        summary_item: Dict[str, Any],
        call_chain_item: Dict[str, Any],
        base_query: str,
        available_user_tokens: int,
    ) -> Optional[str]:
        primary_name = self._normalize_function_name(summary_item.get("function", ""))
        primary_source = self._clean_source(call_chain_item.get("source", ""))
        if not primary_name:
            return None

        static_prefix = "\n".join(
            [
                f"### Ranked Sensitive Function {rank}",
                f"Primary Function: {primary_name}",
                f"Sensitivity Score: {summary_item.get('sensitivity_score', 0)}",
                "Below is raw decompiled code for the primary function. The primary block is truncated to fit the token budget.",
                "",
                "Primary Decompiled Function",
                f"Function: {primary_name}",
                "Code:",
            ]
        ).strip()
        truncation_note = "[PRIMARY FUNCTION TRUNCATED TO FIT TOKEN BUDGET]"

        # If the full primary source fits, keep it.
        full_block = f"{static_prefix}\n{primary_source}"
        full_query = f"{base_query}\n\n{full_block}".strip()
        if count_tokens(full_query, self.model_name) <= available_user_tokens:
            return full_block

        if not primary_source:
            candidate_block = f"{static_prefix}\n{truncation_note}"
            candidate_query = f"{base_query}\n\n{candidate_block}".strip()
            if count_tokens(candidate_query, self.model_name) <= available_user_tokens:
                return candidate_block
            return None

        low, high = 0, len(primary_source)
        best_block: Optional[str] = None
        while low <= high:
            mid = (low + high) // 2
            truncated_source = primary_source[:mid].rstrip()
            candidate_block = f"{static_prefix}\n{truncated_source}\n{truncation_note}".strip()
            candidate_query = f"{base_query}\n\n{candidate_block}".strip()
            if count_tokens(candidate_query, self.model_name) <= available_user_tokens:
                best_block = candidate_block
                low = mid + 1
            else:
                high = mid - 1

        return best_block

    def _build_query(self, apk_name: str) -> Tuple[str, Dict[str, Any]]:
        ranked_summaries = self._load_ranked_context_summaries(apk_name)
        call_chain_map = self._load_call_chain_map(apk_name)

        token_limit, reserved_for_completion, safety_margin = get_model_budget(self.model_name)
        system_tokens = count_tokens(RAW_CODE_BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING, self.model_name)
        available_user_tokens = token_limit - reserved_for_completion - safety_margin - system_tokens
        if available_user_tokens <= 0:
            raise ValueError("System prompt exceeds model token budget.")

        header = """
Please analyze the following raw decompiled Android code evidence and generate a final malware report in the specified JSON format.

Important:
- Primary functions are ordered by sensitivity score from high to low.
- For each primary function, I provide its own decompiled body and one-hop caller/callee code when available.
- Ground the report strictly in this raw code evidence.
""".strip()

        query = header
        seen_functions: Set[str] = set()
        included_primary = 0
        missing_from_call_chain = 0
        truncated = False

        for rank, summary_item in enumerate(ranked_summaries, start=1):
            func_name = self._normalize_function_name(summary_item.get("function", ""))
            call_chain_item = call_chain_map.get(func_name)
            if call_chain_item is None:
                missing_from_call_chain += 1
                continue

            block, local_seen = self._format_primary_function_block(
                rank=rank,
                summary_item=summary_item,
                call_chain_item=call_chain_item,
                seen_functions=seen_functions,
            )
            if not local_seen:
                continue

            final_block = block
            candidate_query = f"{query}\n\n{final_block}".strip()
            if count_tokens(candidate_query, self.model_name) > available_user_tokens:
                truncated = True
                if included_primary == 0:
                    truncated_block = self._build_minimal_primary_block_to_fit(
                        rank=rank,
                        summary_item=summary_item,
                        call_chain_item=call_chain_item,
                        base_query=query,
                        available_user_tokens=available_user_tokens,
                    )
                else:
                    truncated_block = self._truncate_block_to_fit(
                        base_query=query,
                        block=block,
                        available_user_tokens=available_user_tokens,
                    )
                if truncated_block is None:
                    break
                final_block = truncated_block
                candidate_query = f"{query}\n\n{final_block}".strip()
                query = candidate_query
                included_primary += 1
                seen_functions.update(self._extract_function_names_from_block(final_block))
                break

            query = candidate_query
            included_primary += 1
            seen_functions.update(self._extract_function_names_from_block(final_block))

        input_stats = {
            "available_primary_items": len(ranked_summaries),
            "included_primary_items": included_primary,
            "included_total_functions": len(seen_functions),
            "missing_from_call_chain": missing_from_call_chain,
            "truncated_by_budget": truncated,
        }
        return query, input_stats

    def get_behavior_single_apk(self, apk_name: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
        output_path = self.output_root / f"{apk_name}.json"
        if output_path.exists() and not self.force:
            return None
        if verbose:
            print(f"[single-apk] {apk_name}: 开始构造 query")
        query, input_stats = self._build_query(apk_name)
        if verbose:
            query_tokens = count_tokens(query, self.model_name)
            total_prompt_tokens = count_tokens(
                RAW_CODE_BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING,
                self.model_name,
            ) + query_tokens
            print(
                f"[single-apk] {apk_name}: query token 数 user={query_tokens}, total_prompt={total_prompt_tokens}"
            )

        generation_start = time.perf_counter()
        llm_client = get_llm_client(model=self.model_name)
        if verbose:
            print(f"[single-apk] {apk_name}: 开始调用 LLM")
        result, final_user_message = llm_client(query, RAW_CODE_BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING)
        if result is None or not isinstance(result, dict):
            print(f"[Warning] Invalid result for {apk_name}: {type(result)}")
            return None

        if self.folder_name == "benign":
            gt_type, gt_family = "benign", "benign"
        else:
            gt_meta = self.gt_info.get(apk_name, {})
            gt_type = gt_meta.get("type", "")
            gt_family = gt_meta.get("family", "")

        token_stats = estimate_total_tokens(
            model_name=self.model_name,
            system_prompt=RAW_CODE_BEHAVIOR_SYSTEM_PROMPT_WITHOUT_THINKING,
            final_user_message=final_user_message,
            result=result,
        )
        result["gt_type"] = gt_type
        result["gt_family"] = gt_family
        result["sas_score"] = compute_sas_score(result, final_user_message)
        result["input_stats"] = {
            **input_stats,
            **token_stats,
            "generation_seconds": time.perf_counter() - generation_start,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    def get_behavior_all(self, apk_list: List[str]) -> None:
        to_process: List[str] = []
        for apk_name in apk_list:
            if self.folder_name != "benign" and apk_name not in self.gt_info:
                continue
            output_path = self.output_root / f"{apk_name}.json"
            if output_path.exists() and not self.force:
                continue
            to_process.append(apk_name)

        print(f"Processing {len(to_process)} APKs for raw-code behavior generation")

        processed_stats: List[Dict[str, Any]] = []
        failed_apks: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_apk = {
                executor.submit(self.get_behavior_single_apk, apk_name): apk_name
                for apk_name in to_process
            }
            progress_bar = tqdm(total=len(future_to_apk), desc="raw_code_behavior", unit="apk")
            for idx, future in enumerate(as_completed(future_to_apk), start=1):
                apk_name = future_to_apk[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[Error] {apk_name}: {exc}")
                    failed_apks.append(apk_name)
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f"done={idx} failed={len(failed_apks)} last={apk_name[:8]}")
                    continue

                if result is None:
                    failed_apks.append(apk_name)
                else:
                    processed_stats.append(result.get("input_stats", {}))
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"done={idx} failed={len(failed_apks)} last={apk_name[:8]}")
            progress_bar.close()

        log_path = self.paths["info"] / "statistic" / f"{self.model_name}_{self.folder_name}_raw_code_behavior.txt"
        write_statistics(
            log_path=log_path,
            header_lines=[
                f"model: {self.model_name}",
                f"folder: {self.folder_name}",
                "source: raw_code",
                f"output_dir: {self.output_root}",
            ],
            processed_stats=processed_stats,
            failed_apks=failed_apks,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate malware behavior reports from raw decompiled code.")
    parser.add_argument("--model", default="deepseek", help="LLM backend name, defaults to deepseek")
    parser.add_argument("--folder", default="new", choices=["new", "malradar", "benign"])
    parser.add_argument("--apk", help="Process a single APK sha256")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-generate reports even if output files already exist")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extractor = RawCodeBehaviorExtractor(
        model_name=args.model,
        folder_name=args.folder,
        max_workers=args.max_workers,
        force=args.force,
    )

    if args.apk:
        extractor.get_behavior_single_apk(args.apk, verbose=True)
    else:
        apk_list = rank_apks_by_call_chain_size(
            extractor.paths["call_chain"] / args.folder,
            indexed_apks=get_indexed_apk_list(args.folder),
        )
        extractor.get_behavior_all(apk_list)
