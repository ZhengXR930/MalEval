from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils


def iter_sample_json_files(dir_path: Path) -> Iterable[Path]:
    """
    Yield per-sample json files under `dir_path`.
    Skips aggregate files like `report_score_results.json` if present.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return
    for p in sorted(dir_path.glob("*.json")):
        if p.name == "report_score_results.json":
            continue
        yield p


def load_scores_from_samples(dir_path: Path) -> Tuple[float, int]:
    """
    Load per-sample json files in a directory and return (sum_scores, count).
    """
    s = 0.0
    n = 0
    for sample_path in iter_sample_json_files(dir_path):
        try:
            with sample_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        val = None
        if isinstance(data, dict):
            val = data.get("report_quality_score", None)
            # Backward/alternate naming safety (if any)
            if val is None:
                val = data.get("overall_quality_score", None)

        if isinstance(val, (int, float)):
            s += float(val)
            n += 1
    return s, n


def main() -> None:
    path_dict = utils.get_path_dict()
    
    base_dir = Path(path_dict["result"]) / "report_quality" / "meta_behavior"

    per_model: Dict[str, Dict[str, float]] = {}
    total_sum = 0.0
    total_count = 0

    for model_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        model = model_dir.name

        malradar_dir = model_dir / "malradar"
        new_dir = model_dir / "new"

        if not malradar_dir.exists() or not new_dir.exists():
            # Skip models without both result files
            continue

        m_sum, m_count = load_scores_from_samples(malradar_dir)
        n_sum, n_count = load_scores_from_samples(new_dir)

        model_sum = m_sum + n_sum
        model_count = m_count + n_count
        if model_count == 0:
            continue

        model_avg = model_sum / model_count
        per_model[model] = {
            "sum": model_sum,
            "count": model_count,
            "avg_report_quality": round(model_avg, 4),
        }

        total_sum += model_sum
        total_count += model_count

    # Print per-model results
    for model, stats in sorted(per_model.items()):
        print(
            f"{model}: avg_report_quality={stats['avg_report_quality']:.4f} "
            f"(n={stats['count']})"
        )

    if total_count > 0:
        overall_avg = total_sum / total_count
        print(f"OVERALL: avg_report_quality={overall_avg:.4f} (n={total_count})")
    else:
        print("No report_quality_score entries found.")


if __name__ == "__main__":
    main()

