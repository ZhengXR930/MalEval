# MalEval

**Artifact version:** 1.0.0 (ISSTA 2026 camera-ready snapshot)

**Article:** *Is “Knowing It’s Malicious” Enough? Evaluating LLMs for
Fine-Grained Malware Behavior Auditing*

**Article DOI:** [10.1145/3832187](https://doi.org/10.1145/3832187)

MalEval is a framework for evaluating Android malware behavior reports generated
by large language models. The code in this repository implements two execution
paths:

1. **From APK**: run static analysis on APK files to produce entry points, call
   chains, function bodies, and reachable functions.
2. **From artifact**: start from the precomputed static-analysis outputs and
   generate function summaries, context-intermediate representations, behavior
   reports, and evaluation metrics.

The released dataset is hosted separately in the
[MalEval Hugging Face dataset repository](https://huggingface.co/datasets/Xinzxr/MalEval).
For the camera-ready artifact, use the immutable dataset revision
[`a5bd8d81116d2936a3edcd7eae5b26403aabdbd4`](https://huggingface.co/datasets/Xinzxr/MalEval/tree/a5bd8d81116d2936a3edcd7eae5b26403aabdbd4).
The dataset consists of downloadable archives rather than a
Hugging Face `datasets` table, so the Dataset Viewer is not required.

The benchmark contains **255 Android applications**: 200 archived malware
samples, 30 recent malware samples, and 25 benign applications used as
simulated false positives. See [DATA.md](DATA.md) for the released files,
provenance, rights, and safe-handling notes.

## Repository Layout

```text
info/                Sample metadata for the benign, MalRadar, and new malware splits.
model_registry.yaml Model/provider configuration for LLM-based stages.
DATA.md              Dataset composition, provenance, and access instructions.
LICENSE.txt          License scope and third-party-material exclusions.
CITATION.cff         Machine-readable citation metadata.
src/                 Static analysis, summarization, behavior generation, and metrics code.
```

The dataset should be extracted so that the repository has this layout:

```text
MalEval/
|-- info/
|-- src/
`-- artifacts/
    |-- apk/
    |-- call_chain/
    |-- entrypoints/
    |-- functions/
    |-- meta_info/
    |-- reachable_func/
    `-- reports/
```

## Environment

Use Python 3.10 or newer. We recommend a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Configure model providers in `model_registry.yaml` before running LLM-based
stages. Each model entry can store either literal values, such as `api_key` and
`base_url`, or environment-variable references, such as `api_key_env` and
`base_url_env`.

Example:

```yaml
models:
  gpt:
    provider: openai
    model: gpt-5
    api_key: <your_openai_key>
    base_url: https://api.openai.com/v1
```

The default registry includes entries for `gpt`, `gpt-5`, `qwen`, `deepseek`,
`llama`, `coder`, `claude`, and `gemini`. If you prefer environment variables,
set the variables referenced by `model_registry.yaml`, for example
`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `QWEN3_API_KEY`, `HUGGINGFACE_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GEMINI_PROJECT`, and
`GEMINI_LOCATION`.

## From APK

This path starts from APK files under `artifacts/apk/<split>/` and regenerates
the static-analysis artifacts. The split must be one of `malradar`, `new`, or
`benign`.

Run one APK:

```bash
sha=<apk_sha256>
python3 src/preparation/run_static_analysis.py \
  --folder malradar \
  --apk "$sha" \
  --output-root results/static_analysis
```

Run a small sampled batch:

```bash
python3 src/preparation/run_static_analysis.py \
  --folder malradar \
  --sample-size 5 \
  --seed 42 \
  --output-root results/static_analysis
```

The generated files are written to:

```text
results/static_analysis/entrypoints/
results/static_analysis/call_chain/
results/static_analysis/functions/
results/static_analysis/reachable_func/
```

## From Artifact

This path starts from the released `artifacts/` directory. It uses the
precomputed `functions`, `call_chain`, `entrypoints`, `reachable_func`, and
`meta_info` files.

### Function Summaries

```bash
model=gpt
split=malradar
sha=<apk_sha256>

python3 src/preparation/get_functionality_summary.py --model "$model" --folder "$split" --apk "$sha"
```

Omit `--apk` to run the whole split.

### Context and No-Context Summaries

```bash
python3 src/preparation/get_context_summary.py --model "$model" --folder "$split" --apk "$sha"
python3 src/preparation/get_no_context_summary.py --model "$model" --folder "$split" --apk "$sha"
```

### Behavior Reports

```bash
python3 src/generate_behavior/get_behavior.py --model "$model" --folder "$split" --flag context --apk "$sha"
python3 src/generate_behavior/get_behavior.py --model "$model" --folder "$split" --flag no_context --apk "$sha"
python3 src/generate_behavior/get_meta_behavior.py --model "$model" --folder "$split" --apk "$sha"
```

Outputs are written under `results/summary/`, `results/context_summary/`,
`results/no_context_summary/`, `results/behavior/`,
`results/no_context_behavior/`, and `results/meta_behavior/`.

## Metrics

After summaries and behavior reports are generated, compute metrics with:

```bash
python3 src/metrics/run_metrics.py --model "$model" --flag context
python3 src/metrics/run_metrics.py --model "$model" --flag no_context
python3 src/metrics/run_metrics.py --model "$model" --flag meta
```

To skip judge-model calls:

```bash
python3 src/metrics/run_metrics.py --model "$model" --flag context --skip-eas
```

Individual metric scripts are also available:

```bash
python3 src/metrics/aec_b_f1.py --model "$model" --flag context
python3 src/metrics/fpcr_tpmr_f1c.py --model "$model" --flag context
python3 src/metrics/eas.py --model "$model" --flag context --split malradar --judge-model gpt-5
```

## Minimal Verification

The source tree can be checked without downloading APKs or configuring model
credentials:

```bash
python3 -m compileall -q src
python3 - <<'PY'
import json
from pathlib import Path

expected = {
    "archived_sample_info.json": 200,
    "latest_sample_info.json": 30,
    "benign_sample_info.json": 25,
}
for name, count in expected.items():
    actual = len(json.loads((Path("info") / name).read_text()))
    assert actual == count, (name, actual, count)
print("metadata check passed: 200 archived malware + 30 recent malware + 25 benign")
PY
```

## Reproducibility Scope

- Metric computation over released reports does not require local model
  inference hardware.
- Regenerating LLM outputs requires the corresponding API credentials or a
  compatible self-hosted endpoint.
- Static analysis and all handling of APKs must take place in an isolated
  research environment. Do not install or execute released samples on a
  personal or production device.
- Exact numerical reproduction can be affected by changes to hosted model
  endpoints. The released reports preserve the outputs used by the paper.

## Citation

Please cite the accompanying article when using MalEval:

```bibtex
@article{zheng2026maleval,
  author  = {Xinran Zheng and Xingzhi Qian and Yiling He and Shuo Yang and Lorenzo Cavallaro},
  title   = {Is {“Knowing It’s Malicious”} Enough? Evaluating {LLMs} for Fine-Grained Malware Behavior Auditing},
  journal = {Proceedings of the ACM on Software Engineering},
  year    = {2026},
  doi     = {10.1145/3832187}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
When the archival artifact DOI is assigned, it should be used to cite the
versioned artifact snapshot in addition to the article.

## License and Safety

See [LICENSE.txt](LICENSE.txt) for the license applying to author-created code,
documentation, metadata, annotations, and derived outputs. Raw APKs and other
third-party materials are not relicensed by the MalEval authors. See
[SECURITY.md](SECURITY.md) before handling malware-related files.
