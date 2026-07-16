# MalEval

MalEval is a framework for evaluating Android malware behavior reports generated
by large language models. The code in this repository implements two execution
paths:

1. **From APK**: run static analysis on APK files to produce entry points, call
   chains, function bodies, and reachable functions.
2. **From artifact**: start from the precomputed static-analysis outputs and
   generate function summaries, context-intermediate representations, behavior
   reports, and evaluation metrics.

The released dataset is hosted separately at
`https://huggingface.co/datasets/Xinzxr/MalEval`.

## Repository Layout

```text
info/   Sample metadata for the benign, MalRadar, and new malware splits.
src/    Static analysis, summarization, behavior generation, and metrics code.
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

Use Python 3.10 or newer. The core scripts require common Python packages used
for Android static analysis and LLM calls, including `androguard`, `openai`,
`tqdm`, `pandas`, and `scikit-learn`.

Set API credentials before running LLM-based stages:

```bash
export OPENAI_API_KEY=<your_key>
```

`src/call_llm.py` also supports provider-specific environment variables for the
models used in the paper.

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
