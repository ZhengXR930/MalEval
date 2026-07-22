# MalEval Data Documentation

## Composition

MalEval indexes 255 Android applications in three fixed splits:

| Split | Metadata file | Applications | Role |
| --- | --- | ---: | --- |
| `malradar` | `info/archived_sample_info.json` | 200 | Archived malware |
| `new` | `info/latest_sample_info.json` | 30 | Recent malware |
| `benign` | `info/benign_sample_info.json` | 25 | Simulated false positives |

The released data include sample metadata, expert-written raw and structured
behavior reports, static-analysis outputs, model-generated reports, and APK
archives used by the evaluation. Sample identifiers are SHA-256 values.

## Access and Version Pinning

Large files are stored in the public MalEval dataset repository:

- Dataset: <https://huggingface.co/datasets/Xinzxr/MalEval>
- Camera-ready revision:
  [`a5bd8d81116d2936a3edcd7eae5b26403aabdbd4`](https://huggingface.co/datasets/Xinzxr/MalEval/tree/a5bd8d81116d2936a3edcd7eae5b26403aabdbd4)

Download the archives from that pinned revision and extract them according to
the layout documented in `README.md`. The repository contains downloadable
archives and is not intended to be loaded as a tabular Hugging Face
`datasets` object; an unavailable Dataset Viewer does not prevent download.

## Provenance

The benchmark was constructed by the authors of the accompanying article from
Android applications paired with expert-written malware reports. The malware
portion comprises 200 archived samples associated with the MalRadar-based
collection and 30 recent expert-reported samples. The benign split contains 25
applications used as simulated false positives. Malware-family information is
inherited from source reports where available. Behavior annotations and
structured evidence used by the evaluation were produced and validated by the
authors. Sample-level report references are recorded in the released metadata.

## Rights and Responsible Use

The licenses in `LICENSE.txt` apply only to materials for which the MalEval
authors hold the necessary rights. Raw APKs, original malware reports, and
other third-party materials are not relicensed by the authors and remain
subject to their original rights, licenses, and terms of use.

This dataset is provided for defensive security research. APKs may contain
malicious functionality. Do not install or execute them on personal or
production systems. Use an isolated analysis environment with appropriate
network and filesystem controls, and comply with applicable laws,
institutional policies, and third-party terms.
