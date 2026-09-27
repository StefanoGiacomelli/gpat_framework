# GP-AT: General-Purpose Audio Tagging Framework

`GP-AT` is a research framework for the controlled integration, evaluation, profiling, and
application-level inference of heterogeneous pretrained audio-tagging and audio-representation
models. The repository exposes source-vendored model implementations through a common PyTorch
interface while preserving checkpoint-specific sampling rates, preprocessing, output semantics,
embedding definitions, and runtime constraints.

The current codebase supports **18 model/checkpoint configurations**, unified evaluation on
**AudioSet, AudioSet-R, and FSD50K**, device-aware profiling, fixed-window live inference, an
interactive replay/inference GUI, and companion taxonomy utilities.

> **Repository policy:** pretrained checkpoint payloads and raw dataset audio are intentionally
> not versioned. They must be acquired from their authoritative sources.

## What you can do

- **Run heterogeneous GP-AT models through a common model contract** based on
  `load_pretrained()`, `forward()`, and `get_embedding()`.
- **Evaluate AudioSet-oriented model outputs** on AudioSet, AudioSet-R, and FSD50K with explicit
  label-space compatibility and projection metadata.
- **Profile local model/checkpoint configurations** on CPU, CUDA, or Apple MPS, including
  latency, parameter/memory statistics, complexity analysis, and minimum-input diagnostics.
- **Run fixed-window live microphone inference** with configurable window, hop, device, and
  Top-\(N\) display settings.
- **Use an interactive PySide6 interface** for local audio, audio-device capture, and YouTube
  ingestion, with probability trajectories and runtime diagnostics.
- **Reuse taxonomy utilities** for AudioSet-oriented metadata normalization and SALT/BST
  interoperability.
- **Inspect source-vendored model provenance and checkpoint expectations** without committing
  multi-gigabyte checkpoint files to the repository.

## Framework organization

The repository separates model-specific scientific identity from reusable orchestration:

```text
GP-AT model/checkpoint
        |
        v
source-vendored model wrapper
        |
        +-------------------+
        |                   |
        v                   v
evaluation              profiling
        |                   |
        +---------+---------+
                  |
                  v
       inference / deployment
          |              |
          v              v
     live terminal      GUI/replay
```

The common orchestration layer does **not** assume that models are numerically or semantically
interchangeable. Native sample rate, input support, preprocessing, output mapping, checkpoint
structure, and embedding semantics remain properties of each selected integration.

## Supported model integrations

The current registry contains the following model/checkpoint configurations:

| Registry name | Family | Native sample rate | Default embedding |
|---|---|---:|---:|
| `ast` | Audio Spectrogram Transformer | 16 kHz | 768 |
| `audioclip` | AudioCLIP | 44.1 kHz | 1024 |
| `audiomae` | AudioMAE | 16 kHz | 768 |
| `beats` | BEATs | 16 kHz | 768 |
| `ced` | CED | 16 kHz | 768 |
| `clap` | CLAP | 48 kHz | 768 |
| `convnext` | ConvNeXt-Audio | 32 kHz | 768 |
| `efficientat_dymn` | EfficientAT DyMN | 32 kHz | 1920 |
| `efficientat_mn` | EfficientAT MobileNetV3 | 32 kHz | 3840 |
| `epanns` | E-PANNs | 32 kHz | 2048 |
| `htsat` | HTS-AT | 32 kHz | 768 |
| `m2d` | M2D | 32 kHz | 768 |
| `panns_resnet38` | PANNs ResNet38 | 32 kHz | 2048 |
| `panns_wavegram_logmel_cnn14` | PANNs Wavegram-Logmel-CNN14 | 32 kHz | 2048 |
| `passt` | PaSST | 32 kHz | 768 |
| `psla` | PSLA | 16 kHz | 1408 |
| `vggish` | VGGish-based integration | 16 kHz | 128 |
| `yamnet` | YAMNet-based integration | 16 kHz | 1024 |

See [`models/README.md`](models/README.md) for checkpoint provenance, model-specific notes, upstream
repositories, and the licensing information retained for the vendored integrations.

## Installation

The audited development environment uses Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional local taxonomy packages can be installed in editable mode when those workflows are needed:

```bash
pip install -e taxonomy/audioset_tools
pip install -e taxonomy/salt/py-salt
```

The repository intentionally does not install or download pretrained checkpoints automatically.

## Checkpoints

Checkpoint files are excluded from Git. The local model registry expects them under `models/`
using the model-specific filenames documented in [`models/README.md`](models/README.md).

For example:

```text
models/
├── panns/
│   ├── model_resnet38.py
│   ├── model_wavegram_logmel_cnn14.py
│   ├── ResNet38_mAP=0.434.pth                 # local only
│   └── Wavegram_Logmel_Cnn14_mAP=0.439.pth   # local only
├── ast/
│   ├── model.py
│   └── audioset_10_10_0.4593.pth              # local only
└── ...
```

The profiling registry discovers only integrations for which both the source module and expected
local checkpoint are present. Evaluation and real-time inference also allow an explicit checkpoint
path where supported by the corresponding entry point.

## Dataset layout

Raw audio is not distributed. The default evaluation code expects local data in the following
locations:

```text
datasets/
├── AudioSet_meta/
│   ├── class_labels_indices.csv
│   └── eval_segments.csv
├── AudioSet-R_meta/
│   └── AudioSet-R_eval.json
├── AudioSet_data/
│   └── eval_segments/              # ignored: local WAV files
└── FSD50K_meta/
    └── FSD50K.ground_truth/
        ├── eval.csv
        └── vocabulary.csv
```

FSD50K audio is expected locally under:

```text
datasets/FSD50K_data/FSD50K.eval_audio/
```

Alternative audio roots can be supplied to the evaluation CLI.

## Quick start

Run commands from the repository root.

### List a model through profiling

After placing its checkpoint at the expected local path:

```bash
python main_profile.py \
  --models panns_resnet38 \
  --device cpu \
  --skip-complexity \
  --skip-minimum-input
```

Profiling results are written to `profile_results/` by default and are intentionally ignored by Git.

### Evaluate a model

AudioSet:

```bash
python main_evaluate.py \
  --model panns_resnet38 \
  --dataset audioset \
  --accelerator auto \
  --audio-root-audioset /absolute/path/to/audioset/eval_segments
```

AudioSet-R:

```bash
python main_evaluate.py \
  --model panns_resnet38 \
  --dataset audioset_r \
  --accelerator auto \
  --audio-root-audioset-r /absolute/path/to/audioset/eval_segments
```

FSD50K:

```bash
python main_evaluate.py \
  --model panns_resnet38 \
  --dataset fsd50k \
  --accelerator auto \
  --audio-root-fsd50k /absolute/path/to/FSD50K.eval_audio
```

Use:

```bash
python main_evaluate.py --help
```

for batch size, precision, threshold, deterministic seed, dataset selection, and partial-evaluation
controls.

### Live real-time inference

List audio devices:

```bash
python main_rt_inference.py --list-devices
```

Run fixed-window microphone inference:

```bash
python main_rt_inference.py \
  --model panns_wavegram_logmel_cnn14 \
  --checkpoint models/panns/Wavegram_Logmel_Cnn14_mAP=0.439.pth \
  --device auto \
  --window-size 1.0 \
  --hop-size 0.5 \
  --top-n 10
```

If multiple acquisition channels are selected, the live interface averages them to the mono
waveform expected by the current GP-AT wrappers.

### Interactive GUI

```bash
python main_inference_gui.py
```

The GUI supports:

- model/checkpoint/device selection;
- local audio-file loading;
- audio-device capture;
- YouTube ingestion through `yt-dlp`;
- spectrogram and Top-\(N\) probability visualization;
- runtime diagnostics;
- structured inference export.

Model execution occurs in a subprocess so that native model/backend failures do not directly
terminate the desktop interface.

## Evaluation semantics

The current evaluation layer supports:

- **AudioSet Standard**: direct alignment to the canonical 527-class AudioSet MID order;
- **AudioSet-R**: exact intersection with the canonical AudioSet class space;
- **FSD50K**: projection from canonical AudioSet outputs into the native 200-class FSD50K space.

The FSD50K mapping combines exact MID correspondences with reviewed aggregate projections stored in:

```text
evaluation/mappings/fsd50k_to_audioset_projection.json
```

Metrics are enabled only when they are compatible with the active target space. Dataset/model
compatibility and mapping coverage are recorded in generated evaluation results.

## Profiling

The profiler operates on the same local model registry used by the evaluation layer.

```bash
python main_profile.py --help
```

Principal controls include:

```text
--models
--device cpu|cuda|mps
--duration-seconds
--warmup-runs
--measured-runs
--skip-complexity
--skip-minimum-input
```

Each attempt produces a timestamped JSON record under `profile_results/`. Device fallback,
environment information, parameter statistics, timing distributions, memory observations,
complexity analysis, and minimum-input diagnostics are recorded where supported.

## Taxonomy utilities

The repository also contains two companion taxonomy components:

```text
taxonomy/audioset_tools/    normalized AudioSet/VGGSound metadata operations
taxonomy/salt/              SALT assets, BST/SALT mapping utilities, examples, and py-salt
```

These components are retained because the GP-AT research workflow treats semantic target-space
definition and mapping as part of reproducible model evaluation rather than as an external
post-processing detail.

## Repository layout

```text
evaluation/                 Unified dataset/model evaluation and metrics
inference/
  profiling/                Device-aware model profiling
  real_time/                Live/replay inference engine and PySide6 GUI
models/                     Source-vendored GP-AT integrations
training/                   Shared losses, metrics, and post-processing assets
taxonomy/
  audioset_tools/           Audio metadata normalization utilities
  salt/                     SALT/BST taxonomy interoperability assets
utils/                      Shared DSP and model utilities
datasets/                   Versioned metadata + ignored local audio payloads
main_evaluate.py            Evaluation entry point
main_profile.py             Profiling entry point
main_rt_inference.py        Live microphone entry point
main_inference_gui.py       Desktop GUI entry point
requirements.txt            Research-environment dependencies
```

## Data, checkpoints, and generated results

The public repository is intentionally source-centered.

The following are **not** committed:

- pretrained model checkpoints;
- AudioSet/FSD50K raw audio;
- large upstream metadata not required by the current public workflows;
- local profiling/evaluation/replay outputs;
- machine-local cache and environment state.

Generated results may include absolute local paths and should be sanitized before being published
as archival evidence.

## Current boundaries

- The framework is research software; commands are intended to be executed from a source checkout.
- The model registry currently covers 18 selected local integrations and assumes their expected
  checkpoint assets have been acquired separately.
- Model families retain heterogeneous preprocessing, minimum-input, output, and embedding semantics.
  A common API does not imply numerical interchangeability.
- AudioCLIP does not expose the same pretrained GP-AT classifier contract as the standard
  527-output integrations; model-specific exceptions are documented in `models/README.md`.
- YAMNet originates from a 521-class configuration and is aligned to the canonical 527-class
  interface by its local integration.
- Evaluation is currently implemented for AudioSet, AudioSet-R, and FSD50K.
- Raw datasets and pretrained weights are not redistributed.
- Real-time input is reduced to mono before inference by the current live workflow.
- Profiling and evaluation outputs are generated locally and are not treated as immutable model
  identity.

## Licensing and third-party code

The repository contains source-vendored components from multiple upstream projects. Their retained
license/provenance files remain authoritative for those components; see
[`models/README.md`](models/README.md).

Model checkpoints and dataset audio are not redistributed by this repository.

## Citation

When using an integrated model, cite the corresponding upstream paper listed in
[`models/README.md`](models/README.md).

## Contact

**Stefano Giacomelli**  
Ph.D. Candidate in Information and Communication Technology  
Department of Information Engineering, Computer Science and Mathematics (DISIM)  
University of L'Aquila, Italy

- GitHub: <https://github.com/StefanoGiacomelli>
- ORCID: <https://orcid.org/0009-0009-0438-1748>
- Google Scholar: <https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=en>
