# General-Purpose Audio Tagging (GP-AT) AudioSet Models Collection

This repository contains source-vendored implementations of general-purpose audio representation and AudioSet-oriented tagging models exposed through a unified PyTorch API. The integrations preserve model- and checkpoint-specific preprocessing, inference, and representation-extraction requirements while providing a common execution interface across CPU, NVIDIA CUDA, and Apple MPS backends.

Most selected checkpoints expose pretrained AudioSet tagging outputs. Exceptions are documented explicitly.

![Python 3.10](https://img.shields.io/badge/python-3.10.11-blue.svg) ![PyTorch](https://img.shields.io/badge/-PyTorch-333?style=flat&logo=pytorch)
---

## Table of Contents

1. [Overview](#overview)
2. [Models Summary](#models-summary)
3. [Vendoring Standard](#vendoring-standard)
4. [Quick Start](#quick-start)
5. [Models Catalog](#models-catalog)
6. [Requirements](#requirements)
7. [License](#license)

---

## Overview

This collection provides a standardized interface to 18 General-Purpose Audio Tagging (GP-AT) model/checkpoint configurations spanning convolutional, Transformer-based, self-supervised, and multimodal audio representation families. Each model has been vendorized following a consistent pattern that ensures:

- **Unified API**: All models expose the same `forward()`, `get_embedding()`, and `load_pretrained()` methods.
- **Controlled vendoring:** model-specific source code required for inference is retained locally while external dependencies are explicitly declared and versioned where required.
- **Reproducibility**: Constants, configurations, and preprocessing are clearly documented within each script.

---

## Models Summary

| Model | Selected checkpoint / configuration | Sample rate | Embedding dim. | Pretrained tagging output | Reference AudioSet mAP | Provenance |
|---|---|---:|---:|---|---:|---|
| AST | `audioset_10_10_0.4593.pth` | 16 kHz | 768 | 527 | 0.4593 | Artifact |
| AudioCLIP | `AudioCLIP-Full-Training.pt` | 44.1 kHz | 1024 | No pretrained GP-AT head | N/A | — |
| AudioMAE | `finetuned.pth` | 16 kHz | 768 | 527 | 0.4729 | Artifact |
| BEATs | `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | 16 kHz | 768 | 527 | 0.486 | Configuration |
| CED | `audiotransformer_base_mAP_4999.pt` | 16 kHz | 768 | 527 | 0.4999 | Artifact |
| CLAP | `630k-audioset-fusion-best.pt` | 48 kHz | 768 | Inherited 527-way AudioSet classifier | N/A | No authoritative tagging mAP reported for the final CLAP checkpoint |
| ConvNeXt | `convnext_tiny_471mAP.pth` | 32 kHz | 768 | 527 | 0.471 | Artifact |
| EfficientAT DyMN | `dymn20_as_mAP_493.pt` — DyMN20, width ×2.0 | 32 kHz | 1920 | 527 | 0.493 | Artifact |
| EfficientAT MobileNetV3 | `mn40_as_ext_mAP_487.pt` — MN40-AS-EXT, width ×4.0 | 32 kHz | 3840 | 527 | 0.487 | Artifact/configuration |
| E-PANNs | `checkpoint_closeto_.44.pt` — 50%-pruned CNN14 | 32 kHz | 2048 | 527 | ≈0.44 | Artifact label |
| HTS-AT | `HTSAT_AudioSet_Saved_3.ckpt` | 32 kHz | 768 | 527 | 0.471 | Configuration |
| M2D | `weights_ep69it3124-0.47998.pth` — M2D-AS, AS2M-FT, 32 kHz | 32 kHz | 768 | 527 | 0.47998 | Artifact |
| PANNs ResNet38 | `ResNet38_mAP=0.434.pth` | 32 kHz | 2048 | 527 | 0.434 | Artifact |
| PANNs Wavegram-Logmel-CNN14 | `Wavegram_Logmel_Cnn14_mAP=0.439.pth` | 32 kHz | 2048 | 527 | 0.439 | Artifact |
| PaSST | `passt-s-kd-ap.486.pt` | 32 kHz | 768 | 527 | 0.486 | Artifact |
| PSLA | `as_mdl_0_wa.pth` — weight-averaged model | 16 kHz | 1408 | 527 | 0.444 | Configuration |
| VGGish | `vggish_with_classifier.pth` | 16 kHz | 128 | Derivative 527-way classifier | N/A | — |
| YAMNet | `yamnet.pth` | 16 kHz | 1024 | 521 pretrained outputs aligned to 527 | 0.306 | Native 521-class configuration |

**Reference-metric provenance.** `Artifact` denotes a value explicitly associated with the selected released checkpoint or artifact. `Configuration` denotes an official value reported for the corresponding model configuration when the exact selected seed/checkpoint is not independently scored. `N/A` indicates that no authoritative pretrained AudioSet tagging mAP is available for the selected checkpoint/head. These reference values are retained for provenance and diagnostic comparison.

---

## Vendoring Standard

Each vendorized model follows a strict structure to ensure consistency and maintainability.

### File Structure

```text
models/
├── README.md
├── <model_name>/
│   ├── model.py          # Main vendorized script
│   └── requirements.txt  # (optional) Model-specific dependencies
```

### Script Sections

Every `model.py` file is organized into the following sections:

1. **Header Docstring**: Model description, original paper citation, repository URL.
2. **Constants**: `SAMPLE_RATE`, `CLASSES_NUM` (527), `EMBED_DIM`, and model-specific hyperparameters.
3. **Helper Functions**: Utility functions (mel spectrogram, data loading, etc.).
4. **Building Blocks**: Core architectural components (attention layers, encoder blocks, etc.).
5. **Main Model**: The primary model class with standardized methods.
6. **Demo**: Usage examples with dummy data.

### Standard API

All models implement the following interface:

```python
class Model(nn.Module):
    def __init__(self, ...):
        """Initialize the model architecture."""
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Perform audio tagging inference.
        
        Args:
            waveform: Input audio tensor of shape (batch, samples).
                      Must be at SAMPLE_RATE Hz.
        
        Returns:
            probs: Class probabilities of shape (batch, 527).
        """
        
    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embeddings.
        
        Args:
            waveform: Input audio tensor of shape (batch, samples).
        
        Returns:
            embedding: Audio embedding of shape (batch, EMBED_DIM).
        """
        
    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pth/.pt checkpoint file.
        """
```

### Naming Conventions

- **Constants**: `UPPER_SNAKE_CASE`
- **Classes**: `CamelCase`
- **Functions/Methods**: `lower_snake_case`
- **Section Headers**: Title Case (e.g., `# Helper Functions`)

---

## Quick Start

### Basic Usage

```python
import torch
from models.panns.model_resnet38 import ResNet38

# Initialize model
model = ResNet38()
model.load_pretrained("checkpoints/ResNet38_mAP=0.434.pth")
model.eval()

# Create dummy input (10 seconds at 32kHz)
waveform = torch.randn(1, 320000)

# Audio tagging
with torch.no_grad():
    probs = model(waveform)
    print(f"Top-5 classes: {probs.topk(5).indices}")

# Embedding extraction
with torch.no_grad():
    embedding = model.get_embedding(waveform)
    print(f"Embedding shape: {embedding.shape}")  # (1, 2048)
```

### Loading Different Models

```python
# PANNs ResNet38
from models.panns.model_resnet38 import ResNet38
model = ResNet38()
model.load_pretrained("path/to/ResNet38_mAP=0.434.pth")

# Audio Spectrogram Transformer
from models.ast.model import ASTModel
model = ASTModel()
model.load_pretrained("path/to/ast_audioset.pth")

# BEATs
from models.beats.model import BEATs, BEATsConfig
cfg = BEATsConfig()
model = BEATs(cfg)
model.load_pretrained("path/to/BEATs_iter3_plus_AS2M.pt")
```

---

## Models Catalog

### AST - Audio Spectrogram Transformer

- **Paper**: Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," in *Proc. Interspeech*, 2021, pp. 571–575.
- **Repository**: [https://github.com/YuanGongND/ast](https://github.com/YuanGongND/ast)
- **Checkpoint**: [audioset_10_10_0.4593.pth](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
- **Architecture**: Vision Transformer (ViT) adapted for audio spectrograms.

### AudioCLIP

- **Paper**: A. Guzhov, F. Raue, J. Hees, and A. Dengel, "AudioCLIP: Extending CLIP to Image, Text and Audio," arXiv:2106.13043, 2021.
- **Repository**: [https://github.com/AndreyGuzhov/AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP)
- **Checkpoint**: [AudioCLIP-Full-Training.pt](https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt)
- **Architecture**: CLIP model extended with ESResNeXt audio encoder.

### AudioMAE - Audio Masked Autoencoder

- **Paper**: P.-Y. Huang, H. Xu, J. Li, A. Baevski, M. Auli, W. Galuba, F. Metze, and C. Feichtenhofer, "Masked Autoencoders that Listen," in *Proc. NeurIPS*, 2022.
- **Repository**: [https://github.com/facebookresearch/AudioMAE](https://github.com/facebookresearch/AudioMAE)
- **Checkpoint**: [finetuned.pth](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq)
- **Architecture**: Masked autoencoder with ViT backbone for audio.

### BEATs

- **Paper**: S. Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers," in *Proc. ICML*, 2023, pp. 5178–5193.
- **Repository**: [https://github.com/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats)
- **Checkpoint**: [BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt](https://msranlcmtteamdrive.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)
- **Architecture**: Transformer with acoustic tokenizers for self-supervised pre-training.

### CED - Consistent Ensemble Distillation

- **Paper**: H. Dinkel, Y. Wang, Z. Yan, J. Zhang, and Y. Wang, "CED: Consistent Ensemble Distillation for Audio Tagging," in *Proc. ICASSP*, 2024.
- **Repository**: [https://github.com/RicherMans/CED](https://github.com/RicherMans/CED)
- **Checkpoint**: [ced-base.pth](https://zenodo.org/record/8275347)
- **Architecture**: Efficient transformer trained via ensemble knowledge distillation.

### CLAP - Contrastive Language-Audio Pretraining

- **Paper**: Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," in *Proc. ICASSP*, 2023.
- **Repository**: [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- **Checkpoint**: [630k-audioset-fusion-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt)
- **Architecture**: Contrastive audio-text model with HTSAT audio encoder.

### ConvNeXt-Audio

- **Paper**: T. Pellegrini, I. Khalfaoui-Hassani, E. Labbé, and T. Masquelier, "Adapting a ConvNeXt Model to Audio Classification on AudioSet," in *Proc. Interspeech*, 2023.
- **Repository**: [https://github.com/topel/audioset-convnext-inf](https://github.com/topel/audioset-convnext-inf)
- **Checkpoint**: [convnext_tiny_471mAP.pth](https://zenodo.org/record/8020843)
- **Architecture**: ConvNeXt-Tiny adapted for audio spectrograms.

### EfficientAT (DyMN / MobileNetV3)

- **Paper**: F. Schmid, S. Koutini, and G. Widmer, "Efficient Large-Scale Audio Tagging via Transformer-To-CNN Knowledge Distillation," in *Proc. ICASSP*, 2023.
- **Paper (DyMN)**: F. Schmid, S. Koutini, and G. Widmer, "Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio Models," *IEEE/ACM Trans. Audio, Speech, Language Process.*, submitted.
- **Repository**: [https://github.com/fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT)
- **Checkpoints**: Available via GitHub Releases.
- **Architecture**: MobileNetV3 and Dynamic MobileNet trained with knowledge distillation from transformers.

### E-PANNs - Efficient PANNs

- **Paper**: A. Singh, H. Liu, and M. D. Plumbley, "E-PANNs: Sound Recognition Using Efficient Pre-trained Audio Neural Networks," in *Proc. Inter-Noise*, 2023, pp. 7220–7228.
- **Paper**: A. Singh and M. D. Plumbley, "Efficient CNNs via Passive Filter Pruning," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 33, pp. 1763–1774, 2025.
- **Repository**: [https://github.com/Arshdeep-Singh-Boparai/E-PANNs](https://github.com/Arshdeep-Singh-Boparai/E-PANNs)
- **Checkpoint**: [efficient_cnn14.pth](https://doi.org/10.5281/zenodo.7939403)
- **Architecture**: Pruned CNN14 with ~70% parameter reduction.

### HTS-AT - Hierarchical Token-Semantic Audio Transformer

- **Paper**: K. Chen, X. Du, B. Zhu, Z. Ma, T. Berg-Kirkpatrick, and S. Dubnov, "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection," in *Proc. ICASSP*, 2022.
- **Repository**: [https://github.com/RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)
- **Checkpoint**: [HTSAT_AudioSet_Saved_1.ckpt](https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6)
- **Architecture**: Swin Transformer with token-semantic module.

### M2D - Masked Modeling Duo

- **Paper**: D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Masked Modeling Duo: Towards a Universal Audio Pre-training Framework," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 32, pp. 2391–2406, 2024.
- **Paper (M2D-CLAP)**: D. Niizumi et al., "M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP," *IEEE Access*, vol. 13, pp. 163313–163330, 2025.
- **Repository**: [https://github.com/nttcslab/m2d](https://github.com/nttcslab/m2d)
- **Checkpoint**: Available via GitHub Releases.
- **Architecture**: Self-supervised learning with dual masked prediction.

### PANNs (CNN14 / ResNet38)

- **Paper**: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 28, pp. 2880–2894, 2020.
- **Repository**: [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- **Checkpoints**: [Zenodo](https://zenodo.org/record/3987831)
- **Architecture**: ResNet38, Wavegram-Logmel-CNN14 variants.

### PaSST - Patchout Audio Spectrogram Transformer

- **Paper**: K. Koutini, J. Schlüter, H. Eghbal-zadeh, and G. Widmer, "Efficient Training of Audio Transformers with Patchout," in *Proc. Interspeech*, 2022, pp. 2753–2757.
- **Repository**: [https://github.com/kkoutini/PaSST](https://github.com/kkoutini/PaSST)
- **Checkpoint**: Available via GitHub Releases.
- **Architecture**: Vision Transformer with Patchout regularization.

### PSLA

- **Paper**: Y. Gong, Y.-A. Chung, and J. Glass, "PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 29, pp. 3292–3306, 2021.
- **Repository**: [https://github.com/YuanGongND/psla](https://github.com/YuanGongND/psla)
- **Checkpoint**: [Dropbox](https://www.dropbox.com/sh/ihfbxcemxamihz9/AAD9zqnUptZzyZlquqpWllDya)
- **Architecture**: EfficientNet-B2 with 4-headed attention.

### VGGish

- **Paper**: S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification," in *Proc. ICASSP*, 2017, pp. 131–135.
- **Repository (Original)**: [https://github.com/tensorflow/models/tree/master/research/audioset](https://github.com/tensorflow/models/tree/master/research/audioset)
- **Repository (PyTorch)**: [https://github.com/w-hc/torch_audioset](https://github.com/w-hc/torch_audioset)
- **Architecture**: VGG-style CNN for audio embedding extraction.

### YAMNet

- **Paper**: (Based on MobileNetV1) A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.
- **Repository (Original)**: [https://github.com/tensorflow/models/tree/master/research/audioset/yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- **Repository (PyTorch)**: [https://github.com/w-hc/torch_audioset](https://github.com/w-hc/torch_audioset)
- **Architecture**: MobileNetV1-based audio classifier.

---

## Requirements

Common dependencies for all models:

```python
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.19.0
scipy>=1.5.0
librosa>=0.8.0
```

Some models may require additional dependencies. Check the `requirements.txt` file in each model subdirectory if present.

---

## Licensing

The GP-AT framework contains source-vendored components released under different upstream licenses. Each vendored model directory retains the corresponding upstream license or provenance information and must be used according to those terms.

| Integration | Upstream license |
|---|---|
| AST | BSD 3-Clause |
| AudioCLIP | MIT |
| AudioMAE | CC BY-NC 4.0 |
| BEATs | MIT |
| CED | GPL v3 |
| CLAP | CC0 1.0 |
| ConvNeXt | MIT |
| EfficientAT | MIT |
| E-PANNs | MIT |
| HTS-AT | MIT |
| M2D | See retained upstream license |
| PANNs | MIT |
| PaSST | Apache 2.0 |
| PSLA | BSD 3-Clause |
| VGGish PyTorch integration | MIT |
| YAMNet PyTorch integration | MIT |

---

## Citation

If you use this collection in your research, please cite the respective papers for each model used. For the vendorization framework itself, please cite:

```bibtex
@misc{vendorized_audio_models,
  author = {Giacomelli, Stefano},
  title = {Vendorized GP-AT AudioSet Models Collection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/StefanoGiacomelli/vendorized-audio-models}
}
```

---

<div align="center">

**Stefano Giacomelli**  
*Ph.D. Candidate in Information and Communication Technology*
Department of Information Engineering, Computer Science and Mathematics (DISIM)
University of L'Aquila, Italy

![DISIM_logo](https://phdict.disim.univaq.it/wp-content/uploads/2024/06/logo-univaq-disim-2-2-768x283.png){width="400" height="150"}

📧 Email: stefano.giacomelli@graduate.univaq.it  
🔗 GitHub: https://github.com/StefanoGiacomelli 
🆔 ORCID: https://orcid.org/0009-0009-0438-1748 
🎓 Scholar: https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=it  
💼 LinkedIn: https://www.linkedin.com/in/stefano-giacomelli-811654135

---

*This project is funded under the Italian National Ministry of University and Research, for the Italian National Recovery and Resilience Plan (NRRP) "Methods of Computational Auditory Scene Analysis and Synthesis supporting eXtended and Immersive Reality Services"*

</div>
