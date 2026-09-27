"""Model registry and adapter for local GP-AT wrappers."""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .device import resolve_torch_device
from .labels import AudioSetLabelResolver, ClassInfo, ClassSelector, default_audioset_labels_path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    class_name: str
    sample_rate: int
    default_checkpoint: str
    num_classes: int = 527
    label_space: str = "audioset"


@dataclass(frozen=True)
class LoadedModelInfo:
    model_name: str
    model_class: str
    checkpoint_path: Optional[str]
    sample_rate: int
    num_classes: int
    requested_device: str
    effective_device: str
    label_space: str
    model_load_seconds: float


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "ast": ModelSpec("ast", "models.ast.model", "ASTModel", 16000, "models/ast/audioset_10_10_0.4593.pth"),
    "audioclip": ModelSpec("audioclip", "models.audioclip.model", "AudioCLIP", 44100, "models/audioclip/AudioCLIP-Full-Training.pt"),
    "audiomae": ModelSpec("audiomae", "models.audiomae.model", "AudioMAE", 16000, "models/audiomae/finetuned.pth"),
    "beats": ModelSpec("beats", "models.beats.model", "BEATs", 16000, "models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
    "ced": ModelSpec("ced", "models.ced.model", "CEDBase", 16000, "models/ced/audiotransformer_base_mAP_4999.pt"),
    "clap": ModelSpec("clap", "models.clap.model", "CLAP", 48000, "models/clap/630k-audioset-fusion-best.pt"),
    "convnext": ModelSpec("convnext", "models.convnext.model", "ConvNeXt", 32000, "models/convnext/convnext_tiny_471mAP.pth"),
    "efficientat_dymn": ModelSpec("efficientat_dymn", "models.efficientat.model_dymn", "EfficientAT_DyMN", 32000, "models/efficientat/dymn20_as_mAP_493.pt"),
    "efficientat_mn": ModelSpec("efficientat_mn", "models.efficientat.model_mn", "EfficientAT_MN", 32000, "models/efficientat/mn40_as_ext_mAP_487.pt"),
    "epanns": ModelSpec("epanns", "models.epanns.model", "EPANNs", 32000, "models/epanns/checkpoint_closeto_.44.pt"),
    "htsat": ModelSpec("htsat", "models.htsat.model", "HTSAT", 32000, "models/htsat/HTSAT_AudioSet_Saved_3.ckpt"),
    "m2d": ModelSpec("m2d", "models.m2d.model", "M2D", 32000, "models/m2d/weights_ep69it3124-0.47998.pth"),
    "panns_resnet38": ModelSpec("panns_resnet38", "models.panns.model_resnet38", "ResNet38", 32000, "models/panns/ResNet38_mAP=0.434.pth"),
    "panns_wavegram_logmel_cnn14": ModelSpec("panns_wavegram_logmel_cnn14", "models.panns.model_wavegram_logmel_cnn14", "Wavegram_Logmel_Cnn14", 32000, "models/panns/Wavegram_Logmel_Cnn14_mAP=0.439.pth"),
    "passt": ModelSpec("passt", "models.passt.model", "PaSST", 32000, "models/passt/passt-s-kd-ap.486.pt"),
    "psla": ModelSpec("psla", "models.psla.model", "EffNetAttention", 16000, "models/psla/as_mdl_0_wa.pth"),
    "vggish": ModelSpec("vggish", "models.vggish.model", "VGGish", 16000, "models/vggish/vggish_with_classifier.pth"),
    "yamnet": ModelSpec("yamnet", "models.yamnet.model", "YAMNet", 16000, "models/yamnet/yamnet.pth"),
}


class GPATModelAdapter:
    """Normalize loading and inference for local GP-AT model wrappers."""

    def __init__(
        self,
        model: nn.Module,
        spec: ModelSpec,
        checkpoint_path: Optional[Union[str, Path]],
        requested_device: str,
        effective_device: str,
        model_load_seconds: float,
        label_resolver: Optional[AudioSetLabelResolver],
    ):
        self.model = model
        self.spec = spec
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.requested_device = requested_device
        self.effective_device = effective_device
        self.model_load_seconds = model_load_seconds
        self.label_resolver = label_resolver
        self.model.eval()

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.model, "sample_rate", self.spec.sample_rate))

    @property
    def num_classes(self) -> int:
        return self.spec.num_classes

    @property
    def info(self) -> LoadedModelInfo:
        return LoadedModelInfo(
            model_name=self.spec.name,
            model_class=f"{self.spec.module}.{self.spec.class_name}",
            checkpoint_path=self.checkpoint_path,
            sample_rate=self.sample_rate,
            num_classes=self.num_classes,
            requested_device=self.requested_device,
            effective_device=self.effective_device,
            label_space=self.spec.label_space,
            model_load_seconds=self.model_load_seconds,
        )

    def predict_proba(self, waveform: Union[torch.Tensor, Any]) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        waveform = waveform.float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform shape (batch, samples), got {tuple(waveform.shape)}")

        waveform = waveform.to(self.effective_device)
        with torch.no_grad():
            output = self.model(waveform)
        probabilities = self._normalize_output(output)
        if probabilities.shape[-1] <= 0:
            raise RuntimeError("Model returned an empty class dimension")
        return probabilities.detach().cpu()

    def resolve_classes(self, selectors: List[ClassSelector]) -> List[ClassInfo]:
        if self.spec.label_space != "audioset" or self.label_resolver is None:
            for selector in selectors:
                if not isinstance(selector, int) and not str(selector).strip().isdigit():
                    raise ValueError(
                        f"Model '{self.spec.name}' does not expose name-based resolution; use indices."
                    )
            return [
                ClassInfo(index=int(selector), mid=None, name=f"class_{int(selector)}", selector=selector)
                for selector in selectors
            ]
        return self.label_resolver.resolve_many(selectors, output_size=self.num_classes)

    def _normalize_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            if "clipwise_output" in output:
                output = output["clipwise_output"]
            elif "probabilities" in output:
                output = output["probabilities"]
            elif "logits" in output:
                output = torch.sigmoid(output["logits"])
            else:
                raise RuntimeError(f"Unsupported model output keys: {sorted(output.keys())}")
        elif isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Unsupported model output type: {type(output).__name__}")
        if output.dim() == 1:
            output = output.unsqueeze(0)
        if output.dim() != 2:
            raise RuntimeError(f"Expected output shape (batch, classes), got {tuple(output.shape)}")
        return output


def available_models() -> List[Dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "sample_rate": spec.sample_rate,
            "default_checkpoint": spec.default_checkpoint,
            "num_classes": spec.num_classes,
            "label_space": spec.label_space,
            "model_class": f"{spec.module}.{spec.class_name}",
        }
        for spec in MODEL_REGISTRY.values()
    ]


def load_model_adapter(
    model_name: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "auto",
    project_root: Optional[Union[str, Path]] = None,
    load_checkpoint: bool = True,
) -> GPATModelAdapter:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    effective_device = resolve_torch_device(device)
    spec = MODEL_REGISTRY[model_name]

    started = time.perf_counter()
    model_class = _import_model_class(spec)
    model = _instantiate_model(model_class, spec)

    resolved_checkpoint: Optional[Path] = None
    if load_checkpoint:
        resolved_checkpoint = _resolve_checkpoint_path(
            checkpoint_path or spec.default_checkpoint,
            project_root=root,
        )
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for '{model_name}': {resolved_checkpoint}")
        if not hasattr(model, "load_pretrained"):
            raise TypeError(f"Model '{model_name}' does not implement load_pretrained()")
        model.load_pretrained(str(resolved_checkpoint))

    model = model.to(effective_device)
    model_load_seconds = time.perf_counter() - started
    resolver = AudioSetLabelResolver(default_audioset_labels_path(root))
    return GPATModelAdapter(
        model=model,
        spec=spec,
        checkpoint_path=resolved_checkpoint,
        requested_device=device,
        effective_device=effective_device,
        model_load_seconds=model_load_seconds,
        label_resolver=resolver,
    )


def _import_model_class(spec: ModelSpec):
    try:
        module = importlib.import_module(spec.module)
        return getattr(module, spec.class_name)
    except Exception as exc:
        raise ImportError(f"Could not import {spec.module}.{spec.class_name}: {exc}") from exc


def _instantiate_model(model_class, spec: ModelSpec) -> nn.Module:
    signature = inspect.signature(model_class)
    kwargs = {"sample_rate": spec.sample_rate} if "sample_rate" in signature.parameters else {}
    try:
        return model_class(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Could not instantiate model '{spec.name}': {exc}") from exc


def _resolve_checkpoint_path(path: Union[str, Path], project_root: Path) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    return checkpoint
