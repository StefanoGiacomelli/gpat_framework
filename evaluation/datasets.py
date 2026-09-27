"""Evaluation dataset parsers and Lightning data module."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

_CACHE_ROOT = Path(tempfile.gettempdir()) / "gpat_eval_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - import failure is reported by CLI/tests.
    pl = None

import torchaudio

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional fallback decoder.
    sf = None

DATASET_NAMES = ("audioset", "audioset_r", "fsd50k")


@dataclass(frozen=True)
class LabelInfo:
    native_id: str
    name: str
    evaluation_index: Optional[int] = None
    model_mid: Optional[str] = None
    model_index: Optional[int] = None
    aggregate_model_indices: Optional[List[int]] = None
    aggregate_model_labels: Optional[List[str]] = None
    mapping_relation: str = "none"
    parent_id: Optional[str] = None


@dataclass
class EvaluationSample:
    clip_id: str
    audio_path: Path
    target_indices: List[int]
    native_label_ids: List[str]
    metadata: Dict[str, Any]


@dataclass
class TopClassInfo:
    rank: int
    native_class_id: str
    evaluation_index: Optional[int]
    native_class_name: str
    positive_sample_count: int
    prevalence_fraction: float
    mapped_model_output_class_id: Optional[str]
    mapped_model_output_index: Optional[int]
    mapped_model_output_indices: Optional[List[int]]
    mapped_class_name: Optional[str]
    mapping_relation: str
    status: str
    reason: Optional[str] = None


@dataclass
class DatasetSpec:
    name: str
    display_name: str
    metadata_path: Path
    default_audio_root: Optional[Path]
    split: str
    native_label_space: str
    evaluation_label_space: str
    labels: List[LabelInfo]
    native_class_count: int
    samples: List[EvaluationSample]
    top10: List[TopClassInfo]
    compatibility: Dict[str, Any]
    class_frequencies: Dict[str, int]
    skipped_samples: List[Dict[str, Any]] = field(default_factory=list)
    mapping_coverage: Dict[str, Any] = field(default_factory=dict)
    prediction_projection: Optional[List[List[int]]] = None
    supports_audioset_hierarchy: bool = True

    @property
    def class_count(self) -> int:
        return len(self.labels)

    @property
    def evaluated_sample_count(self) -> int:
        return len(self.samples)

    @property
    def total_metadata_samples(self) -> int:
        return len(self.samples) + len(self.skipped_samples)

    def effective_evaluation_indices(
        self,
        valid_model_output_indices: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Return evaluation-space classes supported by the current model."""
        valid_model = None if valid_model_output_indices is None else set(int(i) for i in valid_model_output_indices)

        if self.prediction_projection is None:
            if valid_model is None:
                return list(range(self.class_count))
            return [index for index in range(self.class_count) if index in valid_model]

        valid_eval: List[int] = []
        for evaluation_index, source_indices in enumerate(self.prediction_projection):
            usable = source_indices if valid_model is None else [i for i in source_indices if i in valid_model]
            if usable:
                valid_eval.append(evaluation_index)
        return valid_eval

    def project_model_scores(
        self,
        scores: torch.Tensor,
        valid_model_output_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Project canonical 527-way model scores into the dataset evaluation space."""
        if self.prediction_projection is None:
            return scores

        if scores.ndim != 2:
            raise ValueError(f"expected 2-D model scores, got shape {tuple(scores.shape)}")

        valid_model = None if valid_model_output_indices is None else set(int(i) for i in valid_model_output_indices)
        projected: List[torch.Tensor] = []
        for source_indices in self.prediction_projection:
            usable = source_indices if valid_model is None else [i for i in source_indices if i in valid_model]
            if not usable:
                # This class is excluded later by the model-specific evaluation mask.
                projected.append(torch.zeros(scores.shape[0], device=scores.device, dtype=scores.dtype))
                continue
            index = torch.as_tensor(usable, device=scores.device, dtype=torch.long)
            projected.append(scores.index_select(1, index).amax(dim=1))

        return torch.stack(projected, dim=1)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_audioset_labels(root: Path) -> List[LabelInfo]:
    path = root / "datasets" / "AudioSet_meta" / "class_labels_indices.csv"
    labels: List[LabelInfo] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            idx = int(row["index"])
            labels.append(LabelInfo(native_id=row["mid"],
                          name=row["display_name"],
                          evaluation_index=idx,
                          model_mid=row["mid"],
                          model_index=idx,
                          mapping_relation="exact_mid"))
    labels.sort(key=lambda item: int(item.model_index or 0))
    
    return labels


def _label_maps(labels: Sequence[LabelInfo]) -> tuple[Dict[str, LabelInfo], Dict[str, int]]:
    by_id = {label.native_id: label for label in labels}
    model_mid_to_index = {label.model_mid: int(label.model_index)
                          for label in labels
                          if label.model_mid is not None and label.model_index is not None}
    
    return by_id, model_mid_to_index


def _multi_hot(indices: Iterable[int], size: int) -> torch.Tensor:
    target = torch.zeros(size, dtype=torch.float32)
    for idx in set(indices):
        if 0 <= idx < size:
            target[idx] = 1.0
    
    return target


def _top10_from_counts(counts: Counter[str],
                       labels_by_id: Dict[str, LabelInfo],
                       total_samples: int,
                       *,
                       tie_order: Dict[str, int]) -> List[TopClassInfo]:
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], tie_order.get(kv[0], math.inf), kv[0]))[:10]
    result: List[TopClassInfo] = []
    
    for rank, (native_id, count) in enumerate(ordered, start=1):
        label = labels_by_id.get(native_id, LabelInfo(native_id=native_id, name=native_id))
        mapped = (label.model_mid is not None and label.model_index is not None) or bool(label.aggregate_model_indices)
        mapped_indices = label.aggregate_model_indices
        if mapped_indices is None and label.model_index is not None:
            mapped_indices = [int(label.model_index)]
        result.append(TopClassInfo(rank=rank,
                                   native_class_id=native_id,
                                   evaluation_index=int(label.evaluation_index) if label.evaluation_index is not None else None,
                                   native_class_name=label.name,
                                   positive_sample_count=int(count),
                                   prevalence_fraction=float(count / total_samples) if total_samples else 0.0,
                                   mapped_model_output_class_id=label.model_mid if mapped else None,
                                   mapped_model_output_index=int(label.model_index) if label.model_index is not None else None,
                                   mapped_model_output_indices=mapped_indices if mapped else None,
                                   mapped_class_name=label.name if mapped and label.model_index is not None else (
                                       "max(" + ", ".join(label.aggregate_model_labels or []) + ")" if mapped else None
                                       ),
                                   mapping_relation=label.mapping_relation if mapped else "none",
                                   status="evaluable" if mapped else "not_applicable",
                                   reason=None if mapped else "no exact mapping to the model AudioSet output space"))
    
    return result


def _path_exists(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _resolve_audio_path(default_root: Optional[Path], 
                        audio_root: Optional[Path], 
                        basename: str, 
                        fallback: Optional[str] = None) -> Path:
    root = audio_root or default_root
    if root is not None:
        return root / basename
    if fallback:
        return Path(fallback)
    
    return Path(basename)


def _filter_existing(samples: List[EvaluationSample]) -> tuple[List[EvaluationSample], List[Dict[str, Any]]]:
    kept: List[EvaluationSample] = []
    skipped: List[Dict[str, Any]] = []
    
    for sample in samples:
        if _path_exists(sample.audio_path):
            kept.append(sample)
        else:
            skipped.append({"clip_id": sample.clip_id,
                            "reason": "missing_audio",
                            "audio_path": str(sample.audio_path)})
    
    return kept, skipped


def _build_audioset(root: Path, audio_root: Optional[Path]) -> DatasetSpec:
    labels = _read_audioset_labels(root)
    labels_by_id, mid_to_index = _label_maps(labels)
    metadata_path = root / "datasets" / "AudioSet_meta" / "eval_segments.csv"
    default_audio_root = root / "datasets" / "AudioSet_data" / "eval_segments"
    samples: List[EvaluationSample] = []
    counts: Counter[str] = Counter()
    
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        
        for row in csv.reader(rows, skipinitialspace=True):
            if len(row) < 4 or row[0] == "YTID":
                continue
            ytid, start, end, labels_text = row[:4]
            native_ids = sorted(set(label.strip() for label in labels_text.strip('"').split(",") if label.strip()))
            indices = [mid_to_index[mid] for mid in native_ids if mid in mid_to_index]
            counts.update(native_ids)
            samples.append(EvaluationSample(clip_id=ytid,
                                            audio_path=_resolve_audio_path(default_audio_root, audio_root, f"Y{ytid}.wav"),
                                            target_indices=indices,
                                            native_label_ids=native_ids,
                                            metadata={"ytid": ytid, 
                                                      "start_seconds": float(start), 
                                                      "end_seconds": float(end), 
                                                      "dataset": "audioset"}))
    samples, skipped = _filter_existing(samples)
    tie_order = {label.native_id: i for i, label in enumerate(labels)}
    top10 = _top10_from_counts(counts, labels_by_id, len(samples) + len(skipped), tie_order=tie_order)
    
    return DatasetSpec(name="audioset",
                       display_name="AudioSet Standard",
                       metadata_path=metadata_path,
                       default_audio_root=default_audio_root,
                       split="eval",
                       native_label_space="AudioSet MID",
                       evaluation_label_space="AudioSet 527 MID model-output space",
                       labels=labels,
                       native_class_count=len(labels),
                       samples=samples,
                       top10=top10,
                       class_frequencies=dict(counts),
                       skipped_samples=skipped,
                       compatibility=_audioset_compatibility("direct AudioSet MID alignment"),
                       mapping_coverage=_coverage(len(labels), len(labels), 0, "direct exact MID"))


def _build_audioset_r(root: Path, audio_root: Optional[Path]) -> DatasetSpec:
    labels = _read_audioset_labels(root)
    labels_by_id, mid_to_index = _label_maps(labels)
    metadata_path = root / "datasets" / "AudioSet-R_meta" / "AudioSet-R_eval.json"
    default_audio_root = root / "datasets" / "AudioSet_data" / "eval_segments"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples: List[EvaluationSample] = []
    counts: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()
    
    for item in raw.get("data", []):
        wav = item.get("wav", "")
        basename = Path(wav).name
        clip_id = Path(basename).stem.removeprefix("Y")
        native_ids = sorted(set(label.strip() for label in item.get("labels", "").split(",") if label.strip()))
        indices = []
        for mid in native_ids:
            if mid in mid_to_index:
                indices.append(mid_to_index[mid])
            else:
                unknown_labels[mid] += 1
        counts.update(native_ids)
        samples.append(EvaluationSample(clip_id=clip_id,
                                        audio_path=_resolve_audio_path(default_audio_root, audio_root, basename, wav),
                                        target_indices=indices,
                                        native_label_ids=native_ids,
                                        metadata={"source_wav": wav, "dataset": "audioset_r"}))
    samples, skipped = _filter_existing(samples)
    tie_order = {label.native_id: i for i, label in enumerate(labels)}
    top10 = _top10_from_counts(counts, labels_by_id, len(samples) + len(skipped), tie_order=tie_order)
    
    return DatasetSpec(name="audioset_r",
                       display_name="AudioSet-R",
                       metadata_path=metadata_path,
                       default_audio_root=default_audio_root,
                       split="eval",
                       native_label_space="AudioSet-R annotation MIDs",
                       evaluation_label_space="AudioSet 527 MID model-output space, filtered to known AudioSet classes",
                       labels=labels,
                       native_class_count=len(counts),
                       samples=samples,
                       top10=top10,
                       class_frequencies=dict(counts),
                       skipped_samples=skipped,
                       compatibility=_audioset_compatibility("AudioSet-R MIDs are filtered to exact 527-class AudioSet MID matches"),
                       mapping_coverage={**_coverage(len(counts), 
                                                     len([mid for mid in counts if mid in mid_to_index]), 
                                                     len(unknown_labels), 
                                                     "exact MID intersection"),
                                         "unmapped_native_label_ids": sorted(unknown_labels)})


def _load_fsd50k_projection_mapping(root: Path) -> Dict[str, Any]:
    path = root / "evaluation" / "mappings" / "fsd50k_to_audioset_projection.json"
    
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_fsd50k(root: Path, audio_root: Optional[Path]) -> DatasetSpec:
    audioset_labels = _read_audioset_labels(root)
    _, mid_to_index = _label_maps(audioset_labels)
    projection_mapping = _load_fsd50k_projection_mapping(root)
    projection_by_label = projection_mapping["mappings"]
    metadata_path = root / "datasets" / "FSD50K_meta" / "FSD50K.ground_truth" / "eval.csv"
    vocab_path = root / "datasets" / "FSD50K_meta" / "FSD50K.ground_truth" / "vocabulary.csv"
    default_audio_root = root / "datasets" / "FSD50K_data" / "FSD50K.eval_audio"
    labels: List[LabelInfo] = []

    with vocab_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            native_index = int(row[0])
            label_name, mid = row[1], row[2]
            if mid in mid_to_index:
                labels.append(LabelInfo(native_id=label_name,
                                        name=label_name,
                                        evaluation_index=native_index,
                                        model_mid=mid,
                                        model_index=mid_to_index[mid],
                                        mapping_relation="exact_mid"))
            elif label_name in projection_by_label:
                mapping = projection_by_label[label_name]
                targets = mapping["audioset_targets"]
                labels.append(LabelInfo(native_id=label_name,
                                        name=label_name,
                                        evaluation_index=native_index,
                                        model_mid=mid,
                                        aggregate_model_indices=[int(item["index"]) for item in targets],
                                        aggregate_model_labels=[item["label"] for item in targets],
                                        mapping_relation=mapping["relation"]))
            else:
                labels.append(LabelInfo(native_id=label_name,
                                        name=label_name,
                                        evaluation_index=native_index,
                                        model_mid=mid,
                                        mapping_relation="unmapped"))

    labels.sort(key=lambda label: int(label.evaluation_index) if label.evaluation_index is not None else math.inf)
    evaluation_indices = [label.evaluation_index for label in labels]
    if evaluation_indices != list(range(len(labels))):
        raise ValueError("FSD50K vocabulary indices must form a contiguous 0-based evaluation space")

    labels_by_id, _ = _label_maps(labels)
    label_to_info = {label.native_id: label for label in labels}
    samples: List[EvaluationSample] = []
    counts: Counter[str] = Counter()
    unmapped_labels: Counter[str] = Counter()

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_names = sorted(set(label.strip() for label in row["labels"].split(",") if label.strip()))
            indices: List[int] = []
            for label_name in label_names:
                info = label_to_info.get(label_name)
                if info is not None and info.evaluation_index is not None and (
                    info.model_index is not None or info.aggregate_model_indices
                ):
                    indices.append(int(info.evaluation_index))
                else:
                    unmapped_labels[label_name] += 1
            counts.update(label_names)
            fname = row["fname"]
            samples.append(EvaluationSample(clip_id=fname,
                                            audio_path=_resolve_audio_path(default_audio_root, audio_root, f"{fname}.wav"),
                                            target_indices=indices,
                                            native_label_ids=label_names,
                                            metadata={"fname": fname, "dataset": "fsd50k"}))

    samples, skipped = _filter_existing(samples)
    tie_order = {label.native_id: i for i, label in enumerate(labels)}
    top10 = _top10_from_counts(counts, labels_by_id, len(samples) + len(skipped), tie_order=tie_order)

    prediction_projection: List[List[int]] = []
    for label in labels:
        if label.model_index is not None:
            prediction_projection.append([int(label.model_index)])
        elif label.aggregate_model_indices:
            prediction_projection.append([int(index) for index in label.aggregate_model_indices])
        else:
            prediction_projection.append([])

    return DatasetSpec(name="fsd50k",
                       display_name="FSD50K",
                       metadata_path=metadata_path,
                       default_audio_root=default_audio_root,
                       split="eval",
                       native_label_space="FSD50K 200-class vocabulary",
                       evaluation_label_space="FSD50K native 200-class space projected from canonical AudioSet model outputs",
                       labels=labels,
                       native_class_count=len(labels),
                       samples=samples,
                       top10=top10,
                       class_frequencies=dict(counts),
                       skipped_samples=skipped,
                       compatibility=_fsd50k_compatibility(),
                       mapping_coverage={**_coverage(len(labels),
                                                     len([label for label in labels if label.model_index is not None or label.aggregate_model_indices]),
                                                     len(unmapped_labels),
                                                     "FSD50K native-space projection from canonical AudioSet outputs"),
                                         "exact_mappings": len([label for label in labels if label.model_index is not None]),
                                         "projection_mappings": len([label for label in labels if label.aggregate_model_indices]),
                                         "unmapped_native_label_ids": sorted(unmapped_labels),
                                         "projection_mapping": projection_mapping},
                       prediction_projection=prediction_projection,
                       supports_audioset_hierarchy=False)


def _fsd50k_compatibility() -> Dict[str, Any]:
    return {"model_output_compatibility": "canonical AudioSet 527 outputs projected into the native FSD50K 200-class space",
            "mapping_required": "192 exact MID projections plus 8 reviewed max-aggregation projections",
            "native_hierarchy_available": False,
            "ontology_hierarchy_source": None,
            "valid_metrics": ["class_wise_ap",
                              "mAP",
                              "lwlrap",
                              "micro_averaged_accuracy",
                              "micro_f1",
                              "macro_f1",
                              "TOP-10 class-specific F1 in native FSD50K space"],
            "invalid_or_not_applicable_metrics": [
                "AudioSet OmAP because the evaluation target space is FSD50K 200-class",
                "AudioSet HLP metrics because the 527-node AudioSet hierarchy does not define the projected FSD50K space",
            ]}


def _audioset_compatibility(alignment: str) -> Dict[str, Any]:
    return {"model_output_compatibility": alignment,
            "mapping_required": alignment,
            "native_hierarchy_available": True,
            "ontology_hierarchy_source": "training/audioset_graph_distance.py and training/HLP_lookup_table.pt",
            "valid_metrics": ["class_wise_ap",
                              "mAP",
                              "lwlrap",
                              "micro_averaged_accuracy",
                              "micro_f1",
                              "macro_f1",
                              "TOP-10 class-specific F1 for exact mapped and projected aggregate classes",
                              "OmAP when using full AudioSet 527 target space",
                              "AudioSet HLP post-processed metrics when asset shape matches"],
                              "invalid_or_not_applicable_metrics": ["native non-AudioSet hierarchy metrics without an exact local hierarchy implementation"]}


def _coverage(total: int, mapped: int, unmapped: int, mechanism: str) -> Dict[str, Any]:
    return {"total_dataset_classes": int(total),
            "mapped_classes": int(mapped),
            "unmapped_classes": int(unmapped),
            "exact_mappings": int(mapped),
            "broader_mappings": 0,
            "narrower_mappings": 0,
            "ambiguous_mappings": 0,
            "mapping_mechanism": mechanism}


def build_dataset_spec(dataset_name: str,
                       *,
                       datasets_root: Path | str = "datasets",
                       audio_root: Optional[Path | str] = None) -> DatasetSpec:
    root = repo_root()
    
    if Path(datasets_root) != Path("datasets"):
        root = Path(datasets_root).resolve().parent
    audio_path = Path(audio_root).expanduser().resolve() if audio_root else None
    name = dataset_name.lower()
    
    if name == "audioset":
        return _build_audioset(root, audio_path)
    if name == "audioset_r":
        return _build_audioset_r(root, audio_path)
    if name == "fsd50k":
        return _build_fsd50k(root, audio_path)
    raise ValueError(f"unsupported dataset {dataset_name!r}; choose from {', '.join(DATASET_NAMES)}")


class EvaluationAudioDataset(Dataset):
    """Decode mono waveforms and return aligned target vectors plus metadata."""

    def __init__(self, spec: DatasetSpec, *, sample_rate: int, duration_seconds: float = 10.0):
        self.spec = spec
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.target_samples = max(1, int(round(sample_rate * duration_seconds)))

    def __len__(self) -> int:
        return len(self.spec.samples)

    @staticmethod
    def _decode_audio(path: Path) -> tuple[torch.Tensor, int, str]:
        errors = []
        try:
            waveform, sr = torchaudio.load(str(path))
            decoder = "torchaudio"
        except Exception as exc:
            errors.append(f"torchaudio: {type(exc).__name__}: {exc}")
        else:
            if waveform.ndim != 2:
                raise ValueError(f"expected torchaudio waveform with shape (channels, samples), got {tuple(waveform.shape)}")
            if waveform.numel() <= 0 or waveform.shape[-1] <= 0:
                errors.append("torchaudio: decoded zero audio frames")
            else:
                return waveform, int(sr), decoder

        if sf is not None:
            try:
                audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
                waveform = torch.from_numpy(audio.T.copy())
                decoder = "soundfile"
            except Exception as exc:
                errors.append(f"soundfile: {type(exc).__name__}: {exc}")
            else:
                if waveform.numel() <= 0 or waveform.shape[-1] <= 0:
                    errors.append("soundfile: decoded zero audio frames")
                else:
                    return waveform, int(sr), decoder
        else:
            errors.append("soundfile: module is not installed")

        raise RuntimeError("; ".join(errors))

    def _failure_item(self, sample: EvaluationSample, exc: Exception) -> Dict[str, Any]:
        return {
            "skip_sample": True,
            "metadata": {
                **sample.metadata,
                "clip_id": sample.clip_id,
                "audio_path": str(sample.audio_path),
                "native_label_ids": sample.native_label_ids,
                "reason": "audio_decode_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.spec.samples[index]
        try:
            waveform, sr, decoder = self._decode_audio(sample.audio_path)
        except Exception as exc:
            return self._failure_item(sample, exc)
        waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.numel() < self.target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_samples - waveform.numel()))
        elif waveform.numel() > self.target_samples:
            waveform = waveform[:self.target_samples]
        target = _multi_hot(sample.target_indices, self.spec.class_count)
        metadata = {**sample.metadata,
                    "clip_id": sample.clip_id,
                    "audio_path": str(sample.audio_path),
                    "native_label_ids": sample.native_label_ids,
                    "effective_sample_rate": self.sample_rate,
                    "audio_decoder": decoder}
        
        return {"waveform": waveform, "target": target, "metadata": metadata}


def collate_evaluation_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in batch if not item.get("skip_sample")]
    failed = [item["metadata"] for item in batch if item.get("skip_sample")]
    if not valid:
        return {"skip_batch": True, "metadata": [], "failed_metadata": failed}
    
    return {"waveform": torch.stack([item["waveform"] for item in valid], dim=0),
            "target": torch.stack([item["target"] for item in valid], dim=0),
            "metadata": [item["metadata"] for item in valid],
            "failed_metadata": failed}


class EvaluationDataModule(pl.LightningDataModule if pl else object):
    """Lightning data module for one parsed evaluation dataset."""

    def __init__(self,
                 spec: DatasetSpec,
                 *,
                 sample_rate: int,
                 batch_size: int,
                 num_workers: int = 0,
                 duration_seconds: float = 10.0,
                 pin_memory: bool = False):
        super().__init__()
        
        self.spec = spec
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.duration_seconds = duration_seconds
        self.pin_memory = pin_memory
        self.dataset: Optional[EvaluationAudioDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        del stage
        self.dataset = EvaluationAudioDataset(self.spec, sample_rate=self.sample_rate, duration_seconds=self.duration_seconds)

    def test_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup("test")
        
        assert self.dataset is not None
        
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=collate_evaluation_batch,
                          persistent_workers=self.num_workers > 0)
