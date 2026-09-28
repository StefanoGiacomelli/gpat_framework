"""
AudioSet-ConvNeXt (ConvNeXt-Tiny)
===========================================
Adapting a ConvNeXt model to audio classification on AudioSet.

Original repository: https://github.com/topel/audioset-convnext-inf

This is a standalone vendorized implementation for AudioSet tagging.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 32000
CLASSES_NUM = 527
EMBED_DIM = 768


# =============================================================================
# Helper Functions
# =============================================================================

def _trunc_normal_(tensor: Tensor, mean: float, std: float, a: float, b: float) -> Tensor:
    """Truncated normal initialization (internal)."""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, 
                  a: float = -2.0, b: float = 2.0) -> Tensor:
    """Fill tensor with values from a truncated normal distribution."""
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True) -> Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        trunc_normal_(layer.weight, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def init_bn(bn: nn.Module) -> None:
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# =============================================================================
# Building Blocks
# =============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm(nn.Module):
    """LayerNorm supporting both channels_last and channels_first formats."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6, 
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    """ConvNeXt Block.
    
    DwConv -> Permute -> LayerNorm -> Linear -> GELU -> Linear -> Permute
    with LayerScale and DropPath (residual connection).
    """
    
    def __init__(self, dim: int, drop_path: float = 0.0, 
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


# =============================================================================
# Main Model
# =============================================================================

class ConvNeXt(nn.Module):
    """ConvNeXt-Tiny adapted for audio classification.
    
    A ConvNet for the 2020s, adapted for audio tagging on AudioSet.
    
    Args:
        sample_rate: Audio sample rate (default: 32000)
        window_size: STFT window size (default: 1024)
        hop_size: STFT hop size (default: 320)
        mel_bins: Number of mel frequency bins (default: 224)
        fmin: Minimum frequency for mel filterbank (default: 50)
        fmax: Maximum frequency for mel filterbank (default: 14000)
        num_classes: Number of output classes (default: 527)
        depths: Number of blocks at each stage (default: [3, 3, 9, 3])
        dims: Feature dimension at each stage (default: [96, 192, 384, 768])
        drop_path_rate: Stochastic depth rate (default: 0.0)
        layer_scale_init_value: Init value for Layer Scale (default: 1e-6)
    """
    
    def __init__(
        self,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 224,
        fmin: int = 50,
        fmax: int = 14000,
        num_classes: int = 527,
        depths: list = None,
        dims: list = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
        
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.embed_dim = dims[-1]
        
        # Audio preprocessing (torchlibrosa)
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window='hann',
            center=True,
            pad_mode='reflect',
            freeze_parameters=True,
        )
        
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        
        # Batch normalization for mel spectrogram
        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        
        # Stem: adapted for audio (1 channel input)
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        
        # Intermediate downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Feature stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Final normalization and classification head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_audioset = nn.Linear(dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: Tensor, return_frame_embeddings: bool = False) -> Tensor:
        """Extract features from spectrogram."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        if return_frame_embeddings:
            return x

        # Global pooling: mean over frequency, then max+mean over time
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for audio classification.
        
        Args:
            x: Input waveform of shape (batch, samples)
            
        Returns:
            Tensor of shape (batch, num_classes) with probabilities
        """
        # Audio preprocessing
        x = self.spectrogram_extractor(x)  # (B, 1, T, freq_bins)
        x = self.logmel_extractor(x)       # (B, 1, T, mel_bins)
        
        # Normalize mel spectrogram
        x = x.transpose(1, 3)  # (B, mel_bins, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, mel_bins)
        
        # Feature extraction
        x = self.forward_features(x)
        
        # Classification
        x = self.head_audioset(x)
        output = torch.sigmoid(x)
        
        return output

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Extract scene-level embedding.
        
        Args:
            x: Input waveform of shape (batch, samples)
            
        Returns:
            Tensor of shape (batch, 768) scene embedding
        """
        # Audio preprocessing
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        
        # Normalize mel spectrogram
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Feature extraction (returns normalized embedding)
        embedding = self.forward_features(x)
        
        return embedding

    def get_frame_embeddings(self, x: Tensor) -> Tensor:
        """
        Extract frame-level embeddings.
        
        Args:
            x: Input waveform of shape (batch, samples)
            
        Returns:
            Tensor of shape (batch, 768, T, F) frame-level embeddings
        """
        # Audio preprocessing
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        
        # Normalize mel spectrogram
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Feature extraction with frame embeddings
        frame_embeddings = self.forward_features(x, return_frame_embeddings=True)
        
        return frame_embeddings

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights
        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint from: {checkpoint_path}")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import os
    import csv
    import numpy as np
    import torchaudio
    import soundfile as sf
    
    # Paths
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "convnext_tiny_471mAP.pth")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("AudioSet-ConvNeXt (ConvNeXt-Tiny) Demo")
    print("=" * 60)

    # 1. Device selection
    print("\n1. Selecting device...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"   Using device: {device}")

    # 2. Loading AudioSet labels
    print("\n2. Loading AudioSet labels...")
    labels = {}
    with open(LABELS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['index'])] = row['display_name']
    print(f"   Loaded {len(labels)} labels")

    # 3. Loading audio
    print("\n3. Loading audio...")
    audio, sr = sf.read(AUDIO_PATH)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # stereo to mono
    waveform = torch.from_numpy(audio).float()
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    waveform = waveform.unsqueeze(0).to(device)  # (1, samples)
    print(f"   Duration: {waveform.shape[1] / SAMPLE_RATE:.2f}s")

    # 4. Creating model
    print("\n4. Creating model...")
    model = ConvNeXt()
    model.load_pretrained(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 5. Running inference
    print("\n5. Running inference...")
    with torch.no_grad():
        probs = model(waveform)
    print(f"   Output shape: {probs.shape}")

    # 6. Top-10 predictions
    print("\n6. Top-10 predictions:")
    probs_np = probs[0].cpu().numpy()
    top_indices = probs_np.argsort()[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1:2d}. {labels[idx]:<40} {probs_np[idx]:.4f}")

    # 7. Embedding extraction
    print("\n7. Extracting embedding...")
    with torch.no_grad():
        embedding = model.get_embedding(waveform)
    print(f"   Embedding shape: {embedding.shape}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
