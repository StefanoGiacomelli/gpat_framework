"""
PSLA (Pretraining, Sampling, Labeling, Aggregation)
===================================================
PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation.

Original repository: https://github.com/YuanGongND/psla

This is a standalone implementation of PSLA (EfficientNet-B2 + Multi-Head Attention)
for AudioSet tagging.
"""

import math
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from efficientnet_pytorch import EfficientNet
import torchaudio


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 16000
CLASSES_NUM = 527
EMBED_DIM = 1408  # EfficientNet-B2 output dimension

# Mel spectrogram defaults
_NUM_MEL_BINS = 128
_TARGET_LENGTH = 1024  # frames
_FRAME_SHIFT = 10  # ms

# Normalization stats (AudioSet)
_NORM_MEAN = -4.2677
_NORM_STD = 4.5689


# =============================================================================
# Helper Functions
# =============================================================================

def init_layer(layer: nn.Module) -> None:
    """Initialize a Conv2d or Linear layer."""
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()
    else:
        return

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


# =============================================================================\n# Building Blocks\n# =============================================================================

class Attention(nn.Module):
    """Single-head attention pooling module."""
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        att_activation: str = 'sigmoid',
        cla_activation: str = 'sigmoid'
    ):
        super().__init__()
        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(n_in, n_out, kernel_size=1, bias=True)
        self.cla = nn.Conv2d(n_in, n_out, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        init_layer(self.att)
        init_layer(self.cla)

    def _activate(self, x: Tensor, activation: str) -> Tensor:
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, channels, time_steps, 1)
        Returns:
            output: (batch, classes)
            norm_att: normalized attention weights
        """
        att = self.att(x)
        att = self._activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self._activate(cla, self.cla_activation)

        att = att[:, :, :, 0]  # (batch, classes, time_steps)
        cla = cla[:, :, :, 0]  # (batch, classes, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2, keepdim=True)
        output = torch.sum(norm_att * cla, dim=2)

        return output, norm_att


class MHeadAttention(nn.Module):
    """Multi-head attention pooling module."""
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        att_activation: str = 'sigmoid',
        cla_activation: str = 'sigmoid',
        head_num: int = 4
    ):
        super().__init__()
        self.head_num = head_num
        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.ModuleList([
            nn.Conv2d(n_in, n_out, kernel_size=1, bias=True)
            for _ in range(head_num)
        ])
        self.cla = nn.ModuleList([
            nn.Conv2d(n_in, n_out, kernel_size=1, bias=True)
            for _ in range(head_num)
        ])

        self.head_weight = nn.Parameter(
            torch.tensor([1.0 / head_num] * head_num)
        )

    def _activate(self, x: Tensor, activation: str) -> Tensor:
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        """
        Args:
            x: (batch, channels, time_steps, 1)
        Returns:
            output: (batch, classes)
            None: placeholder for compatibility
        """
        x_out = []
        for i in range(self.head_num):
            att = self.att[i](x)
            att = self._activate(att, self.att_activation)

            cla = self.cla[i](x)
            cla = self._activate(cla, self.cla_activation)

            att = att[:, :, :, 0]  # (batch, classes, time_steps)
            cla = cla[:, :, :, 0]  # (batch, classes, time_steps)

            epsilon = 1e-7
            att = torch.clamp(att, epsilon, 1. - epsilon)

            norm_att = att / torch.sum(att, dim=2, keepdim=True)
            x_out.append(torch.sum(norm_att * cla, dim=2) * self.head_weight[i])

        output = torch.stack(x_out, dim=0).sum(dim=0)

        return output, None


class MeanPooling(nn.Module):
    """Mean pooling module (no attention)."""
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        att_activation: str = 'sigmoid',
        cla_activation: str = 'sigmoid'
    ):
        super().__init__()
        self.cla_activation = cla_activation
        self.cla = nn.Conv2d(n_in, n_out, kernel_size=1, bias=True)
        init_layer(self.cla)

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        """
        Args:
            x: (batch, channels, time_steps, 1)
        Returns:
            output: (batch, classes)
            None: placeholder for compatibility
        """
        cla = self.cla(x)
        cla = torch.sigmoid(cla)
        cla = cla[:, :, :, 0]  # (batch, classes, time_steps)
        output = torch.mean(cla, dim=2)

        return output, None


# =============================================================================
# Main Model
# =============================================================================

class EffNetAttention(nn.Module):
    """
    PSLA model: EfficientNet backbone with multi-head attention pooling.
    
    Args:
        sample_rate: Audio sample rate (default: 16000)
        num_mel_bins: Number of mel filterbank bins (default: 128)
        target_length: Target number of frames (default: 1024)
        label_dim: Number of output classes (default: 527 for AudioSet)
        b: EfficientNet variant (0-7, default: 2 for B2)
        pretrain: Use ImageNet pretrained backbone (default: False)
        head_num: Number of attention heads (0=mean, 1=single, >1=multi)
    """
    
    # EfficientNet embedding dimensions for each variant
    MIDDIM = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        num_mel_bins: int = _NUM_MEL_BINS,
        target_length: int = _TARGET_LENGTH,
        label_dim: int = CLASSES_NUM,
        b: int = 2,
        pretrain: bool = False,
        head_num: int = 4
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.label_dim = label_dim
        self.b = b
        self.head_num = head_num
        
        # EfficientNet backbone
        if pretrain:
            self.effnet = EfficientNet.from_pretrained(
                f'efficientnet-b{b}', in_channels=1
            )
        else:
            self.effnet = EfficientNet.from_name(
                f'efficientnet-b{b}', in_channels=1
            )
        
        # Remove original classification layer
        self.effnet._fc = nn.Identity()
        
        # Attention pooling
        middim = self.MIDDIM[b]
        if head_num > 1:
            self.attention = MHeadAttention(
                middim, label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid',
                head_num=head_num
            )
        elif head_num == 1:
            self.attention = Attention(
                middim, label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid'
            )
        else:
            self.attention = MeanPooling(
                middim, label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid'
            )
        
        self.avgpool = nn.AvgPool2d((4, 1))
    
    def _wav2fbank(self, waveform: Tensor) -> Tensor:
        """
        Convert waveform to mel filterbank features.
        
        Args:
            waveform: (batch, samples) at self.sample_rate
        Returns:
            fbank: (batch, target_length, num_mel_bins)
        """
        batch_size = waveform.shape[0]
        fbank_list = []
        
        for i in range(batch_size):
            wav = waveform[i:i+1]  # (1, samples)
            
            # Mean normalization
            wav = wav - wav.mean()
            
            # Compute mel filterbank (Kaldi-style)
            fbank = torchaudio.compliance.kaldi.fbank(
                wav,
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.num_mel_bins,
                dither=0.0,
                frame_shift=_FRAME_SHIFT
            )  # (time, mel_bins)
            
            # Pad or truncate to target_length
            n_frames = fbank.shape[0]
            if n_frames < self.target_length:
                pad_size = self.target_length - n_frames
                fbank = F.pad(fbank, (0, 0, 0, pad_size))
            else:
                fbank = fbank[:self.target_length, :]
            
            # Normalize
            fbank = (fbank - _NORM_MEAN) / _NORM_STD
            
            fbank_list.append(fbank)
        
        return torch.stack(fbank_list, dim=0)  # (batch, target_length, mel_bins)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass from raw waveform to class probabilities.
        
        Args:
            x: (batch, samples) waveform at sample_rate
        Returns:
            (batch, label_dim) class probabilities
        """
        # Convert waveform to mel filterbank
        fbank = self._wav2fbank(x)  # (batch, time, mel)
        
        # Reshape for EfficientNet: (batch, 1, mel, time)
        fbank = fbank.unsqueeze(1)  # (batch, 1, time, mel)
        fbank = fbank.transpose(2, 3)  # (batch, 1, mel, time)
        
        # EfficientNet feature extraction
        features = self.effnet.extract_features(fbank)  # (batch, middim, h, w)
        
        # Pool and reshape for attention
        features = self.avgpool(features)  # (batch, middim, h', 1) typically
        features = features.transpose(2, 3)  # (batch, middim, 1, h')
        
        # Attention pooling
        out, _ = self.attention(features)  # (batch, label_dim)
        
        return out
    
    def forward_fbank(self, fbank: Tensor) -> Tensor:
        """
        Forward pass from mel filterbank features.
        
        Args:
            fbank: (batch, time, mel_bins) normalized mel filterbank
        Returns:
            (batch, label_dim) class probabilities
        """
        # Reshape for EfficientNet: (batch, 1, mel, time)
        x = fbank.unsqueeze(1)  # (batch, 1, time, mel)
        x = x.transpose(2, 3)  # (batch, 1, mel, time)
        
        # EfficientNet feature extraction
        features = self.effnet.extract_features(x)  # (batch, middim, h, w)
        
        # Pool and reshape for attention
        features = self.avgpool(features)
        features = features.transpose(2, 3)
        
        # Attention pooling
        out, _ = self.attention(features)
        
        return out
    
    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            x: (batch, samples) waveform at sample_rate
        Returns:
            (batch, embed_dim) embedding vector
        """
        # Convert waveform to mel filterbank
        fbank = self._wav2fbank(x)  # (batch, time, mel)
        
        # Reshape for EfficientNet
        fbank = fbank.unsqueeze(1).transpose(2, 3)  # (batch, 1, mel, time)
        
        # EfficientNet feature extraction
        features = self.effnet.extract_features(fbank)  # (batch, middim, h, w)
        
        # Global average pooling
        embedding = features.mean(dim=[2, 3])  # (batch, middim)
        
        return embedding
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained PSLA checkpoint.
        
        Handles 'module.' prefix from DataParallel training.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Strip 'module.' prefix from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # Remove 'module.' prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        self.load_state_dict(new_state_dict, strict=False)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import csv
    import numpy as np
    import soundfile as sf
    
    # Paths
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "as_mdl_0_wa.pth")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("PSLA (EfficientNet-B2 + Multi-Head Attention) Demo")
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
    model = EffNetAttention(
        sample_rate=SAMPLE_RATE,
        label_dim=CLASSES_NUM,
        b=2,
        pretrain=False,
        head_num=4
    )
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
