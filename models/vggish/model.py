"""
VGGish (Audio Classification with VGG-style CNN)
=================================================
CNN Architectures for Large-Scale Audio Classification.

Original repository: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
PyTorch port: https://github.com/w-hc/torch_audioset

This is a standalone implementation of VGGish for AudioSet tagging.
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio.transforms as T


# ============================================
# CONSTANTS
# ============================================
SAMPLE_RATE = 16000
CLASSES_NUM = 527
EMBED_DIM = 128

# Mel spectrogram params
NUM_MEL_BINS = 64
NUM_FRAMES = 96  # 0.96s window
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
# Note: Original VGGish spec says 0.01, but torch_audioset checkpoint
# was trained with 0.001 (CommonParams.LOG_OFFSET), so we use 0.001 for compatibility
LOG_OFFSET = 0.001

# Derived params
WINDOW_LENGTH_SAMPLES = int(round(SAMPLE_RATE * STFT_WINDOW_LENGTH_SECONDS))  # 400
HOP_LENGTH_SAMPLES = int(round(SAMPLE_RATE * STFT_HOP_LENGTH_SECONDS))  # 160
FFT_LENGTH = 512


# ============================================
# Preprocessing
# ============================================
class VGGishLogMelSpectrogram(nn.Module):
    """Log Mel Spectrogram following VGGish preprocessing.
    
    Note: The order of operations is important:
    1. STFT (power spectrogram)
    2. sqrt (convert power to magnitude)  
    3. Mel filterbank
    4. log compression
    
    This matches the original TensorFlow VGGish implementation.
    """
    
    def __init__(self):
        super().__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=FFT_LENGTH,
            win_length=WINDOW_LENGTH_SAMPLES,
            hop_length=HOP_LENGTH_SAMPLES,
            power=2.0
        )
        self.mel_scale = T.MelScale(
            n_mels=NUM_MEL_BINS,
            sample_rate=SAMPLE_RATE,
            n_stft=FFT_LENGTH // 2 + 1,
            f_min=MEL_MIN_HZ,
            f_max=MEL_MAX_HZ
        )
    
    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: (batch, samples) or (samples,)
        Returns:
            log_mel: (batch, 1, num_frames, num_mel_bins)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Step 1: Power spectrogram
        spec = self.spectrogram(waveform)
        
        # Step 2: Convert power to magnitude (sqrt)
        spec = spec ** 0.5
        
        # Step 3: Apply mel filterbank
        mel = self.mel_scale(spec)
        
        # Step 4: Log compression
        log_mel = torch.log(mel + LOG_OFFSET)
        
        # Transpose to (batch, 1, time, freq) for CNN input
        log_mel = log_mel.unsqueeze(1).transpose(2, 3)
        
        return log_mel


class AudioPreprocessor(nn.Module):
    """Preprocess waveform to VGGish input patches."""
    
    def __init__(self):
        super().__init__()
        self.log_mel = VGGishLogMelSpectrogram()
        self.resample = None
        self._cached_sample_rate = None
    
    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """
        Args:
            waveform: (batch, samples) or (samples,)
            sample_rate: input sample rate
        Returns:
            patches: (num_patches, 1, 96, 64) mel spectrogram patches
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Mono
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            if self._cached_sample_rate != sample_rate:
                self.resample = T.Resample(sample_rate, SAMPLE_RATE).to(waveform.device)
                self._cached_sample_rate = sample_rate
            waveform = self.resample(waveform)
        
        # Compute log mel spectrogram
        log_mel = self.log_mel(waveform)  # (batch, 1, time, 64)
        
        # Split into non-overlapping patches of 96 frames
        batch_size = log_mel.shape[0]
        time_frames = log_mel.shape[2]
        num_patches = time_frames // NUM_FRAMES
        
        if num_patches == 0:
            # Pad if too short
            pad_frames = NUM_FRAMES - time_frames
            log_mel = F.pad(log_mel, (0, 0, 0, pad_frames))
            num_patches = 1
        
        # Reshape to patches
        num_frames_to_use = num_patches * NUM_FRAMES
        log_mel = log_mel[:, :, :num_frames_to_use, :]
        patches = log_mel.reshape(batch_size * num_patches, 1, NUM_FRAMES, NUM_MEL_BINS)
        
        return patches


# ============================================
# Building Blocks
# ============================================
class VGGishBackbone(nn.Module):
    """VGG-style CNN backbone for audio."""
    
    def __init__(self):
        super().__init__()
        self.features = self._make_layers()
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, EMBED_DIM),
            nn.ReLU(True),
        )
    
    @staticmethod
    def _make_layers():
        layer_config = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]
        in_channels = 1
        layers = []
        for curr in layer_config:
            if curr == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, curr, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = curr
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 1, 96, 64) mel spectrogram
        Returns:
            embedding: (batch, 128)
        """
        x = self.features(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = x.reshape(x.shape[0], -1)
        x = self.embeddings(x)
        return x


class VGGishClassifier(nn.Module):
    """Classification head for VGGish."""
    
    def __init__(self, num_hidden_units: int = 100, num_classes: int = CLASSES_NUM):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, num_hidden_units),
            nn.ReLU(True),
            nn.Linear(num_hidden_units, num_classes),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


# ============================================
# Main Model
# ============================================
class VGGish(nn.Module):
    """
    VGGish wrapper with standard API for audio classification.
    
    Args:
        sample_rate: Expected input sample rate (default: 16000)
    
    Standard API:
        - forward(waveform) -> probs (527 AudioSet classes)
        - forward_with_embedding(waveform) -> (probs, embedding)
        - get_embedding(waveform) -> embedding (128 dim)
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor()
        self.backbone = VGGishBackbone()
        self.classifier = VGGishClassifier()
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained VGGish checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Map keys from original format to our format
        # Original: features.X, embeddings.X, classifier.X
        # Ours: backbone.features.X, backbone.embeddings.X, classifier.classifier.X
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.'):
                new_state_dict[f'backbone.{k}'] = v
            elif k.startswith('embeddings.'):
                new_state_dict[f'backbone.{k}'] = v
            elif k.startswith('classifier.'):
                new_state_dict[f'classifier.{k}'] = v
            else:
                new_state_dict[k] = v
        
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def forward(self, waveform: Tensor) -> Tensor:
        """
        Forward pass returning AudioSet class probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at any sample rate
        Returns:
            probs: (batch, 527) AudioSet class probabilities
        """
        probs, _ = self.forward_with_embedding(waveform)
        return probs
    
    def forward_with_embedding(self, waveform: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at any sample rate
        Returns:
            probs: (batch, 527) AudioSet class probabilities
            embedding: (batch, 128) feature embedding
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        
        # Preprocess to patches
        patches = self.preprocessor(waveform, self.sample_rate)
        
        # Get embeddings for each patch
        embeddings = self.backbone(patches)
        
        # Get logits for each patch
        logits = self.classifier(embeddings)
        
        # Average over patches (for multi-patch inputs)
        batch_size = waveform.shape[0]
        num_patches = patches.shape[0] // batch_size
        
        if num_patches > 1:
            embeddings = embeddings.reshape(batch_size, num_patches, -1).mean(dim=1)
            logits = logits.reshape(batch_size, num_patches, -1).mean(dim=1)
        
        # Sigmoid for multi-label classification
        probs = torch.sigmoid(logits)
        
        return probs, embeddings
    
    def get_embedding(self, waveform: Tensor) -> Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at any sample rate
        Returns:
            embedding: (batch, 128) feature embedding
        """
        _, embedding = self.forward_with_embedding(waveform)
        return embedding


# ============================================
# Demo
# ============================================
if __name__ == "__main__":
    import csv
    import soundfile as sf
    import torchaudio.functional as F_audio
    
    print("=" * 60)
    print("VGGish (VGG-style Audio CNN) - Demo")
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
    
    # 2. Load AudioSet labels
    print("\n2. Loading AudioSet labels...")
    labels_path = "/Users/stefano/Documents/PhD_main_project/datasets_gui_data/youtube/audioset/class_labels_indices.csv"
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        labels = {int(row[0]): row[2] for row in reader}
    print(f"   Loaded {len(labels)} labels")
    
    # 3. Load audio
    print("\n3. Loading audio...")
    audio_path = "/Users/stefano/Documents/PhD_main_project/utils/R9_ZSCveAHg_7s.wav"
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    
    print(f"   Duration: {waveform.shape[0]/SAMPLE_RATE:.2f}s")
    waveform = waveform.to(device)
    
    # 4. Create model and load checkpoint
    print("\n4. Creating model...")
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/vggish/vggish_with_classifier.pth"
    model = VGGish(sample_rate=SAMPLE_RATE)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 5. Inference
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
