"""
M2D
============================================
Masked Modeling Duo: Towards a Universal Audio Pre-Training Framework.

Original repository: https://github.com/nttcslab/m2d

This is a standalone version of the M2D model pre-trained on AudioSet2M at 32KHz.
"""

import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.layers import trunc_normal_
from einops import rearrange
import nnAudio.features

from models.audioset_labels import canonical_reorder_index, load_canonical_audioset_mids


# ============================================
# CONSTANTS (hardcoded for M2D-AS @ 32kHz)
# ============================================
SAMPLE_RATE = 32000
EMBED_DIM = 768  # flat features (averaged over frequency bins)
CLASSES_NUM = 527

# Mel spectrogram defaults for 32kHz
N_FFT = 800
WIN_LENGTH = 800
HOP_LENGTH = 320
N_MELS = 80
F_MIN = 50
F_MAX = 16000

# ViT architecture
VIT_EMBED_DIM = 768
VIT_DEPTH = 12
VIT_NUM_HEADS = 12
VIT_MLP_RATIO = 4

# Input/patch sizes
INPUT_SIZE = [80, 1001]  # mel bins × time frames
PATCH_SIZE = [16, 16]


# ============================================
# Helper Functions
# ============================================
def expand_size(sz):
    """Expand single int to [int, int] tuple."""
    if isinstance(sz, int):
        return [sz, sz]
    return sz


# ============================================
# Building Blocks
# ============================================
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    
    Borrowed from timm 0.4.12 with modifications for flexible input sizes.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = expand_size(img_size)
        patch_size = expand_size(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# ============================================
# LOCAL VIT (M2D Audio Backbone)
# ============================================
class LocalViT(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer for M2D Audio.
    
    Modifications from standard ViT:
    - Custom PatchEmbed to avoid assertion failures with variable input sizes
    - Stores normalization statistics as buffer
    - No classification head (we add our own)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace PatchEmbed to avoid assertion failures
        self.patch_embed = PatchEmbed(self.patch_embed.img_size, 
                                      self.patch_embed.patch_size,
                                      self.patch_embed.proj.in_channels, 
                                      self.patch_embed.proj.out_channels)
        # Normalization statistics (will be loaded from checkpoint)
        self.norm_stats = nn.Parameter(torch.tensor([-7.1, 4.2]), requires_grad=False)
        # Remove default head (we add AudioSetHead separately)
        del self.head

    def patch_size(self):
        return np.array(self.patch_embed.patch_size)

    def grid_size(self):
        """Get grid size (num patches in each dimension)."""
        img_size = np.array(self.patch_embed.img_size)
        patch_size = self.patch_size()
        grid_size = img_size // patch_size
        return grid_size

    def forward_encoder(self, x):
        """
        Encode input through patch embedding and transformer blocks.
        
        Args:
            x: (batch, 1, mel_bins, time_frames) log-mel spectrogram
            
        Returns:
            (batch, num_patches + 1, embed_dim) encoded features with cls token
        """
        x = self.patch_embed(x)

        # Add positional embedding (without cls token)
        pos_embed = self.pos_embed[:, 1:, :]
        if x.shape[1] < pos_embed.shape[1]:
            # Shorten pos_embed for shorter inputs
            dims = pos_embed.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos_embed = pos_embed.reshape(1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins * frames, dims)
        x = x + pos_embed

        # Append cls token with its positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


# ============================================
# MEL SPECTROGRAM WRAPPER
# ============================================
class MelSpectrogramTransform(nn.Module):
    """
    Mel spectrogram transform using nnAudio.
    
    Produces log-mel spectrogram for M2D input.
    """
    
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
                 hop_length=HOP_LENGTH, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
        super().__init__()
        self.to_spec = nnAudio.features.MelSpectrogram(sr=sample_rate,
                                                       n_fft=n_fft,
                                                       win_length=win_length,
                                                       hop_length=hop_length,
                                                       n_mels=n_mels,
                                                       fmin=f_min,
                                                       fmax=f_max,
                                                       center=True,
                                                       power=2,
                                                       verbose=False)

    def forward(self, x):
        """
        Args:
            x: (batch, samples) waveform
            
        Returns:
            (batch, 1, n_mels, time_frames) log-mel spectrogram
        """
        spec = self.to_spec(x)
        spec = (spec + torch.finfo(spec.dtype).eps).log()
        spec = spec.unsqueeze(1)  # Add channel dimension
        return spec


# ============================================
# M2D AUDIO ENCODER
# ============================================
class M2DAudioEncoder(nn.Module):
    """
    M2D Audio Encoder (ViT-Base).
    
    This is the core encoder that produces frame-level embeddings.
    For classification, uses flat features (768-dim averaged over frequency bins).
    """
    
    def __init__(self, input_size=INPUT_SIZE, patch_size=PATCH_SIZE):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        
        # Mel spectrogram transform
        self.mel_transform = MelSpectrogramTransform()
        
        # ViT backbone
        self.backbone = LocalViT(in_chans=1,
                                 img_size=input_size,
                                 patch_size=patch_size,
                                 embed_dim=VIT_EMBED_DIM,
                                 depth=VIT_DEPTH,
                                 num_heads=VIT_NUM_HEADS,
                                 mlp_ratio=VIT_MLP_RATIO,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        # Feature dimensions
        self.n_freq_patches = input_size[0] // patch_size[0]  # 80 / 16 = 5
        self.flat_feature_dim = VIT_EMBED_DIM  # 768 (flat)
        self.stacked_feature_dim = VIT_EMBED_DIM * self.n_freq_patches  # 768 * 5 = 3840 (stacked)

    def normalize(self, x):
        """Normalize log-mel spectrogram using learned statistics."""
        mean, std = self.backbone.norm_stats
        return (x - mean) / std

    def encode_lms(self, x, flat_features=True, average_per_time_frame=False):
        """
        Encode log-mel spectrogram to frame-level features.
        
        Args:
            x: (batch, 1, mel_bins, time_frames) log-mel spectrogram
            flat_features: if True, keep patches as [B, f*t, d] (768-dim per patch)
                          if False, stack frequency bins [B, t, f*d] (3840-dim)
            average_per_time_frame: if True and flat_features=True, average over 
                                   frequency bins to get [B, t, d]
            
        Returns:
            (batch, num_patches_or_frames, feature_dim) embeddings
        """
        patch_fbins = self.backbone.grid_size()[0]
        unit_frames = self.input_size[1]
        patch_frames = self.backbone.patch_size()[1]
        embed_d = VIT_EMBED_DIM
        
        # Handle variable length inputs
        n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
        pad_frames = (patch_frames - (x.shape[-1] % unit_frames % patch_frames)) % patch_frames
        if pad_frames > 0:
            x = F.pad(x, (0, pad_frames))

        embeddings = []
        if flat_features:
            # Flat: keep all patch embeddings [B, f*t, d]
            for i in range(n_chunk):
                emb = self.backbone.forward_encoder(x[..., i * unit_frames:(i + 1) * unit_frames])
                emb = emb[..., 1:, :]  # Remove cls token -> [B, f*t, d]
                if average_per_time_frame:
                    # Average over frequency patches -> [B, t, d]
                    emb = rearrange(emb, 'b (f t) d -> b t d f', f=patch_fbins, d=embed_d).mean(-1)
                embeddings.append(emb)
        else:
            # Stacked: concatenate frequency bins -> [B, t, f*d]
            for i in range(n_chunk):
                emb = self.backbone.forward_encoder(x[..., i * unit_frames:(i + 1) * unit_frames])
                emb = emb[..., 1:, :]  # Remove cls token
                emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
                embeddings.append(emb)
        
        # Concatenate chunks along time axis
        x = torch.cat(embeddings, axis=-2)
        return x

    def forward(self, waveform, flat_features=True, average_per_time_frame=False):
        """
        Forward pass to get frame-level embeddings.
        
        Args:
            waveform: (batch, samples) @ 32kHz
            flat_features: if True, returns 768-dim features per patch
                          if False, returns 3840-dim features (stacked)
            average_per_time_frame: if True and flat_features=True, 
                                   average over frequency bins
            
        Returns:
            (batch, num_patches_or_frames, feature_dim) embeddings
        """
        # Compute log-mel spectrogram
        x = self.mel_transform(waveform)
        # Normalize
        x = self.normalize(x)
        # Encode
        return self.encode_lms(x, flat_features=flat_features, average_per_time_frame=average_per_time_frame)


# ============================================
# M2D (Main Model with Classification Head)
# ============================================
class M2D(nn.Module):
    """
    M2D AudioSet Tagger.
    
    Complete model with encoder and classification head for AudioSet tagging.
    Uses flat features (768-dim) with BatchNorm + Linear head.
    """
    
    def __init__(self, num_classes=CLASSES_NUM, feature_dim=EMBED_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Audio encoder
        self.encoder = M2DAudioEncoder()
        
        # Classification head (as in original EVAR fine-tuning)
        self.head_norm = nn.BatchNorm1d(feature_dim, affine=False)
        self.head = nn.Linear(feature_dim, num_classes)
        
        # Initialize head weights (will be overwritten by checkpoint)
        trunc_normal_(self.head.weight, std=2e-5)

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from EVAR fine-tuned checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Build new state dict with correct key mapping
        encoder_state_dict = {}
        head_norm_state_dict = {}
        head_state_dict = {}
        
        for key, value in state_dict.items():
            # Remove EVAR prefix: module.ar.runtime.backbone.* -> backbone.*
            if key.startswith('module.ar.runtime.backbone.'):
                new_key = key.replace('module.ar.runtime.backbone.', 'backbone.')
                encoder_state_dict[new_key] = value
            
            # Head norm: module.head.norm.* -> head_norm.*
            elif key.startswith('module.head.norm.'):
                new_key = key.replace('module.head.norm.', '')
                head_norm_state_dict[new_key] = value
            
            # Head linear: module.head.mlp.mlp.0.* -> head.*
            elif key.startswith('module.head.mlp.mlp.0.'):
                new_key = key.replace('module.head.mlp.mlp.0.', '')
                head_state_dict[new_key] = value
        
        # Load encoder weights
        msg = self.encoder.load_state_dict({'backbone.' + k.replace('backbone.', ''): v for k, v in encoder_state_dict.items()}, strict=False)
        
        # nnAudio keys are computed at runtime - this is normal
        other_missing = [k for k in msg.missing_keys if 'mel_transform.to_spec' not in k]
        if other_missing:
            print(f"   WARNING: Missing encoder keys: {other_missing}")
        
        # Load head norm weights (running_mean and running_var)
        if head_norm_state_dict:
            norm_sd = {'running_mean': head_norm_state_dict.get('running_mean'),
                       'running_var': head_norm_state_dict.get('running_var')}
            # Filter out None values
            norm_sd = {k: v for k, v in norm_sd.items() if v is not None}
            self.head_norm.load_state_dict(norm_sd, strict=False)
        
        # A fine-tuned AudioSet checkpoint must contain the EVAR classifier head.
        if not head_state_dict:
            raise RuntimeError(
                "M2D checkpoint does not contain module.head.mlp.mlp.0.*; "
                "a pretrained AudioSet classifier cannot be reconstructed"
            )
        self.head.load_state_dict(head_state_dict)

        # EVAR creates AudioSet targets with MultiLabelBinarizer without an
        # explicit class list, yielding lexicographically sorted MIDs. Reorder
        # the released classifier rows into the canonical AudioSet CSV order.
        canonical_mids = load_canonical_audioset_mids()
        evar_mids = sorted(canonical_mids)
        permutation = canonical_reorder_index(evar_mids, canonical_mids=canonical_mids)
        with torch.no_grad():
            weight = self.head.weight.detach().clone()
            bias = self.head.bias.detach().clone() if self.head.bias is not None else None
            self.head.weight.copy_(weight.index_select(0, permutation))
            if bias is not None:
                self.head.bias.copy_(bias.index_select(0, permutation))

        print(f"Loaded pretrained weights from {checkpoint_path}")
        print("Reordered M2D EVAR classifier from lexicographic MID order to canonical AudioSet order")

    def get_embedding(self, waveform: torch.Tensor, flat_features=True) -> torch.Tensor:
        """
        Extract audio embedding without classification.
        
        Args:
            waveform: (batch, samples) @ 32kHz
            flat_features: if True, returns 768-dim (default for classification)
                          if False, returns 3840-dim (stacked)
            
        Returns:
            embedding: (batch, feature_dim) time-averaged embedding
        """
        # Get frame-level embeddings
        frame_emb = self.encoder(waveform, flat_features=flat_features)
        # Time average
        clip_emb = frame_emb.mean(dim=1)
        return clip_emb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AudioSet classification.
        
        Args:
            waveform: (batch, samples) @ 32kHz
            
        Returns:
            (batch, 527) class probabilities (sigmoid)
        """
        # Get flat embedding (768-dim)
        embed = self.get_embedding(waveform, flat_features=True)
        # Apply BatchNorm (as in original EVAR)
        embed = self.head_norm(embed.unsqueeze(-1)).squeeze(-1)
        # Classify
        logits = self.head(embed)
        return torch.sigmoid(logits)


# ============================================
# Demo
# ============================================
if __name__ == "__main__":
    import os
    import csv
    import numpy as np
    import torchaudio
    import soundfile as sf

    # Paths
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "weights_ep69it3124-0.47998.pth")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("M2D (Masked Modeling Duo) - AudioSet Tagging Demo")
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
    model = M2D()
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
