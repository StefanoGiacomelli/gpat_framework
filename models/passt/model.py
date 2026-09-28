"""
PaSST (Patchout faSt Spectrogram Transformer)
=============================================
Efficient Training of Audio Transformers with Patchout.

Original repository: https://github.com/kkoutini/PaSST

This is a standalone implementation of PaSST-S for AudioSet tagging.
"""

import math
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ============================================
# CONSTANTS (hardcoded for PaSST-S / AudioSet)
# ============================================
SAMPLE_RATE = 32000
CLASSES_NUM = 527
EMBED_DIM = 768

# Mel spectrogram params
N_MELS = 128
N_FFT = 1024
HOP_SIZE = 320      # 10ms @ 32kHz
WIN_LENGTH = 800    # 25ms @ 32kHz
F_MIN = 0.0
F_MAX = None        # Will be set to sr//2 - fmax_aug_range//2

# PaSST-S architecture params
PATCH_SIZE = 16
STRIDE = (10, 10)
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0

# Expected input size for 10-second audio
INPUT_FDIM = 128    # mel bins
INPUT_TDIM = 998    # time frames (~10s @ hop=320)


# ============================================
# Helper Functions
# ============================================
def _ntuple(n):
    """Convert to n-tuple."""
    def parse(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return tuple([x] * n)
    return parse

to_2tuple = _ntuple(2)


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fill tensor with truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================
# MEL SPECTROGRAM EXTRACTOR
# ============================================
class AugmentMelSTFT(nn.Module):
    """Mel spectrogram extractor for PaSST."""
    
    def __init__(self,
                 n_mels: int = N_MELS,
                 sr: int = SAMPLE_RATE,
                 win_length: int = WIN_LENGTH,
                 hopsize: int = HOP_SIZE,
                 n_fft: int = N_FFT,
                 fmin: float = 0.0,
                 fmax: float = None,
                 fmin_aug_range: int = 1,
                 fmax_aug_range: int = 1000):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer('window', torch.hann_window(win_length, periodic=False), persistent=False)
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples) waveform
        Returns:
            melspec: (batch, n_mels, time) mel spectrogram
        """
        # Pre-emphasis
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        
        # STFT
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)
        x = x.abs() ** 2  # power magnitude
        
        # Mel filterbank (no augmentation in inference)
        fmin = self.fmin
        fmax = self.fmax
        
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr, fmin, fmax,
            vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0
        )
        mel_basis = torch.as_tensor(
            F.pad(mel_basis, (0, 1), mode='constant', value=0),
            device=x.device
        )
        
        with torch.amp.autocast('cuda', enabled=False):
            melspec = torch.matmul(mel_basis, x)
        
        # Log and normalize
        melspec = (melspec + 0.00001).log()
        melspec = (melspec + 4.5) / 5.  # fast normalization
        
        return melspec


# ============================================
# TRANSFORMER BLOCKS
# ============================================
class Mlp(nn.Module):
    """MLP block for Vision Transformer."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """2D spectrogram to Patch Embedding."""
    
    def __init__(self, img_size=(128, 998), patch_size=16, stride=16, in_chans=1, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# ============================================
# PaSST MODEL
# ============================================
class PaSSTModel(nn.Module):
    """
    PaSST: Patchout faSt Spectrogram Transformer.
    
    Based on DeiT (distilled ViT) with frequency/time positional embeddings.
    """
    
    def __init__(self,
                 img_size: Tuple[int, int] = (INPUT_FDIM, INPUT_TDIM),
                 patch_size: int = PATCH_SIZE,
                 stride: Tuple[int, int] = STRIDE,
                 in_chans: int = 1,
                 num_classes: int = CLASSES_NUM,
                 embed_dim: int = EMBED_DIM,
                 depth: int = DEPTH,
                 num_heads: int = NUM_HEADS,
                 mlp_ratio: float = MLP_RATIO,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 distilled: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride,
            in_chans=in_chans, embed_dim=embed_dim, flatten=False
        )

        # CLS and DIST tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        
        # Positional embeddings (separate for freq and time)
        self.new_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.grid_size[0], 1))
        self.time_new_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, self.patch_embed.grid_size[1]))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.head_dist = nn.Linear(embed_dim, num_classes) if distilled and num_classes > 0 else None

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.new_pos_embed, std=.02)
        trunc_normal_(self.freq_new_pos_embed, std=.02)
        trunc_normal_(self.time_new_pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through transformer, returns (CLS features, DIST features).
        """
        x = self.patch_embed(x)  # [B, E, F, T]
        B, E, F_dim, T_dim = x.shape
        
        # Add time/freq positional embeddings
        time_pos = self.time_new_pos_embed
        if x.shape[-1] < time_pos.shape[-1]:
            time_pos = time_pos[:, :, :, :x.shape[-1]]
        elif x.shape[-1] > time_pos.shape[-1]:
            x = x[:, :, :, :time_pos.shape[-1]]
        
        x = x + time_pos
        x = x + self.freq_new_pos_embed
        
        # Flatten: [B, E, F, T] -> [B, F*T, E]
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS/DIST tokens
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.new_pos_embed[:, :1, :]
        if self.dist_token is not None:
            dist_token = self.dist_token.expand(B, -1, -1) + self.new_pos_embed[:, 1:, :]
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        
        if self.dist_token is not None:
            return x[:, 0], x[:, 1]
        else:
            return x[:, 0], x[:, 0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, 1, n_mels, time) mel spectrogram
            
        Returns:
            logits: (batch, num_classes) raw logits
            features: (batch, embed_dim) averaged CLS+DIST features
        """
        cls_feat, dist_feat = self.forward_features(x)
        features = (cls_feat + dist_feat) / 2
        logits = self.head(features)
        return logits, features


# ============================================
# MAIN WRAPPER CLASS
# ============================================
class PaSST(nn.Module):
    """
    PaSST wrapper with integrated preprocessing.
    
    Args:
        sample_rate: Expected input sample rate
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Mel spectrogram extractor
        self.mel = AugmentMelSTFT(
            n_mels=N_MELS,
            sr=SAMPLE_RATE,
            win_length=WIN_LENGTH,
            hopsize=HOP_SIZE,
            n_fft=N_FFT,
            fmin=F_MIN,
            fmax=F_MAX
        )
        
        # PaSST model
        self.model = PaSSTModel()

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess waveform to mel spectrogram.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            mel: (batch, 1, n_mels, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Get mel spectrogram: (batch, n_mels, time)
        mel = self.mel(waveform)
        
        # Add channel dim: (batch, 1, n_mels, time)
        mel = mel.unsqueeze(1)
        
        return mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from raw waveform to class probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            probs: (batch, 527) sigmoid probabilities
        """
        mel = self.preprocess(waveform)
        logits, _ = self.model(mel)
        return torch.sigmoid(logits)

    def forward_with_embedding(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            probs: (batch, 527) sigmoid probabilities
            embedding: (batch, 768) feature embedding
        """
        mel = self.preprocess(waveform)
        logits, embedding = self.model(mel)
        return torch.sigmoid(logits), embedding

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            embedding: (batch, 768) raw feature embedding (avg of CLS+DIST)
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
    print("PaSST (Patchout faSt Spectrogram Transformer) - AudioSet Tagging Demo")
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
        labels = {i: row[2] for i, row in enumerate(reader)}
    print(f"   Loaded {len(labels)} labels")
    
    # 3. Load audio
    print("\n3. Loading audio...")
    audio_path = "/Users/stefano/Documents/PhD_main_project/utils/R9_ZSCveAHg_7s.wav"
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)  # mono
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    
    print(f"   Duration: {waveform.shape[0]/SAMPLE_RATE:.2f}s")
    waveform = waveform.to(device)
    
    # 4. Create model and load checkpoint
    print("\n4. Creating model...")
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/passt/passt-s-kd-ap.486.pt"
    model = PaSST(sample_rate=SAMPLE_RATE)
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
