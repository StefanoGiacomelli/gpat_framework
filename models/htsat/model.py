"""
HTS-AT (Hierarchical Token-Semantic Audio Transformer)
========================================
HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection.

Original repository: https://github.com/RetroCirce/HTS-Audio-Transformer

This is a standalone implementation of HTS-AT for AudioSet tagging.
"""

import math
import os
import warnings
from typing import Optional, Tuple, Union

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

# Model architecture defaults (internal)
_WINDOW_SIZE = 1024
_HOP_SIZE = 320
_MEL_BINS = 64
_FMIN = 50
_FMAX = 14000
_SPEC_SIZE = 256
_PATCH_SIZE = 4
_DEPTHS = [2, 2, 6, 2]
_NUM_HEADS = [4, 8, 16, 32]
_WINDOW_SIZE_ATTN = 8

# =============================================================================
# Helper Functions
# =============================================================================

def to_2tuple(x):
    """Convert to 2-tuple."""
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float, b: float) -> Tensor:
    """Truncated normal initialization without gradient."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    """Fill tensor with truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# =============================================================================
# Model Components (from layers.py)
# =============================================================================

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP module."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Window Attention (from htsat.py)
# =============================================================================

def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""
    
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Swin Transformer Blocks (from htsat.py)
# =============================================================================

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int,
                 num_heads: int, window_size: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: Union[float, list] = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, downsample: Optional[nn.Module] = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# =============================================================================
# HTS-AT Main Model
# =============================================================================

class HTSAT(nn.Module):
    """HTS-AT (Hierarchical Token-Semantic Audio Transformer) for AudioSet tagging."""
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_size: int = _WINDOW_SIZE,
        hop_size: int = _HOP_SIZE,
        mel_bins: int = _MEL_BINS,
        fmin: int = _FMIN,
        fmax: int = _FMAX,
        classes_num: int = CLASSES_NUM,
        spec_size: int = _SPEC_SIZE,
        patch_size: int = _PATCH_SIZE,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size_attn: int = _WINDOW_SIZE_ATTN,
    ):
        super().__init__()
        
        if depths is None:
            depths = _DEPTHS
        if num_heads is None:
            num_heads = _NUM_HEADS
            
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.classes_num = classes_num
        self.spec_size = spec_size
        self.patch_size = patch_size
        self.patch_stride = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.freq_ratio = spec_size // mel_bins  # 256 // 64 = 4

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window='hann',
            center=True, pad_mode='reflect', freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10,
            top_db=None, freeze_parameters=True)

        # Batch norm
        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        norm_layer = nn.LayerNorm
        
        # Split spectrogram into patches
        self.patch_embed = PatchEmbed(
            img_size=spec_size, patch_size=patch_size, in_chans=1,
            embed_dim=embed_dim, norm_layer=norm_layer)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=0.0)

        # Stochastic depth
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build transformer layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size_attn,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Token Semantic Module (TSCAM)
        SF = spec_size // (2 ** (self.num_layers - 1)) // self.patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=classes_num,
            kernel_size=(SF, 3),
            padding=(0, 1))
        self.head = nn.Linear(classes_num, classes_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor, frames_num: int) -> Tuple[Tensor, Tensor]:
        """
        Extract features through the transformer.
        
        Args:
            x: Patch embedded input (B, N, C)
            frames_num: Number of frames in spectrogram
            
        Returns:
            Tuple of (clipwise_output, latent_output/embedding)
        """
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, freq, time = x.shape
        
        # Group 2D CNN
        c_freq_bin = freq // self.freq_ratio
        x = x.reshape(B, C, freq // c_freq_bin, c_freq_bin, time)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        
        # Get latent output (embedding)
        latent_output = self.avgpool(torch.flatten(x, 2))
        latent_output = torch.flatten(latent_output, 1)
        
        # TSCAM convolution
        x = self.tscam_conv(x)
        x = torch.flatten(x, 2)  # B, C, T
        
        # Average pooling for clipwise output
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x, latent_output

    def reshape_wav2img(self, x: Tensor) -> Tensor:
        """
        Reshape mel spectrogram to image format for transformer.
        
        Original logic from HTS-AT: reshape to (B, 1, spec_size, spec_size)
        
        Args:
            x: Mel spectrogram (B, 1, T, F) where F is mel_bins
            
        Returns:
            Reshaped tensor (B, 1, spec_size, spec_size)
        """
        B, C, T, freq = x.shape
        target_T = int(self.spec_size * self.freq_ratio)  # 256 * 4 = 1024
        target_F = self.spec_size // self.freq_ratio       # 256 // 4 = 64
        
        assert T <= target_T and freq <= target_F, \
            f"Input spectrogram size ({T}, {freq}) should be <= ({target_T}, {target_F})"
        
        # Interpolate time dimension if needed
        if T < target_T:
            x = F.interpolate(x, (target_T, x.shape[3]), mode='bicubic', align_corners=True)
        
        # Interpolate frequency dimension if needed  
        if freq < target_F:
            x = F.interpolate(x, (x.shape[2], target_F), mode='bicubic', align_corners=True)
        
        # Reshape to square image
        # x: (B, C, target_T, target_F) = (B, 1, 1024, 64)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, F, T) = (B, 1, 64, 1024)
        # Reshape: split T dimension by freq_ratio
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        # (B, 1, 64, 4, 256)
        x = x.permute(0, 1, 3, 2, 4).contiguous()  # (B, 1, 4, 64, 256)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        # (B, 1, 256, 256)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass returning class probabilities.
        
        Args:
            x: Audio waveform (B, samples) or (B, 1, samples)
            
        Returns:
            Class probabilities (B, classes_num)
        """
        # Ensure 2D input for spectrogram extractor: (B, samples)
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, samples) -> (B, samples)
        
        # Spectrogram extraction
        x = self.spectrogram_extractor(x)  # (B, 1, T, F)
        x = self.logmel_extractor(x)  # (B, 1, T, M)
        
        # Batch norm (transpose for correct dimensions)
        x = x.transpose(1, 3)  # (B, M, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, M)
        
        # Reshape to image format
        x = self.reshape_wav2img(x)  # (B, 1, spec_size, spec_size)
        
        # Get frames_num from reshaped tensor (before patch_embed)
        frames_num = x.shape[2]  # spec_size = 256
        
        # Patch embedding
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        # Forward through transformer
        logits, _ = self.forward_features(x, frames_num)
        
        return torch.sigmoid(logits)

    def forward_with_embedding(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning both output and embedding.
        
        Args:
            x: Audio waveform (B, samples) or (B, 1, samples)
            
        Returns:
            Tuple of (clipwise_output, embedding)
        """
        # Ensure 2D input for spectrogram extractor: (B, samples)
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Spectrogram extraction
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        
        # Batch norm
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Reshape to image format
        x = self.reshape_wav2img(x)
        frames_num = x.shape[2]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        # Forward through transformer
        logits, embedding = self.forward_features(x, frames_num)
        
        return torch.sigmoid(logits), embedding

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Get only the embedding.
        
        Args:
            x: Audio waveform (B, samples) or (B, 1, samples)
            
        Returns:
            Embedding tensor (B, num_features)
        """
        _, embedding = self.forward_with_embedding(x)
        return embedding

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'sed_model.' prefix if present (from PyTorch Lightning wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('sed_model.'):
                new_key = k[10:]  # Remove 'sed_model.' prefix
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
    import torchaudio
    
    # Paths
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "HTSAT_AudioSet_Saved_3.ckpt")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("HTS-AT (Hierarchical Token-Semantic Audio Transformer) Demo")
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
    model = HTSAT()
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
