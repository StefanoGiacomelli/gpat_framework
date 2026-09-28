"""
AudioMAE (Masked Autoencoders that Listen)
==========================================
Vision Transformer finetuned on AudioSet with masked autoencoder pretraining.

Original repository: https://github.com/facebookresearch/AudioMAE

This is a standalone implementation of AudioMAE for AudioSet tagging.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# timm imports (using timm 0.4.5)
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_, to_2tuple


# ============================================
# CONSTANTS (hardcoded for AudioMAE / AudioSet)
# ============================================
SAMPLE_RATE = 16000
CLASSES_NUM = 527
EMBED_DIM = 768

# Mel spectrogram params (Kaldi-style fbank)
NUM_MEL_BINS = 128
FRAME_SHIFT = 10  # ms
TARGET_LENGTH = 1024  # frames (~10s @ 16kHz)

# AudioMAE architecture params (ViT-Base)
PATCH_SIZE = 16
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0

# AudioSet normalization stats
NORM_MEAN = -4.2677393
NORM_STD = 4.5689974


# ============================================
# Helper Functions
# ============================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.
    """
    if isinstance(grid_size, int):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    else:
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
    
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, len(grid_h), len(grid_w)])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# ============================================
# Building Blocks
# ============================================
class PatchEmbed(nn.Module):
    """
    Audio spectrogram to Patch Embedding.
    Supports non-square inputs (T x F mel spectrogram).
    """
    def __init__(self, img_size=(1024, 128), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_hw = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_hw[0] * self.patch_hw[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================================
# AUDIOMAE VISION TRANSFORMER
# ============================================
class AudioMAEViT(nn.Module):
    """
    Vision Transformer for AudioMAE, adapted for audio spectrograms.
    """
    def __init__(
        self,
        img_size=(TARGET_LENGTH, NUM_MEL_BINS),
        patch_size=PATCH_SIZE,
        in_chans=1,
        num_classes=CLASSES_NUM,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        global_pool=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        
        # Classifier head
        self.fc_norm = norm_layer(embed_dim) if global_pool else None
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Position embedding (sinusoidal)
        pos_embed = get_2d_sincos_pos_embed(
            self.embed_dim, 
            self.patch_embed.patch_hw,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Patch embedding projection
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Class token and head
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # Global average pooling (exclude cls)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]  # Use cls token
        
        embedding = x
        logits = self.head(x)
        return logits, embedding


# ============================================
# Preprocessing
# ============================================
class AudioPreprocessor:
    """
    Audio preprocessing: waveform -> normalized mel filterbank features.
    Uses Kaldi-style fbank computation.
    """
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        target_length=TARGET_LENGTH,
        num_mel_bins=NUM_MEL_BINS,
        mean=NORM_MEAN,
        std=NORM_STD,
    ):
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.num_mel_bins = num_mel_bins
        self.mean = mean
        self.std = std
    
    def __call__(self, waveform, sr=None):
        """
        Args:
            waveform: (channels, samples) or (samples,)
            sr: original sample rate (if different from self.sample_rate)
        Returns:
            fbank: (1, target_length, num_mel_bins)
        """
        # Ensure 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr is not None and sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Remove DC offset
        waveform = waveform - waveform.mean()
        
        # Kaldi-style fbank
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=FRAME_SHIFT,
        )
        
        # Pad or truncate
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            fbank = F.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:self.target_length, :]
        
        # Normalize
        fbank = (fbank - self.mean) / (self.std * 2)
        
        # Add channel: (T, F) -> (1, T, F)
        return fbank.unsqueeze(0)


# ============================================
# Main Model
# ============================================
class AudioMAE(nn.Module):
    """
    AudioMAE wrapper class with standard interface.
    
    Usage:
        model = AudioMAE()
        model.load_pretrained('finetuned.pth')
        probs = model(waveform)  # sigmoid probabilities
        embedding = model.get_embedding(waveform)
    """
    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.model = AudioMAEViT()
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Remove 'module.' prefix from DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
        
        self.model.load_state_dict(new_state_dict, strict=True)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """Preprocess waveform to mel spectrogram."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.shape[0]
        mels = []
        for i in range(batch_size):
            mel = self.preprocessor(waveform[i])
            mels.append(mel)
        
        return torch.stack(mels, dim=0)  # (B, 1, T, F)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning sigmoid probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
        Returns:
            probs: (batch, 527) sigmoid probabilities
        """
        mel = self.preprocess(waveform)
        mel = mel.to(next(self.model.parameters()).device)
        logits, _ = self.model(mel)
        return torch.sigmoid(logits)
    
    def forward_with_embedding(self, waveform: torch.Tensor):
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
        Returns:
            probs: (batch, 527) sigmoid probabilities
            embedding: (batch, 768) feature embedding
        """
        mel = self.preprocess(waveform)
        mel = mel.to(next(self.model.parameters()).device)
        logits, embedding = self.model(mel)
        return torch.sigmoid(logits), embedding
    
    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
        Returns:
            embedding: (batch, 768) feature embedding
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
    print("AudioMAE (Masked Autoencoders that Listen) - AudioSet Tagging Demo")
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
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/audiomae/finetuned.pth"
    model = AudioMAE(sample_rate=SAMPLE_RATE)
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
