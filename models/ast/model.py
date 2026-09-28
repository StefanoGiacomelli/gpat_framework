"""
AST (Audio Spectrogram Transformer)
========================================
AST: Audio Spectrogram Transformer.

Original repository: https://github.com/YuanGongND/ast

This is a standalone implementation of AST for AudioSet tagging.
"""

import torch
import torch.nn as nn
import torchaudio
import timm
from timm.models.layers import to_2tuple, trunc_normal_


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 16000
CLASSES_NUM = 527
EMBED_DIM = 768

# Filterbank parameters
AUDIOSET_MEAN = -4.2677393
AUDIOSET_STD = 4.5689974


# =============================================================================
# Helper Functions
# =============================================================================

# (No standalone helper functions - patch embedding is a building block)


# =============================================================================
# Building Blocks
# =============================================================================

class PatchEmbed(nn.Module):
    """Override timm PatchEmbed to relax input shape constraint."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# =============================================================================
# Main Model
# =============================================================================

class ASTModel(nn.Module):
    """
    Audio Spectrogram Transformer for AudioSet tagging.
    
    This model takes raw waveform as input, computes mel-filterbank features,
    and processes them through a Vision Transformer backbone.
    
    Args:
        label_dim: Number of output classes (default: 527 for AudioSet)
        fstride: Frequency stride for patch splitting (default: 10)
        tstride: Time stride for patch splitting (default: 10)
        input_fdim: Number of frequency bins (default: 128)
        input_tdim: Number of time frames (default: 1024, ~10s at 16kHz)
        sample_rate: Audio sample rate (default: 16000)
        model_size: ViT model size (default: 'base384')
    
    Input:
        (batch_size, num_samples) - Raw audio waveform at sample_rate Hz
    
    Output:
        (batch_size, label_dim) - Class probabilities (sigmoid activated)
    """
    
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, 
                 input_tdim=1024, sample_rate=16000, model_size='base384'):
        
        super(ASTModel, self).__init__()
        
        # Store config for preprocessing
        self.sample_rate = sample_rate
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        
        # Override timm PatchEmbed
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # Create ViT backbone (without pretrained weights - we'll load AudioSet checkpoint)
        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=False)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        
        # MLP head for classification
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), 
                                      nn.Linear(self.original_embedding_dim, label_dim))

        # Calculate output shape
        f_dim, t_dim = self._get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        # Modify patch embedding for single-channel spectrogram input
        new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        self.v.patch_embed.proj = new_proj

        # Initialize positional embedding for audio dimensions
        new_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.original_embedding_dim))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

    def _get_shape(self, fstride, tstride, input_fdim, input_tdim):
        """Calculate output shape after patch embedding."""
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def _waveform_to_fbank(self, waveform):
        """
        Convert raw waveform to mel-filterbank features.
        
        Args:
            waveform: (batch_size, num_samples) - Raw audio at self.sample_rate
        
        Returns:
            fbank: (batch_size, input_tdim, input_fdim) - Normalized filterbank features
        """
        batch_size = waveform.shape[0]
        fbanks = []
        
        for i in range(batch_size):
            # Get single waveform and ensure it's 2D for torchaudio
            wav = waveform[i].unsqueeze(0)  # (1, samples)
            
            # Zero-mean normalization
            wav = wav - wav.mean()
            
            # Compute filterbank features using Kaldi-compatible function (time_frames, num_mel_bins)
            fbank = torchaudio.compliance.kaldi.fbank(wav,
                                                      htk_compat=True,
                                                      sample_frequency=self.sample_rate,
                                                      use_energy=False,
                                                      window_type='hanning',
                                                      num_mel_bins=self.input_fdim,
                                                      dither=0.0,
                                                      frame_shift=10)  # 10ms frame shift
            
            # Pad or truncate to target length
            n_frames = fbank.shape[0]
            p = self.input_tdim - n_frames
            
            if p > 0:
                # Pad with zeros
                fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
            elif p < 0:
                # Truncate
                fbank = fbank[:self.input_tdim, :]
            
            # Normalize with AudioSet stats
            fbank = (fbank - AUDIOSET_MEAN) / (AUDIOSET_STD * 2)
            
            fbanks.append(fbank)
        
        # Stack batch
        fbanks = torch.stack(fbanks, dim=0)  # (batch, time, freq)
        
        return fbanks

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from checkpoint.
        
        The AudioSet checkpoint was saved with DataParallel wrapper,
        so we need to handle the 'module.' prefix in state dict keys.
        
        Args:
            checkpoint_path: Path to the .pth checkpoint file
        """
        device = next(self.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Handle DataParallel checkpoint (keys have 'module.' prefix)
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def get_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embedding without classification.
        
        Args:
            input: (batch_size, num_samples) - Raw audio waveform at sample_rate Hz
        
        Returns:
            embedding: (batch_size, 768) - Pre-classification embedding
        """
        # Convert waveform to filterbank features
        x = self._waveform_to_fbank(input)  # (batch, time, freq)
        
        # Add channel dimension and transpose for Conv2d
        x = x.unsqueeze(1)      # (batch, 1, time, freq)
        x = x.transpose(2, 3)   # (batch, 1, freq, time)

        # Patch embedding
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Add cls and distillation tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # Transformer blocks
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        
        # Average cls and distillation token outputs (pre-classification embedding)
        embedding = (x[:, 0] + x[:, 1]) / 2
        
        return embedding

    def forward(self, input):
        """
        Forward pass.
        
        Args:
            input: (batch_size, num_samples) - Raw audio waveform at sample_rate Hz
        
        Returns:
            output: (batch_size, label_dim) - Class probabilities (sigmoid activated)
        """
        # Convert waveform to filterbank features
        x = self._waveform_to_fbank(input)  # (batch, time, freq)
        
        # Add channel dimension and transpose for Conv2d
        x = x.unsqueeze(1)      # (batch, 1, time, freq)
        x = x.transpose(2, 3)   # (batch, 1, freq, time)

        # Patch embedding
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Add cls and distillation tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # Transformer blocks
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        
        # Average cls and distillation token outputs
        x = (x[:, 0] + x[:, 1]) / 2

        # Classification head
        x = self.mlp_head(x)
        
        # Apply sigmoid for probability output
        output = torch.sigmoid(x)
        
        return output


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import os
    import csv
    import numpy as np
    import soundfile as sf
    
    # Paths
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "audioset_10_10_0.4593.pth")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("AST (Audio Spectrogram Transformer) - AudioSet Tagging Demo")
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
    model = ASTModel()
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
