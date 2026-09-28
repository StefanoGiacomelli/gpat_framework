"""
PANNs (Wavegram-Logmel-CNN14)
========================================
PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.

Original repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

This is a standalone implementation of Wavegram-Logmel-CNN14 for AudioSet tagging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 32000
CLASSES_NUM = 527
EMBED_DIM = 2048


# =============================================================================
# Helper Functions
# =============================================================================

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=(3, 3), 
                               stride=(1, 1),
                               padding=(1, 1), 
                               bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=(3, 3), 
                               stride=(1, 1),
                               padding=(1, 1), 
                               bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect pool_type!')
        
        return x


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=3, 
                               stride=1,
                               padding=1, 
                               bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=3, 
                               stride=1, 
                               dilation=2, 
                               padding=2, 
                               bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x


# =============================================================================
# Main Model
# =============================================================================

class Wavegram_Logmel_Cnn14(nn.Module):
    """
    Wavegram-Logmel-CNN14 model for AudioSet tagging.
    
    This model combines raw waveform processing (Wavegram) with log-mel
    spectrogram features, concatenating them before the CNN backbone.
    
    Args:
        sample_rate: Audio sample rate (default: 32000 Hz for AudioSet)
        window_size: STFT window size (default: 1024)
        hop_size: STFT hop size (default: 320)
        mel_bins: Number of mel bins (default: 64)
        fmin: Minimum frequency for mel filterbank (default: 50 Hz)
        fmax: Maximum frequency for mel filterbank (default: 14000 Hz)
        classes_num: Number of output classes (default: 527 for AudioSet)
    
    Input:
        (batch_size, num_samples) - Raw audio waveform
    
    Output:
        (batch_size, classes_num) - Class probabilities (sigmoid activated)
    """
    
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Wavegram branch
        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
                                                 hop_length=hop_size, 
                                                 win_length=window_size, 
                                                 window=window, 
                                                 center=center, 
                                                 pad_mode=pad_mode, 
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
                                                 n_fft=window_size, 
                                                 n_mels=mel_bins, 
                                                 fmin=fmin, 
                                                 fmax=fmax, 
                                                 ref=ref, 
                                                 amin=amin, 
                                                 top_db=top_db, 
                                                 freeze_parameters=True)

        # Spec augmenter (used only during training)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        # CNN backbone
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # Classification head
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the .pth checkpoint file
        """
        device = next(self.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model'])
        print(f"Loaded pretrained weights from {checkpoint_path}")
 
    def forward(self, input, mixup_lambda=None):
        """
        Forward pass.
        
        Args:
            input: (batch_size, num_samples) - Raw audio waveform
            mixup_lambda: Optional mixup coefficients (used only during training)
        
        Returns:
            clipwise_output: (batch_size, classes_num) - Class probabilities
        """
        # Wavegram branch
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram branch
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)            # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram (training only)
        if self.training and mixup_lambda is not None:
            x = self._do_mixup(x, mixup_lambda)
            a1 = self._do_mixup(a1, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output

    @staticmethod
    def _do_mixup(x, mixup_lambda):
        """Mixup augmentation helper."""
        out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] + 
               x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
        return out

    def get_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embedding without classification.
        
        Args:
            input: (batch_size, num_samples) - Raw audio waveform @ 32000 Hz
        
        Returns:
            embedding: (batch_size, 2048) - Pre-classification embedding
        """
        # Wavegram branch
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram branch
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram
        x = torch.cat((x, a1), dim=1)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # fc1 output is the embedding (pre-classification)
        embedding = F.relu_(self.fc1(x))
        
        return embedding


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
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "Wavegram_Logmel_Cnn14_mAP=0.439.pth")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("PANNs (Wavegram-Logmel-CNN14) - AudioSet Tagging Demo")
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
    model = Wavegram_Logmel_Cnn14()
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
