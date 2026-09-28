"""
PANNs (ResNet38)
========================================
PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.

Original repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

This is a standalone vendorized implementation of ResNet38 for AudioSet tagging.
"""

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
EMBED_DIM = 2048


# =============================================================================
# Helper Functions
# =============================================================================

def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn: nn.Module) -> None:
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def _resnet_conv3x3(in_planes: int, out_planes: int) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with two conv layers and pooling."""
    
    def __init__(self, in_channels: int, out_channels: int):
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
        
    def forward(self, input: Tensor, pool_size: tuple = (2, 2), 
                pool_type: str = 'avg') -> Tensor:
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


class _ResnetBasicBlock(nn.Module):
    """ResNet BasicBlock for PANNs.
    
    Modified from torchvision ResNet to use AvgPool for downsampling
    instead of strided convolution.
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: nn.Module = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, 
                 norm_layer: nn.Module = None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")

        self.stride = stride
        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Apply avgpool before conv if stride==2
        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    """ResNet backbone for PANNs.
    
    Modified from torchvision to use AvgPool for downsampling.
    """
    
    def __init__(self, block: nn.Module, layers: list, zero_init_residual: bool = False,
                 groups: int = 1, width_per_group: int = 64, 
                 replace_stride_with_dilation: list = None, norm_layer: nn.Module = None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, 
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# =============================================================================
# Main Model
# =============================================================================

class ResNet38(nn.Module):
    """PANNs ResNet38 for audio classification.
    
    A ResNet-based model pretrained on AudioSet for audio tagging.
    Uses 38 layers (similar to ResNet-34 but with modifications for audio).
    
    Args:
        sample_rate: Audio sample rate (default: 32000)
        window_size: STFT window size (default: 1024)
        hop_size: STFT hop size (default: 320)
        mel_bins: Number of mel frequency bins (default: 64)
        fmin: Minimum frequency for mel filterbank (default: 50)
        fmax: Maximum frequency for mel filterbank (default: 14000)
        classes_num: Number of output classes (default: 527)
    """
    
    def __init__(
        self,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
    ):
        super(ResNet38, self).__init__()

        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, 
            hop_length=hop_size, 
            win_length=window_size, 
            window=window, 
            center=center, 
            pad_mode=pad_mode, 
            freeze_parameters=True
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax, 
            ref=ref, 
            amin=amin, 
            top_db=top_db, 
            freeze_parameters=True
        )

        # Batch normalization for mel spectrogram
        self.bn0 = nn.BatchNorm2d(mel_bins)

        # Initial conv block
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)

        # ResNet backbone: [3, 4, 6, 3] = 38 layers
        self.resnet = _ResNet(
            block=_ResnetBasicBlock, 
            layers=[3, 4, 6, 3], 
            zero_init_residual=True
        )

        # Post-ResNet conv block
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        # Fully connected layers
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for audio classification.
        
        Args:
            x: Input waveform of shape (batch, samples)
            
        Returns:
            Tensor of shape (batch, classes_num) with probabilities
        """
        # Spectrogram extraction
        x = self.spectrogram_extractor(x)   # (batch, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)        # (batch, 1, time_steps, mel_bins)
        
        # Normalize mel spectrogram
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Initial conv block
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        # ResNet backbone
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        # Post-ResNet conv block
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        # Global pooling
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # FC layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Classification
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        return clipwise_output

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Extract embedding (pre-classification features).
        
        Args:
            x: Input waveform of shape (batch, samples)
            
        Returns:
            Tensor of shape (batch, 2048) embedding
        """
        # Spectrogram extraction
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        
        # Normalize mel spectrogram
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Initial conv block
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        
        # ResNet backbone
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        # Post-ResNet conv block
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        
        # Global pooling
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # FC1 output is the embedding
        embedding = F.relu_(self.fc1(x))
        
        return embedding

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
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "ResNet38_mAP=0.434.pth")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("PANNs (ResNet38) Demo")
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
    model = ResNet38()
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
