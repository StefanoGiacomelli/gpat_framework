"""
AudioCLIP
========================================
AudioCLIP: Extending CLIP to Image, Text and Audio.

Original repository: https://github.com/AndreyGuzhov/AudioCLIP

This is a standalone version of the ESResNeXtFBSP audio encoder
used in AudioCLIP, with an additional classification head for AudioSet tagging.
"""

import math
import numpy as np
import scipy.signal as sps

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

from typing import cast
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional


# ============================================
# CONSTANTS (hardcoded for AudioCLIP/AudioSet)
# ============================================
SAMPLE_RATE = 44100
EMBED_DIM = 1024
CLASSES_NUM = 527

# Audio encoder defaults from audioclip.py
N_FFT = 2048
HOP_LENGTH = 561
WIN_LENGTH = 1654
WINDOW = 'blackmanharris'
NORMALIZED = True
ONESIDED = True
SPEC_HEIGHT = -1  # adaptive
SPEC_WIDTH = -1   # adaptive


# ============================================
# Helper Functions
# ============================================
def scale(old_value, old_min, old_max, new_min, new_max):
    """Scale value from one range to another."""
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def frame_signal(signal: torch.Tensor,
                 frame_length: int,
                 hop_length: int,
                 window: torch.Tensor = None) -> torch.Tensor:
    """Frame a signal into overlapping frames."""
    if window is None:
        window = torch.ones(frame_length, dtype=signal.dtype, device=signal.device)

    if window.shape[0] != frame_length:
        raise ValueError('Wrong `window` length: expected {}, got {}'.format(window.shape[0], frame_length))

    signal_length = signal.shape[-1]

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * signal_length - frame_length) / hop_length))

    pad_len = int((num_frames - 1) * hop_length + frame_length)
    if pad_len > signal_length:
        zeros = torch.zeros(pad_len - signal_length, device=signal.device, dtype=signal.dtype)

        while zeros.dim() < signal.dim():
            zeros.unsqueeze_(0)

        pad_signal = torch.cat((zeros.expand(*signal.shape[:-1], -1)[..., :zeros.shape[-1] // 2], signal), dim=-1)
        pad_signal = torch.cat((pad_signal, zeros.expand(*signal.shape[:-1], -1)[..., zeros.shape[-1] // 2:]), dim=-1)
    else:
        pad_signal = signal

    indices = torch.arange(0, frame_length, device=signal.device).repeat(num_frames, 1)
    indices += torch.arange(0, num_frames * hop_length, hop_length, device=signal.device).repeat(frame_length, 1).t_()
    indices = indices.long()

    frames = pad_signal[..., indices]
    frames = frames * window

    return frames


# ============================================
# CONVOLUTION HELPERS
# ============================================
def conv3x3(in_planes: int, out_planes: int, stride=1, groups: int = 1, dilation: Union[int, Tuple[int, int]] = 1):
    """3x3 convolution with padding. CREDITS: https://github.com/pytorch/vision"""
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: Union[int, Tuple[int, int]] = 1):
    """1x1 convolution. CREDITS: https://github.com/pytorch/vision"""
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


# ============================================
# ATTENTION MODULE
# ============================================
class Attention2d(nn.Module):
    """2D Attention module for ESResNet."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_kernels: int,
                 kernel_size: Tuple[int, int],
                 padding_size: Tuple[int, int]):
        super(Attention2d, self).__init__()

        self.conv_depth = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels * num_kernels,
                                    kernel_size=kernel_size,
                                    padding=padding_size,
                                    groups=in_channels)
        self.conv_point = nn.Conv2d(in_channels=in_channels * num_kernels,
                                    out_channels=out_channels,
                                    kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = F.adaptive_max_pool2d(x, size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ============================================
# RESNET BUILDING BLOCKS
# ============================================
class BasicBlock(nn.Module):
    """BasicBlock for ResNet. CREDITS: https://github.com/pytorch/vision"""

    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_layer: Optional[Type[nn.Module]] = None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet. CREDITS: https://github.com/pytorch/vision"""

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_layer: Optional[Type[nn.Module]] = None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ============================================
# LINEAR FBSP (Learnable Filterbank)
# ============================================
class LinearFBSP(nn.Module):
    """Learnable Filterbank Spectrogram layer."""

    def __init__(self, out_features: int, bias: bool = True, normalized: bool = False):
        super(LinearFBSP, self).__init__()

        self.out_features = out_features
        self.normalized = normalized
        self.eps = 1e-8

        default_dtype = torch.get_default_dtype()

        self.register_parameter('m', nn.Parameter(torch.zeros(self.out_features, dtype=default_dtype)))
        self.register_parameter('fb', nn.Parameter(torch.ones(self.out_features, dtype=default_dtype)))
        self.register_parameter('fc', nn.Parameter(torch.arange(self.out_features, dtype=default_dtype)))
        # Note: bias=False creates an empty tensor (shape [0]) for checkpoint compatibility
        self.register_parameter('bias',
                                nn.Parameter(torch.normal(0.0, 0.5, (self.out_features, 2), dtype=default_dtype)) if bias else nn.Parameter(torch.empty(0, dtype=default_dtype)))

        self.m.register_hook(lambda grad: grad / (torch.norm(grad, p=float('inf')) + self.eps))
        self.fb.register_hook(lambda grad: grad / (torch.norm(grad, p=float('inf')) + self.eps))
        self.fc.register_hook(lambda grad: grad / (torch.norm(grad, p=float('inf')) + self.eps))

    @staticmethod
    def power(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        magnitudes = (x1[..., 0] ** 2 + x1[..., 1] ** 2) ** 0.5
        phases = x1[..., 1].atan2(x1[..., 0])

        power_real = x2[..., 0]
        power_imag = x2[..., 1]

        mag_out = ((magnitudes ** 2) ** (0.5 * power_real) * torch.exp(-power_imag * phases))

        return mag_out.unsqueeze(-1) * torch.stack(((power_real * phases + 0.5 * power_imag * (magnitudes ** 2).log()).cos(),
                                                    (power_real * phases + 0.5 * power_imag * (magnitudes ** 2).log()).sin()), dim=-1)

    @staticmethod
    def sinc(x: torch.Tensor) -> torch.Tensor:
        return torch.where(cast(torch.Tensor, x == 0), torch.ones_like(x), torch.sin(x) / x)

    def _materialize_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]

        t = np.pi * torch.linspace(-1.0, 1.0, in_features, dtype=x.dtype, device=x.device).reshape(1, -1, 1) + self.eps

        m = self.m.reshape(-1, 1, 1)
        fb = self.fb.reshape(-1, 1, 1)
        fc = self.fc.reshape(-1, 1, 1)

        kernel = torch.cat((torch.cos(fc * t), -torch.sin(fc * t)), dim=-1)
        scale_factor = fb.sqrt()
        win = self.sinc(fb * t / (m + self.eps))
        win = self.power(torch.cat((win, torch.zeros_like(win)), dim=-1),
                         torch.cat((m, torch.zeros_like(m)), dim=-1))

        weights = scale_factor * torch.cat((win[..., :1] * kernel[..., :1] - win[..., 1:] * kernel[..., 1:],
                                            win[..., :1] * kernel[..., 1:] + win[..., 1:] * kernel[..., :1]), dim=-1)

        if self.normalized:
            weights = weights / (in_features ** 0.5)

        return weights, x_is_complex

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, x_is_complex = self._materialize_weights(x)

        if x_is_complex:
            x = torch.stack((F.linear(x[..., 0], weights[..., 0]) - F.linear(x[..., 1], weights[..., 1]),
                             F.linear(x[..., 0], weights[..., 1]) + F.linear(x[..., 1], weights[..., 0])), dim=-1)
        else:
            x = torch.stack((F.linear(x, weights[..., 0]),
                             F.linear(x, weights[..., 1])), dim=-1)

        if (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)):
            x = x + self.bias

        return x, weights

    def extra_repr(self) -> str:
        return 'out_features={}, bias={}, normalized={}'.format(self.out_features,
                                                                (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)),
                                                                self.normalized)


# ============================================
# RESNET WITH ATTENTION (Base Class)
# ============================================
class ResNetWithAttention(nn.Module):
    """ResNet with optional attention modules. CREDITS: https://github.com/pytorch/vision"""

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 apply_attention: bool = False,
                 num_channels: int = 3,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: bool = None,
                 norm_layer: Optional[Type[nn.Module]] = None):

        super(ResNetWithAttention, self).__init__()

        self.apply_attention = apply_attention

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f'replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}')

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if self.apply_attention:
            self.att1 = Attention2d(in_channels=64,
                                    out_channels=64 * block.expansion,
                                    num_kernels=1,
                                    kernel_size=(3, 1),
                                    padding_size=(1, 0))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        if self.apply_attention:
            self.att2 = Attention2d(in_channels=64 * block.expansion,
                                    out_channels=128 * block.expansion,
                                    num_kernels=1,
                                    kernel_size=(1, 5),
                                    padding_size=(0, 2))

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        if self.apply_attention:
            self.att3 = Attention2d(in_channels=128 * block.expansion,
                                    out_channels=256 * block.expansion,
                                    num_kernels=1,
                                    kernel_size=(3, 1),
                                    padding_size=(1, 0))

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if self.apply_attention:
            self.att4 = Attention2d(in_channels=256 * block.expansion,
                                    out_channels=512 * block.expansion,
                                    num_kernels=1,
                                    kernel_size=(1, 5),
                                    padding_size=(0, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.apply_attention:
            self.att5 = Attention2d(in_channels=512 * block.expansion,
                                    out_channels=512 * block.expansion,
                                    num_kernels=1,
                                    kernel_size=(3, 5),
                                    padding_size=(1, 2))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: Union[int, Tuple[int, int]] = 1,
                    dilate: bool = False) -> nn.Module:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))

        layers = list()
        layers.append(block(self.inplanes,
                            planes,
                            stride,
                            downsample,
                            self.groups,
                            self.base_width,
                            previous_dilation,
                            norm_layer))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_pre_processing(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.get_default_dtype())
        return x

    def _forward_pre_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_pre_features(x)

        if self.apply_attention:
            x_att = x.clone()
            x = self.layer1(x)
            x_att = self.att1(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer2(x)
            x_att = self.att2(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer3(x)
            x_att = self.att3(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer4(x)
            x_att = self.att4(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def _forward_reduction(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_attention:
            x_att = x.clone()
            x = self.avgpool(x)
            x_att = self.att5(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        return x

    def _forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_pre_processing(x)
        x = self._forward_features(x)
        x = self._forward_reduction(x)
        x = self._forward_classifier(x)
        return x


# ============================================
# ESRESNET BASE (Environmental Sound ResNet)
# ============================================
class _ESResNet(ResNetWithAttention):
    """Base class for Environmental Sound ResNet."""

    loading_func = staticmethod(tv.models.resnet50)

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 apply_attention: bool = False,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 pretrained: Union[bool, str] = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: bool = None,
                 norm_layer: Optional[Type[nn.Module]] = None):

        super(_ESResNet, self).__init__(block=block,
                                        layers=layers,
                                        apply_attention=apply_attention,
                                        num_channels=3,
                                        num_classes=num_classes,
                                        zero_init_residual=zero_init_residual,
                                        groups=groups,
                                        width_per_group=width_per_group,
                                        replace_stride_with_dilation=replace_stride_with_dilation,
                                        norm_layer=norm_layer)

        self.num_classes = num_classes

        self.fc = nn.Linear(in_features=self.fc.in_features,
                            out_features=self.num_classes,
                            bias=self.fc.bias is not None)

        if hop_length is None:
            hop_length = int(np.floor(n_fft / 4))

        if win_length is None:
            win_length = n_fft

        if window is None:
            window = 'boxcar'

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.normalized = normalized
        self.onesided = onesided

        self.spec_height = spec_height
        self.spec_width = spec_width

        self.pretrained = pretrained
        self._inject_members()

        window_buffer: torch.Tensor = torch.from_numpy(sps.get_window(window=window, Nx=win_length, fftbins=True)).to(torch.get_default_dtype())
        self.register_buffer('window', window_buffer)

        self.log10_eps = 1e-18

    def _inject_members(self):
        """Override in subclasses to inject additional members."""
        pass

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectrogram using STFT."""
        spec = torch.stft(x.view(-1, x.shape[-1]),
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window,
                          pad_mode='reflect',
                          normalized=self.normalized,
                          onesided=True,
                          return_complex=False)

        if not self.onesided:
            spec = torch.cat((torch.flip(spec, dims=(-3,)), spec), dim=-3)

        return spec

    def split_spectrogram(self, spec: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split spectrogram into frequency bands."""
        spec_height_per_band = spec.shape[-3] // self.conv1.in_channels
        spec_height_single_band = self.conv1.in_channels * spec_height_per_band
        spec = spec[:, :spec_height_single_band]

        spec = spec.reshape(batch_size, -1, spec.shape[-3] // self.conv1.in_channels, *spec.shape[-2:])

        return spec

    def spectrogram_to_power(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to power spectrogram."""
        spec_height = spec.shape[-3] if self.spec_height < 1 else self.spec_height
        spec_width = spec.shape[-2] if self.spec_width < 1 else self.spec_width

        pow_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2

        if spec_height != pow_spec.shape[-2] or spec_width != pow_spec.shape[-1]:
            pow_spec = F.interpolate(pow_spec,
                                     size=(spec_height, spec_width),
                                     mode='bilinear',
                                     align_corners=True)

        return pow_spec

    def _forward_pre_processing(self, x: torch.Tensor) -> torch.Tensor:
        x = super(_ESResNet, self)._forward_pre_processing(x)
        x = scale(x, -32768.0, 32767, -1.0, 1.0)

        spec = self.spectrogram(x)
        spec_split_ch = self.split_spectrogram(spec, x.shape[0])
        pow_spec_split_ch = self.spectrogram_to_power(spec_split_ch)
        pow_spec_split_ch = torch.where(cast(torch.Tensor, pow_spec_split_ch > 0.0),
                                        pow_spec_split_ch,
                                        torch.full_like(pow_spec_split_ch, self.log10_eps))
        pow_spec_split_ch = pow_spec_split_ch.reshape(x.shape[0], -1, self.conv1.in_channels, *pow_spec_split_ch.shape[-2:])
        x_db = torch.log10(pow_spec_split_ch).mul(10.0)

        return x_db

    def _forward_features(self, x_db: torch.Tensor) -> List[torch.Tensor]:
        outputs = list()
        for ch_idx in range(x_db.shape[1]):
            ch = x_db[:, ch_idx]
            out = super(_ESResNet, self)._forward_features(ch)
            outputs.append(out)

        return outputs

    def _forward_reduction(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = list()
        for ch in x:
            out = super(_ESResNet, self)._forward_reduction(ch)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=-1).sum(dim=-1)

        return outputs


# ============================================
# ESRESNET FBSP (with Learnable Filterbank)
# ============================================

# Global dict to store TTF weights for loss computation
_ttf_weights = dict()


class _ESResNetFBSP(_ESResNet):
    """ESResNet with learnable filterbank spectrogram."""

    def _inject_members(self):
        self.add_module('fbsp',
                        LinearFBSP(out_features=int(round(self.n_fft / 2)) + 1 if self.onesided else self.n_fft,
                                   normalized=self.normalized,
                                   bias=False))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectrogram using learnable filterbank."""
        with torch.no_grad():
            frames = frame_signal(signal=x.view(-1, x.shape[-1]),
                                  frame_length=self.win_length,
                                  hop_length=self.hop_length,
                                  window=self.window)

            if self.n_fft > self.win_length:
                pad_length = self.n_fft - self.win_length
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                frames = F.pad(frames, [pad_left, pad_right])

        spec, ttf_weights_ = self.fbsp(frames)

        spec = spec.transpose(-2, -3)
        _ttf_weights[x.device] = ttf_weights_

        return spec

    def loss_ttf(self, device: torch.device) -> torch.Tensor:
        """Compute TTF regularization loss."""
        ttf_norm = torch.norm(_ttf_weights[device], p=2, dim=[-1, -2])
        loss_ttf_norm = F.mse_loss(ttf_norm,
                                   torch.full_like(ttf_norm, 1.0 if self.normalized else self.n_fft ** 0.5))
        return loss_ttf_norm


# ============================================
# ESRESNEXT FBSP (Final Audio Encoder)
# ============================================
class ESResNeXtFBSP(_ESResNetFBSP):
    """
    ESResNeXt with learnable filterbank spectrogram.
    This is the audio encoder used in AudioCLIP.
    """

    loading_func = staticmethod(tv.models.resnext50_32x4d)

    def __init__(self,
                 n_fft: int = 2048,
                 hop_length: int = 561,
                 win_length: int = 1654,
                 window: str = 'blackmanharris',
                 normalized: bool = True,
                 onesided: bool = True,
                 spec_height: int = -1,
                 spec_width: int = -1,
                 num_classes: int = 1024,
                 apply_attention: bool = True,
                 pretrained: Union[bool, str] = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None):

        super(ESResNeXtFBSP, self).__init__(block=Bottleneck,
                                            layers=[3, 4, 6, 3],
                                            apply_attention=apply_attention,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            win_length=win_length,
                                            window=window,
                                            normalized=normalized,
                                            onesided=onesided,
                                            spec_height=spec_height,
                                            spec_width=spec_width,
                                            num_classes=num_classes,
                                            pretrained=pretrained,
                                            lock_pretrained=lock_pretrained,
                                            groups=32,
                                            width_per_group=4)


# ============================================
# AUDIOCLIP MODEL (Main Class)
# ============================================
class AudioCLIP(nn.Module):
    """
    AudioCLIP audio encoder with classification head for AudioSet.
    
    This model wraps the ESResNeXtFBSP audio encoder from AudioCLIP and adds
    a classification head for AudioSet tagging. The encoder weights can be
    loaded from the original AudioCLIP checkpoint.
    
    Args:
        sample_rate: Expected sample rate of input audio (default: 44100)
        embed_dim: Dimension of the audio embedding (default: 1024)
        classes_num: Number of output classes (default: 527 for AudioSet)
    
    Input:
        waveform: (batch, samples) tensor @ 44100 Hz
        
    Output:
        forward(): (batch, 527) sigmoid probabilities
        get_embedding(): (batch, 1024) L2-normalized embedding
    """

    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 embed_dim: int = EMBED_DIM,
                 classes_num: int = CLASSES_NUM):
        super(AudioCLIP, self).__init__()

        self.sample_rate = sample_rate
        self.embed_dim = embed_dim
        self.classes_num = classes_num

        # Audio encoder (ESResNeXtFBSP from AudioCLIP)
        self.encoder = ESResNeXtFBSP(n_fft=N_FFT,
                                     hop_length=HOP_LENGTH,
                                     win_length=WIN_LENGTH,
                                     window=WINDOW,
                                     normalized=NORMALIZED,
                                     onesided=ONESIDED,
                                     spec_height=SPEC_HEIGHT,
                                     spec_width=SPEC_WIDTH,
                                     num_classes=embed_dim,
                                     apply_attention=True,
                                     pretrained=False)

        # Classification head (randomly initialized, needs training)
        self.fc = nn.Linear(embed_dim, classes_num)
        self._init_classifier()
        # The released AudioCLIP checkpoint contains no pretrained 527-class head.
        # Keep the head available for future fine-tuning, but prevent the current
        # random classifier from being treated as a valid pretrained benchmark.
        self.evaluation_classifier_pretrained = False

    def _init_classifier(self):
        """Initialize classifier weights (Xavier uniform, like PANNs/AST)."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained AudioCLIP weights (encoder only).
        
        The original checkpoint contains full AudioCLIP model (audio + image + text).
        We extract only the 'audio.*' weights for our encoder.
        
        Args:
            checkpoint_path: Path to AudioCLIP-Full-Training.pt
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Filter only audio encoder weights
        audio_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('audio.'):
                # Remove 'audio.' prefix
                new_key = key[6:]  # len('audio.') = 6
                audio_state_dict[new_key] = value

        # Load into encoder
        self.encoder.load_state_dict(audio_state_dict, strict=True)
        print(f"Loaded pretrained audio encoder from {checkpoint_path}")

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embedding without classification.
        
        This produces the same embedding as the original AudioCLIP model's
        encode_audio() method.
        
        Args:
            waveform: (batch, samples) @ 44100 Hz
            
        Returns:
            embedding: (batch, 1024) L2-normalized
        """
        embed = self.encoder(waveform)
        embed = F.normalize(embed, dim=-1)
        return embed

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with classification.
        
        Args:
            waveform: (batch, samples) @ 44100 Hz
            
        Returns:
            probabilities: (batch, 527) sigmoid probabilities
        """
        embed = self.encoder(waveform)
        logits = self.fc(embed)
        return torch.sigmoid(logits)


# ============================================
# Demo
# ============================================
if __name__ == "__main__":
    import os
    import csv
    import numpy as np
    import soundfile as sf
    import torchaudio

    # Paths
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "AudioCLIP-Full-Training.pt")
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets_gui_data", "youtube", "audioset", "class_labels_indices.csv")

    print("=" * 60)
    print("AudioCLIP - AudioSet Tagging Demo")
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
    model = AudioCLIP()
    model.load_pretrained(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 5. Running inference
    print("\n5. Running inference (classifier NOT trained - random weights)...")
    with torch.no_grad():
        probs = model(waveform)
    print(f"   Output shape: {probs.shape}")

    # 6. Top-10 predictions
    print("\n6. Top-10 predictions (RANDOM - classifier not trained):")
    probs_np = probs[0].cpu().numpy()
    top_indices = probs_np.argsort()[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1:2d}. {labels[idx]:<40} {probs_np[idx]:.4f}")

    # 7. Embedding extraction
    print("\n7. Extracting embedding...")
    with torch.no_grad():
        embedding = model.get_embedding(waveform)
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {embedding.norm(dim=-1).item():.4f} (L2-normalized)")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Note: Classification results are random (classifier not trained).")
    print("=" * 60)
