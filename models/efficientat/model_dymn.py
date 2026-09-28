"""
EfficientAT - Dynamic MobileNet (DyMN)
========================================
Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio Models.

Original repository: https://github.com/fschmid56/EfficientAT

This is a standalone implementation of the Dynamic MobileNet (DyMN) for AudioSet tagging.
"""

import math
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.ops import Conv2dNormActivation


# ============================================
# CONSTANTS (hardcoded for DyMN / AudioSet)
# ============================================
SAMPLE_RATE = 32000
CLASSES_NUM = 527
EMBED_DIM = 1920  # lastconv_output_channels = 6 * 320

# Mel spectrogram params
N_MELS = 128
N_FFT = 1024
HOP_SIZE = 320      # 10ms @ 32kHz
WIN_LENGTH = 800    # 25ms @ 32kHz
F_MIN = 0.0
F_MAX = None        # Will be set to sr//2 - fmax_aug_range//2

# DyMN-20 architecture params (width_mult=2.0)
WIDTH_MULT = 2.0
STRIDES = (2, 2, 2, 2)
HEAD_TYPE = 'mlp'
DROPOUT = 0.2

# Dynamic block params
CONTEXT_RATIO = 4
MAX_CONTEXT_SIZE = 128
MIN_CONTEXT_SIZE = 32
DYRELU_K = 2
DYCONV_K = 4
TEMP_SCHEDULE = (1.0, 1.0, 1.0, 0.05)  # For pretrained AudioSet models


# ============================================
# Helper Functions
# ============================================
def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure channel count is divisible by divisor (typically 8)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    """Calculate CNN output size."""
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(x: torch.Tensor, dim: int, mode: str = "pool",
                 pool_fn: Callable[[torch.Tensor, int], torch.Tensor] = torch.mean,
                 combine_dim: int = None):
    """Collapse dimension by pooling or combining."""
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)


# ============================================
# AUDIO PREPROCESSING
# ============================================
class AugmentMelSTFT(nn.Module):
    """Mel spectrogram extractor with optional augmentation."""
    
    def __init__(self,
                 n_mels: int = N_MELS,
                 sr: int = SAMPLE_RATE,
                 win_length: int = WIN_LENGTH,
                 hopsize: int = HOP_SIZE,
                 n_fft: int = N_FFT,
                 freqm: int = 0,
                 timem: int = 0,
                 fmin: float = F_MIN,
                 fmax: float = None,
                 fmin_aug_range: int = 1,
                 fmax_aug_range: int = 1):
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
        
        if freqm == 0:
            self.freqm = nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-emphasis
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        
        # STFT
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)
        x = torch.view_as_real(x)
        x = (x ** 2).sum(dim=-1)  # power magnitude
        
        # Mel filterbank
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item() if self.training else self.fmin
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item() if self.training else self.fmax
        
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr, fmin, fmax,
                                                                 vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        
        mel_basis = torch.as_tensor(F.pad(mel_basis, (0, 1), mode='constant', value=0), device=x.device)
        
        with torch.amp.autocast('cuda', enabled=False):
            melspec = torch.matmul(mel_basis, x)
        
        # Log and normalize
        melspec = (melspec + 0.00001).log()
        
        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)
        
        melspec = (melspec + 4.5) / 5.0  # fast normalization
        return melspec


# ============================================
# INVERTED RESIDUAL (for non-dynamic blocks)
# ============================================
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self,
                 input_dim: int,
                 squeeze_dim: int,
                 se_dim: int,
                 activation: Callable[..., nn.Module] = nn.ReLU,
                 scale_activation: Callable[..., nn.Module] = nn.Sigmoid):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        assert se_dim in [1, 2, 3]
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(input, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale).view(shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    """Configuration for InvertedResidual blocks."""
    
    def __init__(self,
                 input_channels: int,
                 kernel: int,
                 expanded_channels: int,
                 out_channels: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 dilation: int,
                 width_mult: float,):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim = None
        self.t_dim = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block (used for non-dynamic positions)."""
    
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 se_cnf,  # Not used in DyMN context
                 norm_layer: Callable[..., nn.Module],
                 depthwise_norm_layer: Callable[..., nn.Module]):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(Conv2dNormActivation(cnf.input_channels,
                                               cnf.expanded_channels,
                                               kernel_size=1,
                                               norm_layer=norm_layer,
                                               activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(Conv2dNormActivation(cnf.expanded_channels,
                                           cnf.expanded_channels,
                                           kernel_size=cnf.kernel,
                                           stride=stride,
                                           dilation=cnf.dilation,
                                           groups=cnf.expanded_channels,
                                           norm_layer=depthwise_norm_layer,
                                           activation_layer=activation_layer))

        # project
        layers.append(Conv2dNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


# ============================================
# DYNAMIC BLOCKS
# ============================================
class DynamicInvertedResidualConfig:
    """Configuration for Dynamic Inverted Residual blocks."""
    
    def __init__(self,
                 input_channels: int,
                 kernel: int,
                 expanded_channels: int,
                 out_channels: int,
                 use_dy_block: bool,
                 activation: str,
                 stride: int,
                 dilation: int,
                 width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_dy_block = use_dy_block
        self.use_hs = activation == "HS"
        self.use_se = False
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class DynamicConv(nn.Module):
    """Dynamic Convolution with input-dependent kernel weighting."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 context_dim: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 att_groups: int = 1,
                 bias: bool = False,
                 k: int = 4,
                 temp_schedule: Tuple[float, float, float, float] = (30, 1, 1, 0.05)):
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.k = k
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = temp_schedule
        self.temperature = self.T_max
        self.att_groups = att_groups

        # Attention over k kernels
        self.residuals = nn.Sequential(nn.Linear(context_dim, k * self.att_groups))

        # k sets of weights for convolution
        weight = torch.randn(k, out_channels, in_channels // groups, kernel_size, kernel_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(k, out_channels), requires_grad=True)
        else:
            self.bias = None

        self._initialize_weights(weight, self.bias)

        weight = weight.view(1, k, att_groups, out_channels, in_channels // groups, kernel_size, kernel_size)
        weight = weight.transpose(1, 2).view(1, self.att_groups, self.k, -1)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def _initialize_weights(self, weight, bias):
        init_func = partial(nn.init.kaiming_normal_, mode="fan_out")
        for i in range(self.k):
            init_func(weight[i])
            if bias is not None:
                nn.init.zeros_(bias[i])

    def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
        b, c, f, t = x.size()
        g_c = g[0].view(b, -1)
        residuals = self.residuals(g_c).view(b, self.att_groups, 1, -1)
        attention = F.softmax(residuals / self.temperature, dim=-1)

        aggregate_weight = (attention @ self.weight).transpose(1, 2).reshape(b, self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        aggregate_weight = aggregate_weight.view(b * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)

        x = x.view(1, -1, f, t)
        if self.bias is not None:
            aggregate_bias = torch.mm(attention.view(b, -1), self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

    def update_params(self, epoch: int):
        """Temperature schedule for attention weights."""
        t0 = self.T_max - self.T0_slope * epoch
        t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
        self.temperature = max(t0, t1, self.T_min)


class DyReLU(nn.Module):
    """Base Dynamic ReLU."""
    
    def __init__(self, channels: int, context_dim: int, M: int = 2):
        super().__init__()
        self.channels = channels
        self.M = M
        self.coef_net = nn.Sequential(nn.Linear(context_dim, 2 * M))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * M + [0.5] * M).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * M - 1)).float())

    def get_relu_coefs(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.coef_net(x)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x: torch.Tensor, g) -> torch.Tensor:
        raise NotImplementedError


class DyReLUB(DyReLU):
    """Dynamic ReLU-B with channel-wise coefficients."""
    
    def __init__(self, channels: int, context_dim: int, M: int = 2):
        super().__init__(channels, context_dim, M)
        self.coef_net[-1] = nn.Linear(context_dim, 2 * M * self.channels)

    def forward(self, x: torch.Tensor, g) -> torch.Tensor:
        assert x.shape[1] == self.channels
        assert g is not None
        b, c, f, t = x.size()
        h_c = g[0].view(b, -1)
        theta = self.get_relu_coefs(h_c)

        relu_coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        x_mapped = x.unsqueeze(-1) * relu_coefs[:, :, :, :, :self.M] + relu_coefs[:, :, :, :, self.M:]
        
        if self.M == 2:
            result = torch.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        else:
            result = torch.max(x_mapped, dim=-1)[0]
        return result


class CoordAtt(nn.Module):
    """Coordinate Attention for frequency/time recalibration."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, g) -> torch.Tensor:
        g_cf, g_ct = g[1], g[2]
        a_f = g_cf.sigmoid()
        a_t = g_ct.sigmoid()
        return x * a_f * a_t


class DynamicWrapper(nn.Module):
    """Wrap a standard module to accept dynamic context argument."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
        return self.module(x)


class ContextGen(nn.Module):
    """Generate context vectors for dynamic modules."""
    
    def __init__(self,
                 context_dim: int,
                 in_ch: int,
                 exp_ch: int,
                 norm_layer: Callable[..., nn.Module],
                 stride: int = 1):
        super().__init__()
        self.joint_conv = nn.Conv2d(in_ch, context_dim, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.joint_norm = norm_layer(context_dim)
        self.joint_act = nn.Hardswish(inplace=True)
        
        self.conv_f = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_t = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

        if stride > 1:
            self.pool_f = nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
            self.pool_t = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        else:
            self.pool_f = nn.Sequential()
            self.pool_t = nn.Sequential()

    def forward(self, x: torch.Tensor, g=None):
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.size(2), ct.size(2)

        g_cat = torch.cat([cf, ct], dim=2)
        g_cat = self.joint_norm(self.joint_conv(g_cat))
        g_cat = self.joint_act(g_cat)

        h_cf, h_ct = torch.split(g_cat, [f, t], dim=2)
        h_ct = h_ct.permute(0, 1, 3, 2)
        h_c = torch.mean(g_cat, dim=2, keepdim=True)
        g_cf, g_ct = self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct))

        return (h_c, g_cf, g_ct)


class DY_Block(nn.Module):
    """Dynamic Block with Dy-Conv, Dy-ReLU, and Coordinate Attention."""
    
    def __init__(self,
                 cnf: DynamicInvertedResidualConfig,
                 context_ratio: int = 4,
                 max_context_size: int = 128,
                 min_context_size: int = 32,
                 temp_schedule: Tuple[float, float, float, float] = (30, 1, 1, 0.05),
                 dyrelu_k: int = 2,
                 dyconv_k: int = 4,
                 no_dyrelu: bool = False,
                 no_dyconv: bool = False,
                 no_ca: bool = False,
                 **kwargs: Any):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.context_dim = int(np.clip(make_divisible(cnf.expanded_channels // context_ratio, 8),
                                       make_divisible(min_context_size * cnf.width_mult, 8),
                                       make_divisible(max_context_size * cnf.width_mult, 8)))

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            if no_dyconv:
                self.exp_conv = DynamicWrapper(nn.Conv2d(cnf.input_channels, cnf.expanded_channels,
                                                         kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=0, bias=False))
            else:
                self.exp_conv = DynamicConv(cnf.input_channels, cnf.expanded_channels, self.context_dim,
                                            kernel_size=1, k=dyconv_k, temp_schedule=temp_schedule,
                                            stride=1, dilation=1, padding=0, bias=False)
            self.exp_norm = norm_layer(cnf.expanded_channels)
            self.exp_act = DynamicWrapper(activation_layer(inplace=True))
        else:
            self.exp_conv = DynamicWrapper(nn.Identity())
            self.exp_norm = nn.Identity()
            self.exp_act = DynamicWrapper(nn.Identity())

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        padding = (cnf.kernel - 1) // 2 * cnf.dilation
        if no_dyconv:
            self.depth_conv = DynamicWrapper(nn.Conv2d(cnf.expanded_channels, cnf.expanded_channels,
                                                       kernel_size=(cnf.kernel, cnf.kernel), groups=cnf.expanded_channels,
                                                       stride=(stride, stride), dilation=(cnf.dilation, cnf.dilation),
                                                       padding=padding, bias=False))
        else:
            self.depth_conv = DynamicConv(cnf.expanded_channels, cnf.expanded_channels, self.context_dim,
                                          kernel_size=cnf.kernel, k=dyconv_k, temp_schedule=temp_schedule,
                                          groups=cnf.expanded_channels, stride=stride, dilation=cnf.dilation,
                                          padding=padding, bias=False)
        self.depth_norm = norm_layer(cnf.expanded_channels)
        self.depth_act = DynamicWrapper(activation_layer(inplace=True)) if no_dyrelu else DyReLUB(cnf.expanded_channels, self.context_dim, M=dyrelu_k)

        self.ca = DynamicWrapper(nn.Identity()) if no_ca else CoordAtt()

        # project
        if no_dyconv:
            self.proj_conv = DynamicWrapper(nn.Conv2d(cnf.expanded_channels, cnf.out_channels,
                                                      kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=0, bias=False))
        else:
            self.proj_conv = DynamicConv(cnf.expanded_channels, cnf.out_channels, self.context_dim,
                                         kernel_size=1, k=dyconv_k, temp_schedule=temp_schedule,
                                         stride=1, dilation=1, padding=0, bias=False)
        self.proj_norm = norm_layer(cnf.out_channels)

        self.context_gen = ContextGen(self.context_dim, cnf.input_channels, cnf.expanded_channels, norm_layer=norm_layer, stride=stride)

    def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
        inp = x
        g = self.context_gen(x, g)
        
        x = self.exp_conv(x, g)
        x = self.exp_norm(x)
        x = self.exp_act(x, g)

        x = self.depth_conv(x, g)
        x = self.depth_norm(x)
        x = self.depth_act(x, g)
        x = self.ca(x, g)

        x = self.proj_conv(x, g)
        x = self.proj_norm(x)

        if self.use_res_connect:
            x += inp
        return x


# ============================================
# Main Model
# ============================================
class DyMN(nn.Module):
    """Dynamic MobileNet for Audio Tagging."""
    
    def __init__(self,
                 inverted_residual_setting: List[DynamicInvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = CLASSES_NUM,
                 head_type: str = HEAD_TYPE,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 dropout: float = DROPOUT,
                 in_conv_kernel: int = 3,
                 in_conv_stride: int = 2,
                 in_channels: int = 1,
                 context_ratio: int = CONTEXT_RATIO,
                 max_context_size: int = MAX_CONTEXT_SIZE,
                 min_context_size: int = MIN_CONTEXT_SIZE,
                 dyrelu_k: int = DYRELU_K,
                 dyconv_k: int = DYCONV_K,
                 no_dyrelu: bool = False,
                 no_dyconv: bool = False,
                 no_ca: bool = False,
                 temp_schedule: Tuple[float, float, float, float] = TEMP_SCHEDULE,
                 **kwargs: Any):
        super().__init__()

        if block is None:
            block = DY_Block

        norm_layer = norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.layers = nn.ModuleList()

        # First conv layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.in_c = Conv2dNormActivation(in_channels, firstconv_output_channels,
                                         kernel_size=in_conv_kernel, stride=in_conv_stride,
                                         norm_layer=norm_layer, activation_layer=nn.Hardswish)

        # Build blocks
        for cnf in inverted_residual_setting:
            if cnf.use_dy_block:
                b = block(cnf,
                          context_ratio=context_ratio,
                          max_context_size=max_context_size,
                          min_context_size=min_context_size,
                          dyrelu_k=dyrelu_k,
                          dyconv_k=dyconv_k,
                          no_dyrelu=no_dyrelu,
                          no_dyconv=no_dyconv,
                          no_ca=no_ca,
                          temp_schedule=temp_schedule)
            else:
                b = InvertedResidual(cnf, None, norm_layer, partial(nn.BatchNorm2d, eps=0.001, momentum=0.01))
            self.layers.append(b)

        # Last conv layer
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_c = Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels,
                                          kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish)

        # Classifier head
        self.head_type = head_type
        if self.head_type == "fully_convolutional":
            self.classifier = nn.Sequential(nn.Conv2d(lastconv_output_channels, num_classes,
                                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                            nn.BatchNorm2d(num_classes),
                                            nn.AdaptiveAvgPool2d((1, 1)))
        elif self.head_type == "mlp":
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(lastconv_output_channels, last_channel),
                                            nn.Hardswish(inplace=True),
                                            nn.Dropout(p=dropout, inplace=True),
                                            nn.Linear(last_channel, num_classes))
        else:
            raise NotImplementedError(f"Head '{self.head_type}' not supported")

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _feature_forward(self, x: torch.Tensor, return_fmaps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        fmaps = []
        x = self.in_c(x)
        if return_fmaps:
            fmaps.append(x)

        for layer in self.layers:
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)

        x = self.out_c(x)
        if return_fmaps:
            fmaps.append(x)
            return x, fmaps
        return x

    def _clf_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x, embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning sigmoid probabilities (batch, 527)."""
        x = self._feature_forward(x)
        logits, _ = self._clf_forward(x)
        return torch.sigmoid(logits)

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (probs, embedding)."""
        x = self._feature_forward(x)
        logits, embed = self._clf_forward(x)
        return torch.sigmoid(logits), embed

    def update_params(self, epoch: int):
        """Update temperature for Dynamic Convolutions."""
        for module in self.modules():
            if isinstance(module, DynamicConv):
                module.update_params(epoch)


# ============================================
# MODEL CONFIGURATION
# ============================================
def _dymn_conf(width_mult: float = WIDTH_MULT,
               reduced_tail: bool = False,
               dilated: bool = False,
               strides: Tuple[int, int, int, int] = STRIDES,
               use_dy_blocks: str = "all",
               **kwargs: Any) -> Tuple[List[DynamicInvertedResidualConfig], int]:
    """Build DyMN configuration."""
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(DynamicInvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(DynamicInvertedResidualConfig.adjust_channels, width_mult=width_mult)

    activations = ["RE", "RE", "RE", "RE", "RE", "RE", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS"]

    if use_dy_blocks == "all":
        use_dy_block = [True] * 15
    elif use_dy_blocks == "replace_se":
        use_dy_block = [False, False, False, True, True, True, False, False, False, False, True, True, True, True, True]
    else:
        raise NotImplementedError(f"Config use_dy_blocks={use_dy_blocks} not implemented.")

    inverted_residual_setting = [bneck_conf(16, 3, 16, 16, use_dy_block[0], activations[0], 1, 1),
                                 bneck_conf(16, 3, 64, 24, use_dy_block[1], activations[1], strides[0], 1),
                                 bneck_conf(24, 3, 72, 24, use_dy_block[2], activations[2], 1, 1),
                                 bneck_conf(24, 5, 72, 40, use_dy_block[3], activations[3], strides[1], 1),
                                 bneck_conf(40, 5, 120, 40, use_dy_block[4], activations[4], 1, 1),
                                 bneck_conf(40, 5, 120, 40, use_dy_block[5], activations[5], 1, 1),
                                 bneck_conf(40, 3, 240, 80, use_dy_block[6], activations[6], strides[2], 1),
                                 bneck_conf(80, 3, 200, 80, use_dy_block[7], activations[7], 1, 1),
                                 bneck_conf(80, 3, 184, 80, use_dy_block[8], activations[8], 1, 1),
                                 bneck_conf(80, 3, 184, 80, use_dy_block[9], activations[9], 1, 1),
                                 bneck_conf(80, 3, 480, 112, use_dy_block[10], activations[10], 1, 1),
                                 bneck_conf(112, 3, 672, 112, use_dy_block[11], activations[11], 1, 1),
                                 bneck_conf(112, 5, 672, 160 // reduce_divider, use_dy_block[12], activations[12], strides[3], dilation),
                                 bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[13], activations[13], 1, dilation),
                                 bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[14], activations[14], 1, dilation)]
    
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


# ============================================
# MAIN CLASS WITH PREPROCESSING
# ============================================
class EfficientAT_DyMN(nn.Module):
    """
    EfficientAT Dynamic MobileNet wrapper with integrated preprocessing.
    
    Args:
        sample_rate: Expected input sample rate (will resample if different)
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Build model
        inverted_residual_setting, last_channel = _dymn_conf()
        self.model = DyMN(inverted_residual_setting, last_channel)
        
        # Mel spectrogram extractor
        self.mel = AugmentMelSTFT()

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
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
        return self.model(mel)

    def forward_with_embedding(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            probs: (batch, 527) sigmoid probabilities
            embedding: (batch, embed_dim) feature embedding
        """
        mel = self.preprocess(waveform)
        return self.model.forward_with_embedding(mel)

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            embedding: (batch, 1920) raw feature embedding
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
    print("EfficientAT DyMN-20 - AudioSet Tagging Demo")
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
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/efficientat/dymn20_as_mAP_493.pt"
    model = EfficientAT_DyMN(sample_rate=SAMPLE_RATE)
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
