"""
EfficientAT - MobileNet (MN)
========================================
Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation.

Original repository: https://github.com/fschmid56/EfficientAT

This is a standalone implementation of the MobileNetV3 (MN) for AudioSet tagging.
"""

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.ops import Conv2dNormActivation


# ============================================
# CONSTANTS (hardcoded for MN / AudioSet)
# ============================================
SAMPLE_RATE = 32000
CLASSES_NUM = 527
EMBED_DIM = 3840  # lastconv_output_channels = 6 * 640 (with width_mult=4.0)

# Mel spectrogram params
N_MELS = 128
N_FFT = 1024
HOP_SIZE = 320      # 10ms @ 32kHz
WIN_LENGTH = 800    # 25ms @ 32kHz
F_MIN = 0.0
F_MAX = None        # Will be set to sr//2 - fmax_aug_range//2

# MN-40 architecture params (width_mult=4.0)
WIDTH_MULT = 4.0
STRIDES = (2, 2, 2, 2)
HEAD_TYPE = 'mlp'
DROPOUT = 0.2

# Squeeze-Excitation config
SE_DIMS = 'c'       # 'c' = channel, 'f' = frequency, 't' = time
SE_AGG = 'max'
SE_R = 4


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
# SQUEEZE-EXCITATION BLOCKS
# ============================================
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block on a specific dimension."""
    
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


class ConcurrentSEBlock(nn.Module):
    """Concurrent Squeeze-Excitation on multiple dimensions."""
    
    def __init__(self,
                 c_dim: int,
                 f_dim: int,
                 t_dim: int,
                 se_cnf: Dict):
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf['se_dims']:
            input_dim = dims[d-1]
            squeeze_dim = make_divisible(input_dim // se_cnf['se_r'], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))
        
        if se_cnf['se_agg'] == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]
        elif se_cnf['se_agg'] == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)
        elif se_cnf['se_agg'] == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)
        elif se_cnf['se_agg'] == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]
        else:
            raise NotImplementedError(f"SE aggregation '{se_cnf['se_agg']}' not implemented")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        se_outs = []
        for se_layer in self.conc_se_layers:
            se_outs.append(se_layer(input))
        out = self.agg_op(torch.stack(se_outs, dim=0))
        return out


# ============================================
# INVERTED RESIDUAL BLOCKS
# ============================================
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
                 width_mult: float):
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
    """MobileNetV3 Inverted Residual block with optional Squeeze-Excitation."""
    
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 se_cnf: Optional[Dict],
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
        
        # Squeeze-Excitation
        if cnf.use_se and se_cnf is not None and se_cnf['se_dims'] is not None:
            layers.append(ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf))

        # project
        layers.append(Conv2dNormActivation(cnf.expanded_channels, cnf.out_channels,
                                           kernel_size=1, norm_layer=norm_layer, activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


# ============================================
# ATTENTION POOLING
# ============================================
class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling from PSLA paper."""
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 att_activation: str = 'sigmoid',
                 clf_activation: str = 'ident',
                 num_heads: int = 4,
                 epsilon: float = 1e-7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.epsilon = epsilon
        self.att_activation = att_activation
        self.clf_activation = clf_activation
        
        self.subspace_proj = nn.Linear(self.in_dim, self.out_dim * 2 * self.num_heads)
        self.head_weight = nn.Parameter(torch.tensor([1.0 / self.num_heads] * self.num_heads).view(1, -1, 1))

    def activate(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        if activation == 'linear' or activation == 'ident':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = collapse_dim(x, dim=2)  # (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        b, n, c = x.shape

        x = self.subspace_proj(x).reshape(b, n, 2, self.num_heads, self.out_dim).permute(2, 0, 3, 1, 4)
        att, val = x[0], x[1]
        val = self.activate(val, self.clf_activation)
        att = self.activate(att, self.att_activation)
        att = torch.clamp(att, self.epsilon, 1. - self.epsilon)
        att = att / torch.sum(att, dim=2, keepdim=True)

        out = torch.sum(att * val, dim=2) * self.head_weight
        out = torch.sum(out, dim=1)
        return out


# ============================================
# Main Model
# ============================================
class MN(nn.Module):
    """MobileNetV3 for Audio Tagging."""
    
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = CLASSES_NUM,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 dropout: float = DROPOUT,
                 in_conv_kernel: int = 3,
                 in_conv_stride: int = 2,
                 in_channels: int = 1,
                 head_type: str = HEAD_TYPE,
                 multihead_attention_heads: int = 4,
                 se_conf: Optional[Dict] = None,
                 input_dims: Tuple[int, int] = (128, 1000),
                 **kwargs: Any):
        super().__init__()

        if block is None:
            block = InvertedResidual

        depthwise_norm_layer = norm_layer = \
            norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # First conv layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(in_channels,
                                           firstconv_output_channels,
                                           kernel_size=in_conv_kernel,
                                           stride=in_conv_stride,
                                           norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish))

        # Track frequency/time dimensions for SE
        f_dim, t_dim = input_dims
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(block(cnf, se_conf, norm_layer, depthwise_norm_layer))

        # Last conv layer
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(Conv2dNormActivation(lastconv_input_channels,
                                           lastconv_output_channels,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.head_type = head_type
        
        # Classifier head
        if self.head_type == "multihead_attention_pooling":
            self.classifier = MultiHeadAttentionPooling(lastconv_output_channels, num_classes, num_heads=multihead_attention_heads)
        elif self.head_type == "fully_convolutional":
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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: torch.Tensor, return_fmaps: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        fmaps = []
        
        for layer in self.features:
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.classifier(x).squeeze()
        
        if features.dim() == 1 and x.dim() == 1:
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)
        
        if return_fmaps:
            return x, fmaps
        else:
            return x, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning sigmoid probabilities (batch, 527)."""
        logits, _ = self._forward_impl(x)
        return torch.sigmoid(logits)

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (probs, embedding)."""
        logits, embed = self._forward_impl(x)
        return torch.sigmoid(logits), embed


# ============================================
# MODEL CONFIGURATION
# ============================================
def _mobilenet_v3_conf(width_mult: float = WIDTH_MULT,
                       reduced_tail: bool = False,
                       dilated: bool = False,
                       strides: Tuple[int, int, int, int] = STRIDES,
                       **kwargs: Any) -> Tuple[List[InvertedResidualConfig], int]:
    """Build MobileNetV3 configuration."""
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                                 bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),
                                 bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                                 bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),
                                 bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                                 bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                                 bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),
                                 bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                                 bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                                 bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                                 bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                                 bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                                 bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", strides[3], dilation),
                                 bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
                                 bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation)]
    
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


# ============================================
# MAIN CLASS WITH PREPROCESSING
# ============================================
class EfficientAT_MN(nn.Module):
    """
    EfficientAT MobileNet wrapper with integrated preprocessing.
    
    Args:
        sample_rate: Expected input sample rate (will resample if different)
        se_dims: Squeeze-Excitation dimensions ('c', 'f', 't' or combination)
        se_agg: SE aggregation operation ('max', 'avg', 'add', 'min')
        se_r: SE reduction ratio
    """
    
    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 se_dims: str = SE_DIMS,
                 se_agg: str = SE_AGG,
                 se_r: int = SE_R):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Parse SE config
        dim_map = {'c': 1, 'f': 2, 't': 3}
        if se_dims == 'none' or se_dims is None:
            se_dims_list = None
        else:
            se_dims_list = [dim_map[s] for s in se_dims]
        se_conf = dict(se_dims=se_dims_list, se_agg=se_agg, se_r=se_r)
        
        # Build model
        inverted_residual_setting, last_channel = _mobilenet_v3_conf()
        self.model = MN(inverted_residual_setting, last_channel,
                        se_conf=se_conf,
                        head_type=HEAD_TYPE)
        
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
            embedding: (batch, 3840) raw feature embedding
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
    print("EfficientAT MN-40 (Extended) - AudioSet Tagging Demo")
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
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/efficientat/mn40_as_ext_mAP_487.pt"
    model = EfficientAT_MN(sample_rate=SAMPLE_RATE)
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
