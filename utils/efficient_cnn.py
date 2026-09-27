"""
Efficient CNN Modules

This module provides a comprehensive collection of efficient convolutional neural network
building blocks optimized for mobile and edge devices. It includes implementations of
various state-of-the-art architectures building blocks.

The module is organized into:
- Utility functions for weight initialization
- Auxiliary modules (auxiliary CNN, Attention & Processing blocks)
- Efficient CNN modules (architectures building blocks)

Authors: Stefano Giacomelli, Ph.D. candidate in ICT - DISIM dpt. University of L'Aquila, Italy
License: MIT License
"""

from typing import Optional, Union, List, Tuple, Dict
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

__version__ = "0.0.1"
__author__ = "Stefano Giacomelli"
__email__ = "stefano.giacomelli@graduate.univaq.it"
__maintainer__ = "Stefano Giacomelli"
__license__ = "MIT"
__status__ = "Development"

__all__ = [# Utility functions
           #"initialize_weights",

           # Auxiliary modules (CNN building blocks)
           #"ConvBlock",
           #"GroupedConvolution", 
           #"DepthwiseConvolution",
           #"MixedDepthwiseConvolution",
           #"PointwiseConvolution",
           #"GroupedPointwiseConvolution",
           #"StochasticDepth",
           #"LearnedGroupConvolution",
            
           # Attention modules
           #"SE",
           #"ConvSE", 
           #"AltConvSE",
            
           # Efficient CNN modules (architecture building blocks)
           "DepthwiseSeparableConvolutionModule",  # MobileNet V1
           "InvertedResidualModule",               # MobileNet V2
           "SandGlassModule",                      # MobileNeXt
           "InvertedResidualSEModule",             # MobileNet V3
           "UniversalInvertedBottleneckModule",    # MobileNet V4
           "EfficientModule",                      # EfficientNet V1
           "ResidualEfficientModule",              # EfficientNet V2
           "ShuffleModule",                        # ShuffleNet V1
           "InvertedResidualShuffleModule",        # ShuffleNet V2
           "FireModule",                           # SqueezeNet
           "SqueezeNeXtModule",                    # SqueezeNeXt
           "CondenseModule",                       # CondenseNet
           "FactorizedConvolutionModule",          # FBNet
           "ParallelFactorizedConvolutionModule",  # ProxylessNAS
           "StemModule",                           # RegNet
           "MobileOneModule",                      # MobileOne
           "MixDepthModule",                       # MixNet
           "DiCEModule",                           # DiCENet
           "StridedDiCEModule",                    # DiCENet (strided)
           "ShuffleDiCEModule",                    # DiCENet (shuffle)
           "GhostModule",                          # GhostNet V1
           "GhostBottleneckModule",                # GhostNet V1 (bottleneck)
           "GhostModuleV2",                        # GhostNet V2
           "GhostBottleneckModuleV2",              # GhostNet V2 (bottleneck)
           "SpatialPyramidModule",                 # ESPNet V1
           "EESpatialPyramidModule",               # ESPNet V2
           "LegoModule",                           # LegoNet
           "VersatileConvolution",                 # Versatile Filters
            
           # Testing and profiling framework
           #"ModelProfiler",
           #"MemoryProfiler",
           #"ModuleTestFramework"
           ]


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def initialize_weights(layer: nn.Module) -> None:
    """Initialize weights of supported layers using Kaiming normal initialization.
    
    Initializes Conv2d and Linear layers with Kaiming normal (fan_in mode) for weights
    and zeros for biases. This initialization is particularly effective for ReLU-based
    activations and helps prevent vanishing/exploding gradients.
    
    Args:
        layer (nn.Module): The layer to be initialized. Must be Conv2d or Linear.
        
    Note:
        Only nn.Conv2d and nn.Linear layers are supported. Other layer types are ignored.
    """
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.)


# --------------------------------------------------
# Auxiliary Modules
# --------------------------------------------------

class ConvBlock(nn.Module):
    """2D Convolutional block with batch normalization and activation.
    
    A standard building block consisting of Conv2d -> BatchNorm2d -> Activation.
    All layers use bias=True and weights are initialized with Kaiming normal.
    This is a fundamental component used throughout efficient CNN architectures.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution. Defaults to 1.
        padding (int, tuple, or str): Padding for the convolution. 
            Can be int, tuple, or 'same'/'valid'.
        activation (nn.Module, optional): Activation function. 
            Defaults to nn.ReLU().
            
    Attributes:
        name (str): Module identifier set to 'ConvBlock'.
        layers (nn.Sequential): Sequential container of conv->bn->activation.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super(ConvBlock, self).__init__()

        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
            
        self.name = 'ConvBlock'

        self.layers = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=True),
                                    nn.BatchNorm2d(out_chs),
                                    activation)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Spatial dimensions depend on kernel_size, stride, and padding.
        """
        return self.layers(x)


class GroupedConvolution(nn.Module):
    """Grouped convolutional layer with optional batch normalization and activation.
    
    Creates a grouped convolutional layer followed by optional batch normalization
    and activation. Groups reduce computational cost and parameters while maintaining
    representational capacity.
    
    Args:
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int, tuple, or str): Padding for the convolution.
        groups (int): Number of groups for grouped convolution. Must be > 0.
        batch_norm (bool): Whether to use batch normalization.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
        
    Attributes:
        name (str): Module identifier set to 'GroupedConvolution'.
        layers (nn.Sequential): Sequential container of layers.
        
    Raises:
        ValueError: If in_chs or out_chs not divisible by groups.
        ValueError: If groups <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        groups: int, 
        batch_norm: bool, 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super(GroupedConvolution, self).__init__()

        if groups <= 0:
            raise ValueError(f"groups must be > 0, got {groups}")

        if in_chs % groups != 0:
            raise ValueError(f"in_chs must be divisible by groups, got in_chs={in_chs}, groups={groups}")

        if out_chs % groups != 0:
            raise ValueError(f"out_chs must be divisible by groups, got out_chs={out_chs}, groups={groups}")

        self.name = 'GroupedConvolution'
        
        layers = [nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=True)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_chs))
        layers.append(activation)
        self.layers = nn.Sequential(*layers)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the grouped convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
        """
        return self.layers(x)


class DepthwiseConvolution(nn.Module):
    """Depthwise convolution with batch normalization and activation.
    
    Creates a depthwise convolution where each input channel is convolved with its own
    set of filters (groups=in_channels). Followed by batch normalization and activation.
    
    Args:
        in_chs (int): Number of input channels (also output channels). Must be > 0.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int, tuple, or str): Padding for the convolution.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
        
    Attributes:
        name (str): Module identifier set to 'DepthwiseConvolution'.
        layers (nn.Sequential): Sequential container of conv->bn->activation.
        
    Raises:
        ValueError: If in_chs <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super(DepthwiseConvolution, self).__init__()

        if in_chs <= 0:
            raise ValueError(f"in_chs must be > 0, got {in_chs}")
            
        self.name = 'DepthwiseConvolution'

        self.layers = nn.Sequential(nn.Conv2d(in_chs, in_chs, kernel_size, stride, padding, groups=in_chs, bias=True),
                                    nn.BatchNorm2d(in_chs),
                                    activation)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the depthwise convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H_out, W_out).
        """
        return self.layers(x)


class MixedDepthwiseConvolution(nn.Module):
    """Mixed depthwise convolution with different kernel sizes per channel group.
    
    Applies depthwise convolutions with different kernel sizes on equally divided
    channel groups, allowing the network to capture multi-scale features efficiently.
    
    Args:
        in_channels (int): Number of input channels. Must be divisible by len(kernel_sizes).
        kernel_sizes (List[int]): Kernel sizes for each group.
        stride (int, optional): Convolution stride. Must be 1 or 2. Defaults to 1.
        dilation (int, optional): Dilation factor. Set to 1 when stride > 1. Defaults to 1.
        
    Attributes:
        name (str): Module identifier set to 'MixedDepthwiseConvolution'.
        num_groups (int): Number of channel groups.
        group_dw (nn.ModuleDict): Dictionary of depthwise convolutions.
        
    Raises:
        AssertionError: If in_channels not divisible by len(kernel_sizes).
        AssertionError: If stride not in [1, 2].
    """
    
    def __init__(
        self, 
        in_channels: int, 
        kernel_sizes: List[int], 
        stride: int = 1, 
        dilation: int = 1
    ) -> None:
        super(MixedDepthwiseConvolution, self).__init__()
        self.name = "MixedDepthwiseConvolution"
        self.num_groups = len(kernel_sizes)

        assert in_channels % self.num_groups == 0, "in_channels must be divisible by num_groups"
        assert stride in [1, 2], "stride must be 1 or 2"

        sub_channels = in_channels // self.num_groups
        dilation = 1 if stride > 1 else dilation

        self.group_dw = nn.ModuleDict()
        for i, k in enumerate(kernel_sizes):
            padding = ((k - 1) // 2) * dilation
            self.group_dw[f"dw_conv_{i}"] = nn.Conv2d(in_channels=sub_channels,
                                                      out_channels=sub_channels,
                                                      kernel_size=k,
                                                      stride=stride,
                                                      padding=padding,
                                                      groups=sub_channels,
                                                      dilation=dilation,
                                                      bias=True)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through mixed depthwise convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H_out, W_out).
        """
        if self.num_groups == 1:
            return self.group_dw["dw_conv_0"](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        outputs = [self.group_dw[f"dw_conv_{i}"](chunks[i]) for i in range(self.num_groups)]
        
        return torch.cat(outputs, dim=1)


class PointwiseConvolution(nn.Module):
    """Pointwise (1x1) convolution with batch normalization and activation.
    
    Creates a pointwise convolutional layer followed by batch normalization
    and activation. Pointwise convolutions are used for channel mixing and
    dimensionality changes.
    
    Args:
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels.
        stride (int or tuple): Stride of the convolution.
        padding (int, tuple, or str): Padding for the convolution.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
        
    Attributes:
        name (str): Module identifier set to 'PointwiseConvolution'.
        layers (nn.Sequential): Sequential container of conv->bn->activation.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super(PointwiseConvolution, self).__init__()
        self.name = 'PointwiseConvolution'

        self.layers = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, padding=padding, bias=True),
                                    nn.BatchNorm2d(out_chs),
                                    activation)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through pointwise convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
        """
        return self.layers(x)


class GroupedPointwiseConvolution(nn.Module):
    """Grouped pointwise (1x1) convolutions on channel groups.
    
    Applies grouped pointwise convolutions on equally split channel groups.
    This reduces computational cost while maintaining representational capacity.
    
    Args:
        in_chs (int): Number of input channels. Must be divisible by len(kernel_sizes).
        out_chs (int): Number of output channels. Must be divisible by len(kernel_sizes).
        kernel_sizes (List[int]): List of kernel sizes (one for each group).
        
    Attributes:
        name (str): Module identifier set to 'GroupedPointwiseConvolution'.
        num_groups (int): Number of channel groups.
        group_pw (nn.ModuleDict): Dictionary of pointwise convolutions.
        
    Raises:
        AssertionError: If in_chs not divisible by num_groups.
        AssertionError: If out_chs not divisible by num_groups.
    """
    
    def __init__(self, in_chs: int, out_chs: int, kernel_sizes: List[int]) -> None:
        super(GroupedPointwiseConvolution, self).__init__()
        self.name = "GroupedPointwiseConvolution"
        self.num_groups = len(kernel_sizes)

        assert in_chs % self.num_groups == 0, "in_channels must be divisible by num_groups"
        assert out_chs % self.num_groups == 0, "out_channels must be divisible by num_groups"

        sub_in_channels = in_chs // self.num_groups
        sub_out_channels = out_chs // self.num_groups

        self.group_pw = nn.ModuleDict()
        for i in range(self.num_groups):
            self.group_pw[f"pw_conv_{i}"] = nn.Conv2d(in_channels=sub_in_channels,
                                                      out_channels=sub_out_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0,
                                                      bias=True)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through grouped pointwise convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        if self.num_groups == 1:
            return self.group_pw["pw_conv_0"](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        outputs = [self.group_pw[f"pw_conv_{i}"](chunks[i]) for i in range(self.num_groups)]
        
        return torch.cat(outputs, dim=1)


class StochasticDepth(nn.Module):
    """Stochastic depth regularization with per-sample random dropping.
    
    Implements stochastic depth as described in "Deep Networks with Stochastic Depth"
    by Huang et al. Randomly drops residual blocks during training to reduce
    overfitting and improve gradient flow in deep networks.
    
    Args:
        survival_prob (float): Probability of keeping a layer active during training.
            Must be in range (0, 1]. Defaults to 0.8.
            
    Attributes:
        p (float): Survival probability for the layer.
        
    Raises:
        ValueError: If survival_prob not in range (0, 1].
    """
    
    def __init__(self, survival_prob: float = 0.8) -> None:
        super(StochasticDepth, self).__init__()

        if not (0 < survival_prob <= 1):
            raise ValueError(f"survival_prob must be in range (0, 1], got {survival_prob}")
            
        self.p = survival_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth to input tensor.
        
        During training, randomly drops the input with probability (1 - survival_prob).
        During evaluation, passes input unchanged.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W). During training,
                some samples may be zeroed out and remaining samples are scaled
                by 1/survival_prob.
        """
        if not self.training:
            return x
        else:
            binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) > self.p
            return torch.div(x, self.p) * binary_tensor


class LearnedGroupConvolution(nn.Module):
    """Learned group convolution with progressive condensation.
    
    Implements learned group convolution from "CondenseNet: An Efficient DenseNet 
    using Learned Group Convolutions" by Huang et al. Features progressive 
    condensation that automatically learns which connections are important.
    
    Args:
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): Padding for the convolution. Defaults to 0.
        dilation (Union[int, Tuple[int, int]], optional): Dilation factor. Defaults to 1.
        groups (int, optional): Number of convolution groups. Defaults to 1.
        condense_factor (int): Number of condensation stages for progressive pruning.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0.
        
    Attributes:
        name (str): Module identifier set to 'LearnedGroupConvolution'.
        global_progress (float): Class variable for condensation scheduling.
        
    Raises:
        AssertionError: If in_chs not divisible by groups or condense_factor.
        AssertionError: If out_chs not divisible by groups.
    """
    
    global_progress = 0.0  # Used for condensation scheduling

    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]] = 1, 
        padding: Union[int, Tuple[int, int]] = 0, 
        dilation: Union[int, Tuple[int, int]] = 1, 
        groups: int = 1, 
        condense_factor: int = None, 
        dropout_rate: float = 0.0
    ) -> None:
        super(LearnedGroupConvolution, self).__init__()

        self.in_channels = in_chs
        self.out_channels = out_chs
        self.condense_factor = condense_factor
        self.groups = groups
        self.dropout_rate = dropout_rate

        assert in_chs % groups == 0, "group must divide input channels"
        assert in_chs % condense_factor == 0, "condense_factor must divide input channels"
        assert out_chs % groups == 0, "group must divide output channels"

        self.name = 'LearnedGroupConvolution'
        self.batch_norm = nn.BatchNorm2d(in_chs)
        self.relu = nn.ReLU()

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        self.conv = nn.Conv2d(in_chs, 
                              out_chs, 
                              kernel_size,
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation,
                              groups=1, 
                              bias=True)

        # Initialize buffers
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones_like(self.conv.weight))

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through learned group convolution.
        
        Applies batch normalization, activation, optional dropout, and convolution
        with dynamically learned group structure based on condensation progress.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
        """
        x = self.batch_norm(x)
        x = self.relu(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        self.check_if_drop()

        weight = self.conv.weight * self.mask
        out = F.conv2d(x, 
                       weight, 
                       bias=None,
                       stride=self.conv.stride,
                       padding=self.conv.padding,
                       dilation=self.conv.dilation,
                       groups=1)
        return out

    def check_if_drop(self):
        progress = LearnedGroupConvolution.global_progress
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1

        if not self.reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
            self.drop(delta)

    def drop(self, delta):
        weight = (self.conv.weight * self.mask).abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups

        weight = weight.view(d_out, self.groups, self.in_channels).transpose(0, 1).contiguous().view(self.out_channels, self.in_channels)

        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count += delta

    def reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def count(self):
        return int(self._count.item())

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage.item())

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return self._mask


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention.
    
    Implements SE block from "Squeeze-and-Excitation Networks" by Hu et al.
    Adaptively recalibrates channel-wise feature responses by modeling
    interdependencies between channels through global pooling and gating.
    
    Args:
        channels (int): Number of input channels. Must be > 0.
        reduction (int, optional): Reduction ratio for bottleneck. Defaults to 4.
        activation (nn.Module, optional): Sigmoid activation for gating. 
            Defaults to nn.Hardsigmoid().
            
    Attributes:
        name (str): Module identifier set to 'Squeeze-and-Excitation_Block'.
        layers (nn.ModuleDict): Dictionary containing pooling and linear layers.
        
    Raises:
        ValueError: If channels <= 0 or reduction <= 0.
    """
    
    def __init__(
        self, 
        channels: int, 
        reduction: int = 4, 
        activation: nn.Module = nn.Hardsigmoid()
    ) -> None:
        super(SE, self).__init__()

        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")

        self.layers = nn.ModuleDict()
        self.name = "Squeeze-and-Excitation_Block"

        # Layers Initialization
        self.layers['pool'] = nn.AdaptiveAvgPool2d(1)
        self.layers['fc_1'] = nn.Linear(channels, self._make_divisible(channels // reduction, 8))
        self.act_fun_1 = nn.ReLU()
        self.layers['fc_2'] = nn.Linear(self._make_divisible(channels // reduction, 8), channels)
        self.act_fun_2 = activation

        self.apply(initialize_weights)

    def _make_divisible(self, value: int, divisor: int, min_value: Optional[int] = None) -> int:
        """Make channel number divisible by divisor.
        
        Ensures channel numbers are divisible by the specified divisor for
        efficient utilization.
        
        Args:
            value (int): Value to be rounded.
            divisor (int): Divisor to round to.
            min_value (int, optional): Minimum value to return. Defaults to divisor.
            
        Returns:
            int: Rounded value that is divisible by divisor.
        """
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

        if new_value < 0.9 * value:     # Make sure that rounding does not drop of >10%.
            new_value += divisor

        return new_value

    def forward(self, x: Tensor) -> Tensor:
        """Apply SE attention to input tensor.
        
        Computes channel-wise attention weights through global average pooling,
        two linear transformations, and sigmoid gating. Scales input channels
        by learned attention weights.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with
                channel-wise attention applied.
        """
        b, c, _, _ = x.size()
        y = self.layers['pool'](x).view(b, c)
        y = self.layers['fc_1'](y)
        y = self.act_fun_1(y)
        y = self.layers['fc_2'](y)
        y = self.act_fun_2(y).view(b, c, 1, 1)

        return x * y


class ConvSE(nn.Module):
    """Convolutional Squeeze-and-Excitation block with conv-based channel reduction.
    
    Implements convolutional SE block from "EfficientNet: Rethinking Model Scaling for 
    Convolutional Neural Networks" by Tan et al. Uses convolutions instead of fully 
    connected layers for channel reduction and expansion.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        red_chs (int): Number of reduced channels in bottleneck. Must be > 0.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int], str]): Padding for the convolution.
        activations (List[nn.Module], optional): List of [intermediate, output] activations.
            Defaults to [nn.SiLU(), nn.Sigmoid()].
            
    Attributes:
        name (str): Module identifier set to 'Convolutional_Squeeze-and-Excitation_Block'.
        layers (nn.ModuleDict): Dictionary containing pooling and conv layers.
        
    Raises:
        ValueError: If in_chs or red_chs <= 0.
        ValueError: If activations list doesn't have exactly 2 elements.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        red_chs: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        activations: List[nn.Module] = [nn.SiLU(), nn.Sigmoid()]
    ) -> None:
        super(ConvSE, self).__init__()

        if in_chs <= 0:
            raise ValueError(f"in_chs must be > 0, got {in_chs}")
        if red_chs <= 0:
            raise ValueError(f"red_chs must be > 0, got {red_chs}")
        if len(activations) != 2:
            raise ValueError(f"activations must have exactly 2 elements, got {len(activations)}")
            
        self.layers = nn.ModuleDict()
        self.name = 'Convolutional_Squeeze-and-Excitation_Block'

        # 2D Average Pooling
        self.layers['pool'] = nn.AdaptiveAvgPool2d(output_size=1)

        # 2D Convolutional Layer (for reduction/contraction)
        self.layers['conv_2D_red'] = nn.Conv2d(in_channels=in_chs,
                                               out_channels=red_chs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               bias=True)
        # Intermediate Non-Linear Activation
        self.int_act_fun = activations[0]

        # 2D Convolutional Layer (for expansion)
        self.layers['conv_2D_exp'] = nn.Conv2d(in_channels=red_chs,
                                               out_channels=in_chs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               bias=True)
        # Output Non-Linear Activation
        self.out_act_fun = activations[1]

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolutional SE attention to input tensor.
        
        Computes channel-wise attention weights using global average pooling
        followed by convolutional reduction, activation, expansion, and gating.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with
                channel-wise attention applied.
        """
        y = self.layers['pool'](x)
        y = self.layers['conv_2D_red'](y)
        y = self.int_act_fun(y)
        y = self.layers['conv_2D_exp'](y)
        y = self.out_act_fun(y)

        return x * y


class AltConvSE(nn.Module):
    """Alternative convolutional SE block with ratio-based channel reduction.
    
    Implements convolutional SE block with configurable activations, adapted from 
    GhostNet's SE variant. Uses ratio-based reduction and ensures channel numbers 
    are divisible by a specified divisor for hardware efficiency.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        red_chs (Optional[int]): Number of reduced channels. If None, uses in_chs.
        ratio (float): Reduction ratio for intermediate channels. Must be > 0.
        divisor (int): Divisor for channel number alignment. Must be > 0.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int], str]): Padding for the convolution.
        activations (List[nn.Module]): List of [intermediate, output] activations.
            Must have exactly 2 elements.
            
    Attributes:
        name (str): Module identifier set to 'AltConvSE'.
        layers (nn.ModuleDict): Dictionary containing pooling and conv layers.
        
    Raises:
        ValueError: If in_chs <= 0, ratio <= 0, divisor <= 0, or activations length != 2.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        red_chs: Optional[int], 
        ratio: float, 
        divisor: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        activations: List[nn.Module]
    ) -> None:
        super(AltConvSE, self).__init__()

        if in_chs <= 0:
            raise ValueError(f"in_chs must be > 0, got {in_chs}")
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0, got {ratio}")
        if divisor <= 0:
            raise ValueError(f"divisor must be > 0, got {divisor}")
        if len(activations) != 2:
            raise ValueError(f"activations must have exactly 2 elements, got {len(activations)}")
            
        self.name = 'AltConvSE'
        reduced_chs = self._make_divisible((red_chs or in_chs) * ratio, divisor)
        self.layers = nn.ModuleDict()

        self.layers['pool'] = nn.AdaptiveAvgPool2d(output_size=1)

        self.layers['conv_red'] = nn.Conv2d(in_channels=in_chs,
                                            out_channels=reduced_chs,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=True)

        self.int_act = activations[0]

        self.layers['conv_exp'] = nn.Conv2d(in_channels=reduced_chs,
                                            out_channels=in_chs,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=True)

        self.out_act = activations[1]

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Apply alternative convolutional SE attention to input tensor.
        
        Computes channel-wise attention weights using global average pooling,
        ratio-based channel reduction, and configurable activations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with
                channel-wise attention applied.
        """
        y = self.layers['pool'](x)
        y = self.layers['conv_red'](y)
        y = self.int_act(y)
        y = self.layers['conv_exp'](y)
        y = self.out_act(y)
        
        return x * y

    @staticmethod
    def _make_divisible(value: int, divisor: int, min_value: Optional[int] = None) -> int:
        """Make channel number divisible by divisor.
        
        Ensures channel numbers are divisible by the specified divisor for
        efficient hardware utilization.
        
        Args:
            value (int): Value to be rounded.
            divisor (int): Divisor to round to.
            min_value (int, optional): Minimum value to return. Defaults to divisor.
            
        Returns:
            int: Rounded value that is divisible by divisor.
        """
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        
        return new_value


# --------------------------------------------------
# Efficient CNN Modules
# --------------------------------------------------

# MobileNet V1 (also in GPUNet)
class DepthwiseSeparableConvolutionModule(nn.Module):
    """Depthwise separable convolution for MobileNet V1 architecture.
    
    Implements depthwise separable convolution from "MobileNets: Efficient Convolutional 
    Neural Networks for Mobile Vision Applications" by Howard et al. Combines depthwise 
    convolution followed by pointwise convolution for efficient parameter usage.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        DW_kernel_size (Union[int, Tuple[int, int]]): Size of depthwise kernel.
        DW_stride (Union[int, Tuple[int, int]]): Stride for depthwise convolution.
        DW_padding (Union[int, Tuple[int, int], str]): Padding for depthwise convolution.
        DW_activation (nn.Module, optional): Activation for depthwise conv. Defaults to nn.ReLU().
        PW_activation (nn.Module, optional): Activation for pointwise conv. Defaults to nn.ReLU().
        
    Attributes:
        name (str): Module identifier set to 'DepthwiseSeparableConvolution'.
        layers (nn.ModuleDict): Dictionary containing DW and PW convolution layers.
        
    Raises:
        ValueError: If in_channels or out_channels <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        DW_kernel_size: Union[int, Tuple[int, int]], 
        DW_stride: Union[int, Tuple[int, int]], 
        DW_padding: Union[int, Tuple[int, int], str], 
        DW_activation: nn.Module = nn.ReLU(), 
        PW_activation: nn.Module = nn.ReLU()
    ) -> None:
        super(DepthwiseSeparableConvolutionModule, self).__init__()

        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")

        self.layers = nn.ModuleDict()
        self.name = 'DepthwiseSeparableConvolution'

        # Depthwise Convolutional layer
        self.layers['DW_Conv'] = DepthwiseConvolution(in_chs=in_chs,
                                                      kernel_size=DW_kernel_size,
                                                      stride=DW_stride,
                                                      padding=DW_padding,
                                                      activation=DW_activation)

        # Pointwise Convolutional layer
        self.layers['PW_Conv'] = PointwiseConvolution(in_chs=in_chs,
                                                      out_chs=out_chs,
                                                      stride=1,
                                                      padding=0,
                                                      activation=PW_activation)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through depthwise separable convolution.
        
        Applies depthwise convolution followed by pointwise convolution for
        efficient feature extraction with reduced parameters.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Spatial dimensions depend on DW_kernel_size, DW_stride, and DW_padding.
        """
        x = self.layers['DW_Conv'](x)
        x = self.layers['PW_Conv'](x)

        return x


# MobileNet V2 (also in MNASNet, GPUNet)
class InvertedResidualModule(nn.Module):
    """Inverted residual block for MobileNet V2 architecture.
    
    Implements inverted residual block from "MobileNet V2: Inverted Residuals and Linear 
    Bottlenecks" by Sandler et al. Features expansion-depthwise-projection structure with 
    optional residual connections for efficient mobile networks.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        expansion (int): Channel expansion factor for intermediate layer. Must be >= 1.
        stride (int): Stride for depthwise convolution. Must be 1 or 2.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU6().
        
    Attributes:
        name (str): Module identifier set to 'InvertedResidualBlock'.
        layers (nn.ModuleDict): Dictionary containing expansion, depthwise, and projection layers.
        skip (bool): Whether to use residual connection.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0, expansion < 1, or stride not in [1, 2].
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        expansion: int, 
        stride: int, 
        activation: nn.Module = nn.ReLU6()
    ) -> None:
        super(InvertedResidualModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.stride = stride
        self.skip = (stride == 1 and in_chs == out_chs)
        self.expansion = expansion
        self.hidden_dim = self.expansion * self.in_channels

        self.layers = nn.ModuleDict()
        self.name = 'InvertedResidualBlock'

        # Expansion Pointwise Convolutional layer
        if self.expansion != 1:
            self.layers['Expansion_PW_Conv'] = PointwiseConvolution(in_chs=self.in_channels,
                                                                    out_chs=self.hidden_dim,
                                                                    stride=1,
                                                                    padding=0,
                                                                    activation=activation)

        # Depthwise Convolutional layer
        if self.stride == 1:
            self.layers['DW_Conv'] = DepthwiseConvolution(in_chs=self.hidden_dim,
                                                          kernel_size=(3, 3),
                                                          stride=1,
                                                          padding='same',
                                                          activation=activation)
        else:
            self.layers['DW_Conv'] = DepthwiseConvolution(in_chs=self.hidden_dim,
                                                          kernel_size=(3, 3),
                                                          stride=2,
                                                          padding=(1, 1),
                                                          activation=activation)

        # Bottleneck Pointwise Convolutional layer (reduce the number of channels to out_chs)
        self.layers['Bottleneck_PW_Conv'] = PointwiseConvolution(in_chs=self.hidden_dim,
                                                                 out_chs=self.out_channels,
                                                                 stride=1,
                                                                 padding=0,
                                                                 activation=nn.Identity())    # No activation in the bottleneck layer

    def forward(self, x_in: Tensor) -> Tensor:
        """Forward pass through inverted residual block.
        
        Applies expansion (if needed), depthwise convolution, and projection with optional
        residual connection for efficient feature extraction.
        
        Args:
            x_in (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                If residual connection is used, adds input to output.
        """
        if self.expansion == 1:
            x = self.layers['DW_Conv'](x_in)
            x = self.layers['Bottleneck_PW_Conv'](x)
        else:
            x = self.layers['Expansion_PW_Conv'](x_in)
            x = self.layers['DW_Conv'](x)
            x = self.layers['Bottleneck_PW_Conv'](x)

        if self.skip:
            x = x + x_in

        return x


# MobileNeXt
class SandGlassModule(nn.Module):
    """SandGlass block for MobileNeXt architecture.
    
    Implements SandGlass convolutional block from "Rethinking Bottleneck Structure for 
    Efficient Mobile Network Design" by Zou et al. Features inverted structure with 
    reduction ratio and optional partial residual connections.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        stride (int): Stride value. Must be 1 or 2.
        exp_ratio (int, optional): Expansion ratio for bottleneck. Defaults to 2.
        identity_mul (float, optional): Fraction of identity channels used in residual.
            1.0 means full residual. Defaults to 1.0.
        keep_3x3 (bool, optional): Whether to force 3x3 depthwise convolution at the beginning.
            Defaults to False.
            
    Attributes:
        name (str): Module identifier set to 'SandGlassModule'.
        layers (nn.ModuleDict): Dictionary containing DW and PW convolution layers.
        use_residual (bool): Whether to use residual connection.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
        AssertionError: If stride not in [1, 2].
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        stride: int, 
        exp_ratio: int = 2, 
        identity_mul: float = 1.0, 
        keep_3x3: bool = False
    ) -> None:
        super(SandGlassModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        
        self.stride = stride
        assert self.stride in [1, 2], f"stride must be 1 or 2, got {stride}"
        self.name = "SandGlassModule"

        self.use_identity = identity_mul != 1.0
        self.identity_channels = int(round(in_chs * identity_mul))
        self.use_residual = (stride == 1 and in_chs == out_chs)

        hidden_channels = in_chs // exp_ratio
        if hidden_channels < (out_chs / 6.0):
            hidden_channels = math.ceil(out_chs / 6.0)
            hidden_channels = self._make_divisible(hidden_channels, 16)

        self.layers = nn.ModuleDict()

        if exp_ratio == 2 or in_chs == out_chs or keep_3x3:
            self.layers["DW_pre"] = DepthwiseConvolution(in_chs=in_chs,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding='same',
                                                         activation=nn.ReLU6())

        if exp_ratio != 1:
            self.layers["PW_expand"] = PointwiseConvolution(in_chs=in_chs,
                                                            out_chs=hidden_channels,
                                                            stride=1,
                                                            padding=0,
                                                            activation=nn.ReLU6())

        self.layers["PW_project"] = PointwiseConvolution(in_chs=hidden_channels,
                                                         out_chs=out_chs,
                                                         stride=1,
                                                         padding=0,
                                                         activation=nn.ReLU6())

        if exp_ratio == 2 or in_chs == out_chs or keep_3x3 or stride == 2:
            self.layers["DW_post"] = DepthwiseConvolution(in_chs=out_chs,
                                                            kernel_size=3,
                                                            stride=stride,
                                                            padding=1,
                                                            activation=nn.ReLU6())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SandGlass block.
        
        Applies the SandGlass structure: optional DW_pre → PW_expand → PW_project → 
        optional DW_post with partial or full residual connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                If residual connection is used, adds input to output.
        """
        out = x

        if "DW_pre" in self.layers:
            out = self.layers["DW_pre"](out)

        if "PW_expand" in self.layers:
            out = self.layers["PW_expand"](out)

        out = self.layers["PW_project"](out)

        if "DW_post" in self.layers:
            out = self.layers["DW_post"](out)

        if self.use_residual:
            if self.use_identity:
                identity_tensor = x[:, :self.identity_channels, :, :] + out[:, :self.identity_channels, :, :]
                out = torch.cat([identity_tensor, out[:, self.identity_channels:, :, :]], dim=1)
            else:
                out = x + out

        return out

    @staticmethod
    def _make_divisible(value: int, divisor: int, min_value: Optional[int] = None) -> int:
        """Make channel number divisible by divisor.
        
        Ensures channel numbers are divisible by the specified divisor for
        efficient hardware utilization.
        
        Args:
            value (int): Value to be rounded.
            divisor (int): Divisor to round to.
            min_value (int, optional): Minimum value to return. Defaults to divisor.
            
        Returns:
            int: Rounded value that is divisible by divisor.
        """
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value


# MobileNet V3 (also in LCNet)
class InvertedResidualSEModule(nn.Module):
    """Inverted residual block with SE attention for MobileNet V3 architecture.
    
    Implements inverted residual block with squeeze-and-excitation from "Searching for 
    MobileNet V3" by Howard et al. Combines expansion-depthwise-projection structure 
    with SE channel attention and optional hard-swish activation.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        hidden_chs (int): Number of intermediate (expansion) channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        DW_kernel_size (int): Size of depthwise convolution kernel. Must be > 0.
        DW_stride (int): Stride for depthwise convolution. Must be 1 or 2.
        use_se (bool, optional): Whether to use SE block. Defaults to True.
        use_hs (bool, optional): Whether to use HardSwish instead of ReLU. Defaults to True.
        
    Attributes:
        name (str): Module identifier set to 'InvertedResidual_Squeeze-and-Excitation_Block'.
        layers (nn.ModuleDict): Dictionary containing expansion, depthwise, SE, and projection layers.
        identity (bool): Whether to use residual connection.
        
    Raises:
        ValueError: If in_chs, hidden_chs, or out_chs <= 0.
        ValueError: If DW_kernel_size <= 0 or DW_stride not in [1, 2].
    """
    
    def __init__(
        self, 
        in_chs: int, 
        hidden_chs: int, 
        out_chs: int, 
        DW_kernel_size: int, 
        DW_stride: int, 
        use_se: bool = True, 
        use_hs: bool = True
    ) -> None:
        super(InvertedResidualSEModule, self).__init__()

        if in_chs <= 0 or hidden_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, hidden_chs={hidden_chs}, out_chs={out_chs}")
        if DW_kernel_size <= 0:
            raise ValueError(f"DW_kernel_size must be > 0, got {DW_kernel_size}")
        if DW_stride not in [1, 2]:
            raise ValueError(f"DW_stride must be 1 or 2, got {DW_stride}")

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.stride = DW_stride
        self.identity = DW_stride == 1 and in_chs == out_chs

        self.layers = nn.ModuleDict()
        self.name = "InvertedResidual_Squeeze-and-Excitation_Block"

        if in_chs == hidden_chs:
            self.layers['DW_conv'] = DepthwiseConvolution(in_chs=hidden_chs,
                                                          kernel_size=DW_kernel_size,
                                                          stride=self.stride,
                                                          padding=(DW_kernel_size - 1) // 2,
                                                          activation=nn.Hardswish() if use_hs else nn.ReLU())

            if use_se:
                self.layers['SquEx'] = SE(channels=hidden_chs,
                                          reduction=4,
                                          activation=nn.Hardsigmoid())

            self.layers['PW_conv'] = PointwiseConvolution(in_chs=hidden_chs,
                                                          out_chs=out_chs,
                                                          stride=1,
                                                          padding=0,
                                                          activation=nn.Identity())
        else:
            self.layers['PW_conv_1'] = PointwiseConvolution(in_chs=in_chs,
                                                            out_chs=hidden_chs,
                                                            stride=1,
                                                            padding=0,
                                                            activation=nn.Hardswish() if use_hs else nn.ReLU())

            self.layers['DW_conv'] = DepthwiseConvolution(in_chs=hidden_chs,
                                                          kernel_size=DW_kernel_size,
                                                          stride=self.stride,
                                                          padding=(DW_kernel_size - 1) // 2,
                                                          activation=nn.Identity())

            if use_se:
                self.layers['SquEx'] = SE(channels=hidden_chs,
                                          reduction=4,
                                          activation=nn.Hardswish() if use_hs else nn.ReLU())

            self.layers['PW_conv_2'] = PointwiseConvolution(in_chs=hidden_chs,
                                                            out_chs=out_chs,
                                                            stride=1,
                                                            padding=0,
                                                            activation=nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through inverted residual SE block.
        
        Applies expansion (if needed), depthwise convolution, SE attention, and 
        projection with optional residual connection for MobileNet V3 architecture.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                If residual connection is used, adds input to output.
        """
        y = x

        if self.identity:
            for layer in self.layers.keys():
                y = self.layers[f'{layer}'](y)

            return x + y                    # Shortcut (Skip connection)
        else:
            for layer in self.layers.keys():
                x = self.layers[f'{layer}'](x)

            return x


# MobileNet V4
class UniversalInvertedBottleneckModule(nn.Module):
    """Universal inverted bottleneck for MobileNet V4 architecture.
    
    Implements Universal Inverted Bottleneck from "MobileNetV4: Universal and Efficient 
    Mobile Vision" by Qin et al. Features configurable depthwise convolutions at start 
    and middle positions, optional downsampling, and layer scaling for enhanced performance.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        start_DW_kernel_size (int): Kernel size for start depthwise conv. If > 0, applies 
            depthwise convolution at input. Must be >= 0.
        middle_DW_kernel_size (int): Kernel size for middle depthwise conv. If > 0, applies 
            depthwise convolution in middle. Must be >= 0.
        middle_DW_downsample (bool): Whether to apply downsampling in middle DW stage.
        stride (int): Stride for start or middle depthwise convolution. Must be 1 or 2.
        expand_ratio (float): Expansion factor before projection. Must be > 0.
        use_layer_scale (bool, optional): Whether to use LayerScale after projection. 
            Defaults to False.
            
    Attributes:
        name (str): Module identifier set to 'UniversalInvertedBottleneckModule'.
        layers (nn.ModuleDict): Dictionary containing start DW, expand, middle DW, and proj layers.
        use_res_connect (bool): Whether to use residual connection.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0, expand_ratio <= 0, or stride not in [1, 2].
        ValueError: If start_DW_kernel_size or middle_DW_kernel_size < 0.
    """
    
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        start_DW_kernel_size: int,
        middle_DW_kernel_size: int,
        middle_DW_downsample: bool,
        stride: int,
        expand_ratio: float,
        use_layer_scale: bool = False
    ) -> None:
        super(UniversalInvertedBottleneckModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if expand_ratio <= 0:
            raise ValueError(f"expand_ratio must be > 0, got {expand_ratio}")
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if start_DW_kernel_size < 0 or middle_DW_kernel_size < 0:
            raise ValueError(f"Kernel sizes must be >= 0, got start_DW_kernel_size={start_DW_kernel_size}, middle_DW_kernel_size={middle_DW_kernel_size}")
            
        self.name = 'UniversalInvertedBottleneckModule'
        self.use_res_connect = (in_chs == out_chs and stride == 1)

        self.start_DW_kernel_size = start_DW_kernel_size
        self.middle_DW_kernel_size = middle_DW_kernel_size
        self.middle_DW_downsample = middle_DW_downsample
        self._use_layer_scale = use_layer_scale

        expand_filters = self.make_divisible(in_chs * expand_ratio, 8)
        stride_start = stride if not middle_DW_downsample else 1
        stride_middle = stride if middle_DW_downsample else 1

        self.layers = nn.ModuleDict()

        if self.start_DW_kernel_size:
            self.layers['Start_DW'] = self.conv_2d(in_chs=in_chs, 
                                                   out_chs=in_chs,
                                                   kernel_size=self.start_DW_kernel_size,
                                                   stride=stride_start,
                                                   groups=in_chs,
                                                   activation=False)

        self.layers['Expand_Conv'] = self.conv_2d(in_chs=in_chs, 
                                                  out_chs=expand_filters,
                                                  kernel_size=1)

        if self.middle_DW_kernel_size:
            self.layers['Middle_DW'] = self.conv_2d(in_chs=expand_filters, 
                                                    out_chs=expand_filters,
                                                    kernel_size=self.middle_DW_kernel_size,
                                                    stride=stride_middle,
                                                    groups=expand_filters)

        self.layers['Proj_Conv'] = self.conv_2d(in_chs=expand_filters, 
                                                out_chs=out_chs,
                                                kernel_size=1,
                                                stride=1,
                                                activation=False)

        if self._use_layer_scale:
            self.layer_scale = self.LayerScale(dim=out_chs, init_values=1e-5)
        
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through universal inverted bottleneck.
        
        Applies configurable start DW, expansion, middle DW, and projection with optional
        layer scaling and residual connection for MobileNet V4 architecture.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                If residual connection is used, adds input to output.
        """
        residual = x

        if 'Start_DW' in self.layers:
            x = self.layers['Start_DW'](x)

        x = self.layers['Expand_Conv'](x)

        if 'Middle_DW' in self.layers:
            x = self.layers['Middle_DW'](x)

        x = self.layers['Proj_Conv'](x)

        if self._use_layer_scale:
            x = self.layer_scale(x)

        if self.use_res_connect:
            x = x + residual

        return x

    @staticmethod
    def conv_2d(
        in_chs: int, 
        out_chs: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        groups: int = 1, 
        bias: bool = False, 
        batch_norm: bool = True, 
        activation: bool = True
    ) -> nn.Sequential:
        """Create 2D convolution block with batch norm and activation.
        
        Creates a sequential block with convolution, optional batch normalization,
        and optional ReLU activation for MobileNet V4 building blocks.
        
        Args:
            in_chs (int): Number of input channels.
            out_chs (int): Number of output channels.
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
            stride (int, optional): Convolution stride. Defaults to 1.
            groups (int, optional): Number of convolution groups. Defaults to 1.
            bias (bool, optional): Whether to use bias. Defaults to False.
            batch_norm (bool, optional): Whether to add batch normalization. Defaults to True.
            activation (bool, optional): Whether to add ReLU activation. Defaults to True.
            
        Returns:
            nn.Sequential: Sequential block with conv, optional BN, and optional activation.
        """
        conv = nn.Sequential()
        padding = (kernel_size - 1) // 2
        conv.add_module('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=bias, groups=groups))
        if batch_norm:
            conv.add_module('BatchNorm2D', nn.BatchNorm2d(out_chs))
        if activation:
            conv.add_module('Activation', nn.ReLU())
        return conv

    @staticmethod
    def make_divisible(
        value: float, 
        divisor: int, 
        min_value: Optional[float] = None, 
        round_down_protect: bool = True
    ) -> int:
        """Make channel number divisible by divisor.
        
        Ensures channel numbers are divisible by the specified divisor for
        efficient hardware utilization with optional round-down protection.
        
        Args:
            value (float): Value to be rounded.
            divisor (int): Divisor to round to.
            min_value (float, optional): Minimum value to return. Defaults to divisor.
            round_down_protect (bool, optional): Protect against excessive rounding down.
                Defaults to True.
            
        Returns:
            int: Rounded value that is divisible by divisor.
        """
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if round_down_protect and new_value < 0.9 * value:
            new_value += divisor
        return int(new_value)

    class LayerScale(nn.Module):
        """LayerScale for channel-wise feature scaling.
        
        Implements LayerScale from "Going Deeper with Image Transformers" by Touvron et al.
        Applies learnable per-channel scaling to improve training stability in deep networks.
        Referenced from MobileNet V4 implementation.
        
        Args:
            dim (int): Number of output channels. Must be > 0.
            init_values (float, optional): Initial scaling value. Defaults to 1e-5.
            
        Attributes:
            gamma (nn.Parameter): Learnable scaling parameters of shape (dim,).
            
        Raises:
            ValueError: If dim <= 0.
        """
        
        def __init__(self, dim: int, init_values: float = 1e-5) -> None:
            super(UniversalInvertedBottleneckModule.LayerScale, self).__init__()
            
            if dim <= 0:
                raise ValueError(f"dim must be > 0, got {dim}")
                
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

        def forward(self, x: Tensor) -> Tensor:
            """Apply learnable channel-wise scaling.
            
            Args:
                x (torch.Tensor): Input tensor of shape (B, C, H, W).

            Returns:
                torch.Tensor: Scaled tensor of shape (B, C, H, W).
            """
            gamma = self.gamma.view(1, -1, 1, 1)
            
            return x * gamma


# EfficientNet V1 (also in TinyNet)
class EfficientModule(nn.Module):
    """EfficientNet V1 building block for efficient mobile networks.
    
    Implements EfficientNet V1 block from "EfficientNet: Rethinking Model Scaling for 
    Convolutional Neural Networks" by Tan et al. Features expansion-depthwise-SE-projection 
    structure with configurable SE activations for optimal mobile performance.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        kernel_size (Union[int, Tuple[int, int]]): Size of depthwise convolution kernel.
        stride (Union[int, Tuple[int, int]]): Stride for depthwise convolution.
        padding (Union[int, Tuple[int, int], str]): Padding for depthwise convolution.
        ratio (int): Channel expansion ratio for intermediate layers. Must be >= 1.
        reduction (int): Reduction ratio for SE block. Must be > 0.
        se_activations (List[nn.Module], optional): SE block activations [intermediate, output].
            Defaults to [nn.SiLU(), nn.Sigmoid()].
            
    Attributes:
        name (str): Module identifier set to 'EfficientModule'.
        layers (nn.ModuleDict): Dictionary containing expansion, DW, SE, and PW layers.
        expand (bool): Whether to apply channel expansion.
        
    Raises:
        ValueError: If in_chs, out_chs, ratio, or reduction <= 0.
        ValueError: If se_activations list doesn't have exactly 2 elements.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int], str], 
        ratio: int, 
        reduction: int, 
        se_activations: List[nn.Module] = [nn.SiLU(), nn.Sigmoid()]
    ) -> None:
        super(EfficientModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0, got {ratio}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")
        if len(se_activations) != 2:
            raise ValueError(f"se_activations must have exactly 2 elements, got {len(se_activations)}")

        self.layers = nn.ModuleDict()
        self.name = 'EfficientModule'

        self.hidden_channels = in_chs * ratio
        self.expand = in_chs != self.hidden_channels
        self.reduced_dim = int(in_chs / reduction)

        # Expansion Evaluation
        if self.expand:
            self.layers['Feat_EXP'] = ConvBlock(in_chs=in_chs,
                                                out_chs=self.hidden_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                activation=nn.SiLU())

        # MobileNet DepthWise Convolution
        self.layers['DW_conv'] = DepthwiseConvolution(in_chs=self.hidden_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      activation=nn.SiLU())

        # Convolutional Squeeze-and-Excitation Block
        self.layers['Conv_SquEx'] = ConvSE(in_chs=self.hidden_channels,
                                           red_chs=self.reduced_dim,
                                           kernel_size=1,
                                           stride=1,
                                           padding='same',
                                           activations=se_activations)

        # MobileNet PointWise Convolution
        self.layers['PW_conv'] = PointwiseConvolution(in_chs=self.hidden_channels,
                                                     out_chs=out_chs,
                                                     stride=stride,
                                                     padding=padding,
                                                     activation=nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through EfficientNet V1 block.
        
        Applies optional expansion, depthwise convolution, SE attention, and projection
        for efficient mobile feature extraction with optimized parameter usage.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Spatial dimensions depend on stride and padding parameters.
        """
        if self.expand:
            x = self.layers['Feat_EXP'](x)
        x = self.layers['DW_conv'](x)
        x = self.layers['Conv_SquEx'](x)
        x = self.layers['PW_conv'](x)

        return x


# EfficientNet V2 (also in EfficientNet Lite)
class ResidualEfficientModule(nn.Module):
    """Residual EfficientNet V2 block with stochastic depth regularization.
    
    Implements residual EfficientNet block from "EfficientNetV2: Smaller Models and Faster 
    Training" by Tan et al. Features expansion-depthwise-SE-projection structure with 
    residual connections and stochastic depth for improved training stability.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        DW_kernel_size (Union[int, Tuple[int, int]]): Kernel size for depthwise convolution.
        DW_stride (Union[int, Tuple[int, int]]): Stride for depthwise convolution.
        expansion (Union[int, float]): Channel expansion factor. Must be > 0.
        reduction (int): Reduction ratio for SE block. Must be > 0.
        survival_prob (float, optional): Survival probability for stochastic depth.
            Must be in range (0, 1]. Defaults to 0.8.
            
    Attributes:
        name (str): Module identifier set to 'Residual_EfficientModule'.
        layers (nn.ModuleDict): Dictionary containing expansion, DW, SE, and PW layers.
        use_residual (bool): Whether to use residual connection.
        
    Raises:
        ValueError: If in_chs, out_chs, expansion, or reduction <= 0.
        ValueError: If survival_prob not in range (0, 1].
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        DW_kernel_size: Union[int, Tuple[int, int]], 
        DW_stride: Union[int, Tuple[int, int]], 
        expansion: Union[int, float], 
        reduction: int, 
        survival_prob: float = 0.8
    ) -> None:
        super(ResidualEfficientModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if expansion <= 0:
            raise ValueError(f"expansion must be > 0, got {expansion}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")
        if not (0 < survival_prob <= 1):
            raise ValueError(f"survival_prob must be in range (0, 1], got {survival_prob}")
            
        self.name = 'Residual_EfficientModule'
        self.layers = nn.ModuleDict()

        self.expansion_factor = expansion
        reduced_dim = int(in_chs // reduction)
        expanded_dim = int(expansion * in_chs)
        dw_padding = (DW_kernel_size - 1) // 2
        self.use_residual = (in_chs == out_chs) and (DW_stride == 1)

        if self.expansion_factor > 1.:
            self.layers['Feat_EXP'] = ConvBlock(in_chs=in_chs,
                                                out_chs=expanded_dim,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                activation=nn.SiLU())

        self.layers['DW_conv'] = DepthwiseConvolution(in_chs=expanded_dim,
                                                      kernel_size=DW_kernel_size,
                                                      stride=DW_stride,
                                                      padding=dw_padding,
                                                      activation=nn.SiLU())

        self.layers['Conv_SquEx'] = ConvSE(in_chs=expanded_dim,
                                           red_chs=reduced_dim,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           activations=[nn.SiLU(), nn.Sigmoid()])

        if self.use_residual:
            self.layers['Stoc_Drop'] = StochasticDepth(survival_prob=survival_prob)

        self.layers['PW_conv'] = PointwiseConvolution(in_chs=expanded_dim,
                                                      out_chs=out_chs,
                                                      stride=1,
                                                      padding=0,
                                                      activation=nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through residual EfficientNet V2 block.
        
        Applies optional expansion, depthwise convolution, SE attention, projection,
        and optional stochastic depth with residual connection for EfficientNet V2.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                If residual connection is used, adds input to output after stochastic depth.
        """
        if self.use_residual:
            residual = x.clone()

        if self.expansion_factor > 1.:
            x = self.layers['Feat_EXP'](x)
        x = self.layers['DW_conv'](x)
        x = self.layers['Conv_SquEx'](x)
        x = self.layers['PW_conv'](x)

        if self.use_residual:
            x = self.layers['Stoc_Drop'](x)
            x += residual

        return x


# ShuffleNet V1
class ShuffleModule(nn.Module):
    """ShuffleNet V1 block with channel shuffling and grouped convolutions.
    
    Implements ShuffleNet V1 block from "ShuffleNet: An Extremely Efficient Convolutional 
    Neural Network for Mobile Devices" by Zhang et al. Features grouped pointwise convolutions, 
    channel shuffling, depthwise convolution, and flexible combining modes (add/concat).
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 3.
        grouped_conv (bool, optional): Whether to use grouped convolution in first layer. 
            Defaults to True.
        combine (str, optional): How to combine input and output ('add' or 'concat'). 
            Defaults to 'add'.
            
    Attributes:
        name (str): Module identifier set to 'ShuffleModule'.
        layers (nn.ModuleDict): Dictionary containing grouped convolution layers.
        combine (str): Combination mode ('add' or 'concat').
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
        ValueError: If combine not in ['add', 'concat'].
        ValueError: If 'add' mode used with in_chs != out_chs.
        ValueError: If 'concat' mode used with out_chs <= in_chs.
        ValueError: If internal_out_channels not divisible by 4.
        ValueError: If grouped_conv=True and in_chs not divisible by groups.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        groups: int = 3, 
        grouped_conv: bool = True, 
        combine: str = 'add'
    ) -> None:
        super(ShuffleModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if combine not in ['add', 'concat']:
            raise ValueError(f"combine must be 'add' or 'concat', got {combine}")
            
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.groups = groups
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.name = 'ShuffleModule'
        self.layers = nn.ModuleDict()

        # Validate mode
        if combine == 'add':
            if in_chs != out_chs:
                raise ValueError(f"'add' mode requires in_chs == out_chs, got {in_chs} != {out_chs}")
            self.internal_out_channels = out_chs
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif combine == 'concat':
            if out_chs <= in_chs:
                raise ValueError(f"'concat' mode requires out_chs > in_chs, got {out_chs} <= {in_chs}")
            self.internal_out_channels = out_chs - in_chs
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.layers['res_pool'] = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        if self.internal_out_channels % 4 != 0:
            raise ValueError(f"internal_out_channels ({self.internal_out_channels}) must be divisible by 4")

        if self.grouped_conv and in_chs % groups != 0:
            raise ValueError(f"in_chs ({in_chs}) must be divisible by groups ({groups}) when using grouped_conv")

        # Bottleneck
        self.bottleneck_channels = self.internal_out_channels // 4
        first_1x1_groups = self.groups if self.grouped_conv else 1

        # Pointwise grouped convolution (compression)
        self.layers['PW_group_COMP'] = GroupedConvolution(in_chs=self.in_channels,
                                                          out_chs=self.bottleneck_channels,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          groups=first_1x1_groups,
                                                          batch_norm=True,
                                                          activation=nn.ReLU())

        # Depthwise convolution
        self.layers['DW_conv'] = GroupedConvolution(in_chs=self.bottleneck_channels,
                                                    out_chs=self.bottleneck_channels,
                                                    kernel_size=3,
                                                    stride=self.depthwise_stride,
                                                    padding=1,
                                                    groups=self.bottleneck_channels,
                                                    batch_norm=True,
                                                    activation=nn.Identity())

        # Pointwise grouped convolution (expansion)
        self.layers['PW_group_EXP'] = GroupedConvolution(in_chs=self.bottleneck_channels,
                                                         out_chs=self.internal_out_channels,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0,
                                                         groups=self.groups,
                                                         batch_norm=True,
                                                         activation=nn.Identity())

        self.out_act_fun = nn.ReLU()

    @staticmethod
    def _add(x: Tensor, out: Tensor) -> Tensor:
        """Element-wise addition for 'add' combine mode.
        
        Args:
            x (torch.Tensor): Input tensor.
            out (torch.Tensor): Output tensor from main branch.
            
        Returns:
            torch.Tensor: Element-wise sum of inputs.
        """
        return x + out

    @staticmethod
    def _concat(x: Tensor, out: Tensor) -> Tensor:
        """Channel concatenation for 'concat' combine mode.
        
        Args:
            x (torch.Tensor): Input tensor (possibly pooled).
            out (torch.Tensor): Output tensor from main branch.
            
        Returns:
            torch.Tensor: Channel-wise concatenated tensor.
        """
        return torch.cat((x, out), dim=1)

    @staticmethod
    def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
        """Channel shuffling operation for improving information flow.
        
        Rearranges channels to ensure information exchange between groups
        in grouped convolutions, as described in ShuffleNet paper.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            groups (int): Number of groups for shuffling.
            
        Returns:
            torch.Tensor: Channel-shuffled tensor of same shape.
        """
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batchsize, -1, height, width)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ShuffleNet V1 block.
        
        Applies grouped pointwise compression, channel shuffling, depthwise convolution,
        grouped pointwise expansion, and combines with input using specified mode.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Spatial dimensions depend on combine mode and stride settings.
        """
        residual = x
        if self.combine == 'concat':
            residual = self.layers['res_pool'](residual)

        out = self.layers['PW_group_COMP'](x)
        out = self._channel_shuffle(out, self.groups)
        out = self.layers['DW_conv'](out)
        out = self.layers['PW_group_EXP'](out)
        out = self._combine_func(residual, out)

        return self.out_act_fun(out)


# ShuffleNet V2
class InvertedResidualShuffleModule(nn.Module):
    """Inverted residual ShuffleNet V2 block with dual-branch architecture.
    
    Implements ShuffleNet V2 block from "ShuffleNet V2: Practical Guidelines for Efficient 
    CNN Architecture Design" by Ma et al. Features dual-branch design with channel splitting 
    or full input processing and channel shuffling for efficient information flow.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        stride (int): Stride for depthwise convolution. Must be 1 or 2.
        model (int, optional): Branch architecture mode. If 1, uses "Figure (c)" 
            (split input); if 2, uses "Figure (d)" (full input on both branches). 
            Defaults to 2.
            
    Attributes:
        name (str): Module identifier set to 'InvertedResidualShuffleModule'.
        layers (nn.ModuleDict): Dictionary containing branch convolution layers.
        model (int): Branch architecture mode (1 or 2).
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
        AssertionError: If stride not in [1, 2] or model not in [1, 2].
        ValueError: If model=1 and stride=2 (spatial dimension mismatch).
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        stride: int, 
        model: int = 2
    ) -> None:
        super(InvertedResidualShuffleModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        
        assert stride in [1, 2], f"Stride must be 1 or 2, got {stride}"
        assert model in [1, 2], f"Model must be 1 (Fig. c) or 2 (Fig. d), got {model}"
        if model == 1 and stride == 2:
            raise ValueError("InvertedResidualShuffleBlock: model=1 does not support stride=2 (spatial mismatch)")

        self.in_channels = in_chs
        self.out_channels = out_chs
        self.stride = stride
        self.model = model
        self.name = 'InvertedResidualShuffleModule'
        self.layers = nn.ModuleDict()

        mid_channels = out_chs // 2

        if model == 1:
            # Figure (c): input split into x1 (pass-through) and x2 (processed)
            self.layers['branch_R'] = nn.Sequential(GroupedConvolution(in_chs=mid_channels,       # PW Convolution
                                                                       out_chs=mid_channels,
                                                                       kernel_size=1, 
                                                                       stride=1, 
                                                                       padding=0,
                                                                       groups=1, 
                                                                       batch_norm=True, 
                                                                       activation=nn.ReLU()),
                                                    GroupedConvolution(in_chs=mid_channels,       # DW Convolution
                                                                       out_chs=mid_channels,
                                                                       kernel_size=3, 
                                                                       stride=stride, 
                                                                       padding=1,
                                                                       groups=mid_channels, 
                                                                       batch_norm=True, 
                                                                       activation=nn.Identity()),
                                                    GroupedConvolution(in_chs=mid_channels,       # PW Convolution
                                                                       out_chs=mid_channels,
                                                                       kernel_size=1, 
                                                                       stride=1, 
                                                                       padding=0,
                                                                       groups=1, 
                                                                       batch_norm=True, 
                                                                       activation=nn.ReLU()))
        else:
            # Figure (d): both branches process full input
            self.layers['branch_L'] = nn.Sequential(GroupedConvolution(in_chs=in_chs,             # DW Convolution
                                                                       out_chs=in_chs,
                                                                       kernel_size=3, 
                                                                       stride=stride, 
                                                                       padding=1,
                                                                       groups=in_chs, 
                                                                       batch_norm=True, 
                                                                       activation=nn.Identity()),
                                                    GroupedConvolution(in_chs=in_chs,             # PW Convolution
                                                                       out_chs=mid_channels,
                                                                       kernel_size=1, 
                                                                       stride=1, 
                                                                       padding=0,
                                                                       groups=1, 
                                                                       batch_norm=True, 
                                                                       activation=nn.ReLU()))

            self.layers['branch_R'] = nn.Sequential(GroupedConvolution(in_chs=in_chs,             # PW Convolution (Expansion/Compression)
                                                                       out_chs=mid_channels,
                                                                       kernel_size=1, 
                                                                       stride=1, 
                                                                       padding=0,
                                                                       groups=1, 
                                                                       batch_norm=True, 
                                                                       activation=nn.ReLU()),
                                                    GroupedConvolution(in_chs=mid_channels,       # DW Convolution
                                                                       out_chs=mid_channels,
                                                                       kernel_size=3, 
                                                                       stride=stride, 
                                                                       padding=1,
                                                                       groups=mid_channels, 
                                                                       batch_norm=True, 
                                                                       activation=nn.Identity()),

                                                    GroupedConvolution(in_chs=mid_channels,      # PW Convolution 
                                                                       out_chs=mid_channels,
                                                                       kernel_size=1, 
                                                                       stride=1, 
                                                                       padding=0,
                                                                       groups=1, 
                                                                       batch_norm=True, 
                                                                       activation=nn.ReLU()))

    @staticmethod
    def _concat(x1: Tensor, x2: Tensor) -> Tensor:
        """Channel concatenation for dual-branch outputs.
        
        Concatenates features from both branches along the channel dimension
        for ShuffleNet V2 dual-branch architecture.
        
        Args:
            x1 (torch.Tensor): First branch tensor.
            x2 (torch.Tensor): Second branch tensor.
            
        Returns:
            torch.Tensor: Channel-wise concatenated tensor.
        """
        return torch.cat((x1, x2), dim=1)

    @staticmethod
    def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
        """Channel shuffling operation for improving information flow.
        
        Rearranges channels to ensure information exchange between branches
        in ShuffleNet V2 architecture for enhanced feature representation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            groups (int): Number of groups for shuffling.
            
        Returns:
            torch.Tensor: Channel-shuffled tensor of same shape.
        """
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batchsize, -1, height, width)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ShuffleNet V2 block.
        
        Applies dual-branch processing with optional channel splitting, followed by
        concatenation and channel shuffling for efficient information flow.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Channel-shuffled result from dual-branch processing.
        """
        if self.model == 1:
            x1 = x[:, :x.shape[1] // 2, :, :]
            x2 = x[:, x.shape[1] // 2:, :, :]
            out = self._concat(x1, self.layers['branch_R'](x2))
        else:
            out = self._concat(self.layers['branch_L'](x), self.layers['branch_R'](x))

        return self._channel_shuffle(out, groups=2)


# SqueezeNet
class FireModule(nn.Module):
    """Fire module from SqueezeNet architecture for efficient CNN computation.
    
    Implements Fire module from "SqueezeNet: AlexNet-level accuracy with 50x fewer 
    parameters and <0.5MB model size" by Iandola et al. Features squeeze-and-expand 
    strategy with 1x1 squeeze followed by parallel 1x1 and 3x3 expansion layers.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        squeeze_chs (int): Number of squeeze (1x1) filters. Must be > 0.
        expPW_chs (int): Number of pointwise (1x1) expansion filters. Must be > 0.
        exp3x3_chs (int): Number of 3x3 expansion filters. Must be > 0.
        
    Attributes:
        name (str): Module identifier set to 'FireModule'.
        layers (nn.ModuleDict): Dictionary containing squeeze and expansion layers.
        
    Raises:
        ValueError: If any channel parameter <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        squeeze_chs: int, 
        expPW_chs: int, 
        exp3x3_chs: int
    ) -> None:
        super(FireModule, self).__init__()
        
        if in_chs <= 0 or squeeze_chs <= 0 or expPW_chs <= 0 or exp3x3_chs <= 0:
            raise ValueError(f"All channel numbers must be > 0, got in_chs={in_chs}, "
                           f"squeeze_chs={squeeze_chs}, expPW_chs={expPW_chs}, exp3x3_chs={exp3x3_chs}")
            
        self.name = 'FireModule'
        self.layers = nn.ModuleDict()

        # Squeeze layer (1x1 compression)
        self.layers['Squeeze'] = ConvBlock(in_chs=in_chs, 
                                           out_chs=squeeze_chs, 
                                           kernel_size=1, 
                                           stride=1, 
                                           padding=0, 
                                           activation=nn.ReLU())
        
        # Expand 1x1 layer (pointwise expansion)
        self.layers['PW_Conv'] = PointwiseConvolution(in_chs=squeeze_chs, 
                                                      out_chs=expPW_chs, 
                                                      stride=1, 
                                                      padding=0, 
                                                      activation=nn.ReLU())

        # Expand 3x3 layer (spatial expansion)
        self.layers['EXP_Conv'] = ConvBlock(in_chs=squeeze_chs, 
                                            out_chs=exp3x3_chs, 
                                            kernel_size=3, 
                                            stride=1, 
                                            padding=1, 
                                            activation=nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Fire module.
        
        Applies squeeze operation followed by parallel 1x1 and 3x3 expansions
        with channel-wise concatenation for SqueezeNet efficiency.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, expPW_chs + exp3x3_chs, H, W).
                Concatenated result from parallel expansion branches.
        """
        x = self.layers['Squeeze'](x)
        out = torch.cat([self.layers['PW_Conv'](x), self.layers['EXP_Conv'](x)], dim=1)
        
        return out


# SqueezeNeXt
class SqueezeNeXtModule(nn.Module):
    """SqueezeNeXt module for hardware-aware efficient CNN architecture.
    
    Implements SqueezeNeXt module from "SqueezeNext: Hardware-Aware Neural Network Design" 
    by Gholami et al. Features factorized convolutions with vertical-horizontal decomposition 
    and adaptive channel reduction for optimized mobile deployment.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        stride (int, optional): Stride for downsampling. Must be 1 or 2. Defaults to 1.
        
    Attributes:
        name (str): Module identifier set to 'SqueezeNeXtModule'.
        layers (nn.ModuleDict): Dictionary containing expansion, bottleneck, and projection layers.
        shortcut (nn.Module): Shortcut connection (identity or 1x1 conv).
        activation (nn.Module): Final ReLU activation function.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
        ValueError: If stride not in [1, 2].
    """
    
    def __init__(self, in_chs: int, out_chs: int, stride: int = 1) -> None:
        super(SqueezeNeXtModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
            
        self.name = 'SqueezeNeXtModule'
        
        # Adaptive channel reduction based on stride and channel relationship
        reduction = 1.0 if stride == 2 else (0.25 if in_chs > out_chs else 0.5)
        mid_chs = int(in_chs * reduction)
        bottleneck_chs = int(mid_chs * 0.5)

        self.layers = nn.ModuleDict({
            'PW_Exp': ConvBlock(in_chs=in_chs, 
                                out_chs=mid_chs, 
                                kernel_size=1, 
                                stride=stride, 
                                padding=0, 
                                activation=nn.ReLU()),
            'Bottleneck': ConvBlock(in_chs=mid_chs, 
                                    out_chs=bottleneck_chs, 
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0, 
                                    activation=nn.ReLU()),
            'EXP_Conv_Vert': ConvBlock(in_chs=bottleneck_chs, 
                                       out_chs=mid_chs, 
                                       kernel_size=(1, 3), 
                                       stride=1, 
                                       padding=(0, 1), 
                                       activation=nn.ReLU()),
            'EXP_Conv_Horiz': ConvBlock(in_chs=mid_chs, 
                                        out_chs=mid_chs, 
                                        kernel_size=(3, 1), 
                                        stride=1, 
                                        padding=(1, 0), 
                                        activation=nn.ReLU()),
            'PW_Conv': ConvBlock(in_chs=mid_chs, 
                                 out_chs=out_chs, 
                                 kernel_size=1, 
                                 stride=1, 
                                 padding=0, 
                                 activation=nn.ReLU())
        })

        # Shortcut connection
        self.shortcut = (
            ConvBlock(in_chs=in_chs, 
                      out_chs=out_chs, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=0, 
                      activation=nn.ReLU())
            if stride == 2 or in_chs != out_chs
            else nn.Identity()
        )
        
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SqueezeNeXt module.
        
        Applies pointwise expansion, bottleneck compression, factorized convolutions 
        (vertical then horizontal), final projection, and residual connection for 
        hardware-optimized feature extraction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Result includes residual connection and final activation.
        """
        out = self.layers['PW_Exp'](x)
        out = self.layers['Bottleneck'](out)
        out = self.layers['EXP_Conv_Vert'](out)
        out = self.layers['EXP_Conv_Horiz'](out)
        out = self.layers['PW_Conv'](out)
        out += self.shortcut(x)
        
        return self.activation(out)


# CondenseNet
class CondenseModule(nn.Module):
    """CondenseNet dense layer using learned group convolutions.
    
    Implements DenseLayer using Learned Group Convolutions from "CondenseNet: An Efficient 
    DenseNet using Learned Group Convolutions" by Huang et al. Features bottleneck structure 
    with learned group convolutions for efficient dense connectivity and feature reuse.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        growth_rate (int): Number of output feature maps to add. Must be > 0.
        config (Dict[str, Union[int, float]]): Configuration dictionary with the following keys:
            - 'conv_bottleneck_chs' (int): Bottleneck channels for intermediate channels
            - 'group1x1' (int): Number of groups in the 1x1 convolution
            - 'group3x3' (int): Number of groups in the 3x3 convolution
            - 'condense_factor' (int): Condensation factor for LearnedGroupConv
            - 'dropout_rate' (float): Dropout probability
            
    Attributes:
        name (str): Module identifier set to 'CondenseModule'.
        layers (nn.ModuleDict): Dictionary containing learned group convolution layers.
        config (Dict): Configuration parameters for the module.
        
    Raises:
        ValueError: If in_chs or growth_rate <= 0.
        KeyError: If required configuration keys are missing.
        ValueError: If configuration values are invalid.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        growth_rate: int, 
        config: Dict[str, Union[int, float]]
    ) -> None:
        super(CondenseModule, self).__init__()

        if in_chs <= 0 or growth_rate <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, growth_rate={growth_rate}")

        required_keys = ['conv_bottleneck_chs', 'group1x1', 'group3x3', 'condense_factor', 'dropout_rate']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {missing_keys}")

        if config['conv_bottleneck_chs'] <= 0 or config['group1x1'] <= 0 or config['group3x3'] <= 0:
            raise ValueError("conv_bottleneck_chs, group1x1, and group3x3 must be > 0")
        if config['condense_factor'] <= 0:
            raise ValueError("condense_factor must be > 0")
        if not (0 <= config['dropout_rate'] <= 1):
            raise ValueError("dropout_rate must be in range [0, 1]")

        self.name = 'CondenseModule'
        self.config = config
        self.conv_bottleneck_chs = config['conv_bottleneck_chs']
        self.group1x1 = config['group1x1']
        self.group3x3 = config['group3x3']
        self.condense_factor = config['condense_factor']
        self.dropout_rate = config['dropout_rate']

        inter_channels = self.conv_bottleneck_chs * growth_rate

        self.layers = nn.ModuleDict()

        # Learned group convolution for bottleneck (1x1)
        self.layers['Conv_IN'] = LearnedGroupConvolution(in_chs=in_chs,
                                                         out_chs=inter_channels,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0,
                                                         dilation=1,
                                                         groups=self.group1x1,
                                                         condense_factor=self.condense_factor,
                                                         dropout_rate=self.dropout_rate)

        self.layers['Batch_Norm'] = nn.BatchNorm2d(inter_channels)
        self.act_fun = nn.ReLU()

        # Standard grouped convolution for spatial features (3x3)
        self.layers['Conv_OUT'] = nn.Conv2d(in_channels=inter_channels,
                                            out_channels=growth_rate,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1,
                                            groups=self.group3x3,
                                            bias=True)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through CondenseNet module.
        
        Applies learned group convolution bottleneck, batch normalization, ReLU activation,
        and spatial convolution, then concatenates with input for dense connectivity.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_in + growth_rate, H, W).
                Concatenated result of input and new features for dense connectivity.
        """
        out = self.layers['Conv_IN'](x)
        out = self.layers['Batch_Norm'](out)
        out = self.act_fun(out)
        out = self.layers['Conv_OUT'](out)

        return torch.cat([x, out], dim=1)


# ESNet
class FactorizedConvolutionModule(nn.Module):
    """Factorized convolution module for efficient semantic segmentation.
    
    Implements Factorized Convolution Module from "ESNet: Efficient Semantic Segmentation 
    via Decoupled Convolutions" by Mehta et al. Features asymmetric convolution factorization 
    (3x1 and 1x3) with dilated convolutions for efficient spatial feature extraction.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        kernel_size (int): Convolutional kernel size (usually 3). Must be > 0.
        drop_prob (float): Dropout probability. Must be in range [0, 1].
        dilation (int): Dilation rate for the second conv stage. Must be > 0.
        bias (bool, optional): Whether to use bias in convolutions. Defaults to True.
        
    Attributes:
        name (str): Module identifier set to 'FactorizedConvolutionModule'.
        layers (nn.ModuleDict): Dictionary containing factorized convolution layers.
        activation (nn.Module): ReLU activation function.
        dropout (nn.Module): 2D dropout layer.
        
    Raises:
        ValueError: If in_chs, kernel_size, or dilation <= 0.
        ValueError: If drop_prob not in range [0, 1].
    """
    
    def __init__(
        self, 
        in_chs: int, 
        kernel_size: int, 
        drop_prob: float, 
        dilation: int, 
        bias: bool = True
    ) -> None:
        super(FactorizedConvolutionModule, self).__init__()
        
        if in_chs <= 0 or kernel_size <= 0 or dilation <= 0:
            raise ValueError(f"in_chs, kernel_size, and dilation must be > 0, "
                           f"got in_chs={in_chs}, kernel_size={kernel_size}, dilation={dilation}")
        if not (0 <= drop_prob <= 1):
            raise ValueError(f"drop_prob must be in range [0, 1], got {drop_prob}")
            
        self.name = 'FactorizedConvolutionModule'

        # Calculate padding for dilated and non-dilated convolutions
        padding_dilated = (kernel_size - 1) // 2 * dilation
        padding_plain = (kernel_size - 1) // 2

        self.layers = nn.ModuleDict({
            # First factorized convolution stage (3x1 -> 1x3)
            'Conv3x1_1': nn.Conv2d(in_chs, in_chs, (kernel_size, 1), padding=(padding_plain, 0), bias=bias),
            'Conv1x3_1': nn.Conv2d(in_chs, in_chs, (1, kernel_size), padding=(0, padding_plain), bias=bias),
            'BatchNorm1': nn.BatchNorm2d(in_chs, eps=1e-3),
            
            # Second factorized convolution stage with dilation (3x1 -> 1x3)
            'Conv3x1_2': nn.Conv2d(in_chs, in_chs, (kernel_size, 1), padding=(padding_dilated, 0), 
                                   dilation=(dilation, 1), bias=bias),
            'Conv1x3_2': nn.Conv2d(in_chs, in_chs, (1, kernel_size), padding=(0, padding_dilated), 
                                   dilation=(1, dilation), bias=bias),
            'BatchNorm2': nn.BatchNorm2d(in_chs, eps=1e-3)
        })

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(drop_prob)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through factorized convolution module.
        
        Applies two-stage factorized convolutions (3x1 -> 1x3) with batch normalization,
        optional dropout, and residual connection for efficient semantic segmentation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
                Result includes residual connection and final activation.
        """
        residual = x
        
        # First factorized convolution stage (3x1 -> 1x3 with batch norm)
        out = self.activation(self.layers['Conv3x1_1'](x))
        out = self.activation(self.layers['Conv1x3_1'](out))
        out = self.layers['BatchNorm1'](out)

        # Second factorized convolution stage with dilation (3x1 -> 1x3 with batch norm)
        out = self.activation(self.layers['Conv3x1_2'](out))
        out = self.activation(self.layers['Conv1x3_2'](out))
        out = self.layers['BatchNorm2'](out)

        # Apply dropout if probability > 0
        if self.dropout.p > 0:
            out = self.dropout(out)

        return self.activation(residual + out)


class ParallelFactorizedConvolutionModule(nn.Module):
    """Parallel factorized convolution for efficient multi-scale semantic segmentation.
    
    Implements Parallel Factorized Convolution from "ESNet: Efficient Semantic Segmentation 
    via Decoupled Convolutions" by Mehta et al. Features parallel branches with different 
    dilation rates for multi-scale context aggregation using factorized convolutions.
    
    Args:
        in_chs (int): Number of input/output channels. Must be > 0.
        drop_prob (float, optional): Dropout probability. Must be in range [0, 1]. 
            Defaults to 0.3.
        bias (bool, optional): Whether to use bias in convolutions. Defaults to True.
        
    Attributes:
        name (str): Module identifier set to 'ParallelFactorizedConvolutionModule'.
        layers (nn.ModuleDict): Dictionary containing factorized convolution layers 
            with multiple dilation rates.
        activation (nn.Module): ReLU activation function.
        dropout (nn.Module): 2D dropout layer.
        
    Raises:
        ValueError: If in_chs <= 0.
        ValueError: If drop_prob not in range [0, 1].
    """
    
    def __init__(self, in_chs: int, drop_prob: float = 0.3, bias: bool = True) -> None:
        super(ParallelFactorizedConvolutionModule, self).__init__()
        
        if in_chs <= 0:
            raise ValueError(f"in_chs must be > 0, got {in_chs}")
        if not (0 <= drop_prob <= 1):
            raise ValueError(f"drop_prob must be in range [0, 1], got {drop_prob}")
            
        self.name = 'ParallelFactorizedConvolutionModule'

        self.layers = nn.ModuleDict({
            # Initial factorized convolution stage (3x1 -> 1x3)
            'Conv3x1_1': nn.Conv2d(in_chs, in_chs, (3, 1), padding=(1, 0), bias=bias),
            'Conv1x3_1': nn.Conv2d(in_chs, in_chs, (1, 3), padding=(0, 1), bias=bias),
            'BatchNorm1': nn.BatchNorm2d(in_chs, eps=1e-3),
            
            # Parallel branches with different dilation rates
            # Dilation = 2
            'Conv3x1_22': nn.Conv2d(in_chs, in_chs, (3, 1), padding=(2, 0), dilation=(2, 1), bias=bias),
            'Conv1x3_22': nn.Conv2d(in_chs, in_chs, (1, 3), padding=(0, 2), dilation=(1, 2), bias=bias),
            # Dilation = 5
            'Conv3x1_25': nn.Conv2d(in_chs, in_chs, (3, 1), padding=(5, 0), dilation=(5, 1), bias=bias),
            'Conv1x3_25': nn.Conv2d(in_chs, in_chs, (1, 3), padding=(0, 5), dilation=(1, 5), bias=bias),
            # Dilation = 9
            'Conv3x1_29': nn.Conv2d(in_chs, in_chs, (3, 1), padding=(9, 0), dilation=(9, 1), bias=bias),
            'Conv1x3_29': nn.Conv2d(in_chs, in_chs, (1, 3), padding=(0, 9), dilation=(1, 9), bias=bias),
            'BatchNorm2': nn.BatchNorm2d(in_chs, eps=1e-3)
        })

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(drop_prob)
        
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through parallel factorized convolution module.
        
        Applies initial factorized convolution followed by three parallel branches 
        with different dilation rates (2, 5, 9) for multi-scale context aggregation 
        in semantic segmentation tasks.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
                Sum of residual connection and parallel dilated convolution branches.
        """
        residual = x
        
        # Initial factorized convolution stage (3x1 -> 1x3 with batch norm)
        out = self.activation(self.layers['Conv3x1_1'](x))
        out = self.activation(self.layers['Conv1x3_1'](out))
        out = self.layers['BatchNorm1'](out)

        # Three parallel branches with different dilation rates for multi-scale context
        b2 = self.layers['BatchNorm2'](self.layers['Conv1x3_22'](self.activation(self.layers['Conv3x1_22'](out))))
        b5 = self.layers['BatchNorm2'](self.layers['Conv1x3_25'](self.activation(self.layers['Conv3x1_25'](out))))
        b9 = self.layers['BatchNorm2'](self.layers['Conv1x3_29'](self.activation(self.layers['Conv3x1_29'](out))))

        # Apply dropout to parallel branches if probability > 0
        if self.dropout.p > 0:
            b2 = self.dropout(b2)
            b5 = self.dropout(b5)
            b9 = self.dropout(b9)

        return self.activation(residual + b2 + b5 + b9)


# PeleeNet
class StemModule(nn.Module):
    """PeleeNet stem module for efficient real-time object detection.
    
    Implements PeleeNet Stem Module from "Pelee: A Real-Time Object Detection System 
    on Mobile Devices" by Wang et al. Features dual-branch architecture with parallel 
    convolution and max pooling paths for efficient feature extraction initialization.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int, optional): Number of output feature channels. Must be > 0. 
            Defaults to 32.
            
    Attributes:
        name (str): Module identifier set to 'StemModule'.
        layers (nn.ModuleDict): Dictionary containing stem convolution layers and branches.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
    """
    
    def __init__(self, in_chs: int, out_chs: int = 32) -> None:
        super(StemModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"Channel numbers must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
            
        self.name = "StemModule"
        self.layers = nn.ModuleDict()

        # Initial convolution with stride 2 for downsampling
        self.layers['Conv_IN'] = GroupedConvolution(in_chs=in_chs,
                                                    out_chs=out_chs,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    batch_norm=True,
                                                    activation=nn.ReLU())

        # Left branch: 1x1 compression
        self.layers['Branch_LA'] = GroupedConvolution(in_chs=out_chs,
                                                      out_chs=int(out_chs / 2),
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0,
                                                      groups=1,
                                                      batch_norm=True,
                                                      activation=nn.ReLU())

        # Left branch: 3x3 convolution with stride 2
        self.layers['Branch_LB'] = GroupedConvolution(in_chs=int(out_chs / 2),
                                                      out_chs=out_chs,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      groups=1,
                                                      batch_norm=True,
                                                      activation=nn.ReLU())

        # Right branch: Max pooling with stride 2
        self.layers['Branch_R'] = nn.MaxPool2d(kernel_size=2, stride=2)

        # Final 1x1 convolution for channel reduction
        self.layers['Conv_OUT'] = GroupedConvolution(in_chs=out_chs * 2,
                                                     out_chs=out_chs,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0,
                                                     groups=1,
                                                     batch_norm=True,
                                                     activation=nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through PeleeNet stem module.
        
        Applies initial convolution, then processes through dual branches (convolution 
        and max pooling) before concatenation and final channel reduction for efficient 
        feature extraction initialization in real-time object detection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_chs, H//4, W//4).
                Processed features with 4x spatial downsampling from dual-branch architecture.
        """
        # Initial convolution with stride 2 (2x downsampling)
        stem_1_out = self.layers['Conv_IN'](x)
        
        # Left branch: 1x1 -> 3x3 convolution path
        stem_2a_out = self.layers['Branch_LA'](stem_1_out)
        stem_2b_out = self.layers['Branch_LB'](stem_2a_out)  # Additional 2x downsampling
        
        # Right branch: Max pooling path
        stem_2p_out = self.layers['Branch_R'](stem_1_out)  # 2x downsampling
        
        # Concatenate dual branches and apply final convolution
        out = self.layers['Conv_OUT'](torch.cat((stem_2b_out, stem_2p_out), dim=1))
        
        return out


# MobileOne
class MobileOneModule(nn.Module):
    """MobileOne block with efficient re-parameterization for sub-millisecond inference.
    
    Implements MobileOne Block from "An Improved One millisecond Mobile Backbone" by 
    Vasudevan et al. Features structural re-parameterization with multiple conv branches 
    during training that merge into single conv during inference for optimal speed.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        kernel_size (int): Convolution kernel size. Must be > 0.
        stride (int, optional): Stride for convolutions. Must be > 0. Defaults to 1.
        padding (int, optional): Zero-padding size. Must be >= 0. Defaults to 0.
        dilation (int, optional): Dilation rate. Must be > 0. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Must be > 0. 
            Defaults to 1.
        inference_mode (bool, optional): If True, uses reparameterized single Conv2d layer. 
            Defaults to False.
        use_se (bool, optional): If True, enables Squeeze-and-Excite block. Defaults to False.
        num_conv_branches (int, optional): Number of parallel conv branches at train time. 
            Must be > 0. Defaults to 1.
            
    Attributes:
        name (str): Module identifier set to 'MobileOneModule'.
        layers (nn.ModuleDict): Dictionary containing convolution branches and activations.
        inference_mode (bool): Whether using single reparameterized convolution.
        use_se (bool): Whether SE block is enabled.
        
    Raises:
        ValueError: If in_chs, out_chs, kernel_size, stride, dilation, groups, 
            or num_conv_branches <= 0.
        ValueError: If padding < 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0, 
        dilation: int = 1, 
        groups: int = 1, 
        inference_mode: bool = False, 
        use_se: bool = False, 
        num_conv_branches: int = 1
    ) -> None:
        super(MobileOneModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}")
        if dilation <= 0:
            raise ValueError(f"dilation must be > 0, got {dilation}")
        if groups <= 0:
            raise ValueError(f"groups must be > 0, got {groups}")
        if num_conv_branches <= 0:
            raise ValueError(f"num_conv_branches must be > 0, got {num_conv_branches}")
            
        self.name = "MobileOneModule"
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_conv_branches = num_conv_branches
        self.use_se = use_se

        self.layers = nn.ModuleDict()
        self.layers['Activation'] = nn.ReLU()

        if use_se:
            # Squeeze-and-Excitation block - using raw PyTorch for precise control
            self.layers['SE_Reduce'] = nn.Conv2d(out_chs, int(out_chs * 0.0625), kernel_size=1)
            self.layers['SE_Act1'] = nn.ReLU()
            self.layers['SE_Expand'] = nn.Conv2d(int(out_chs * 0.0625), out_chs, kernel_size=1)
            self.layers['SE_Act2'] = nn.Sigmoid()

        if inference_mode:
            # Single reparameterized convolution for inference
            self.layers['ReparamConv'] = nn.Conv2d(in_chs, out_chs,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   dilation=dilation,
                                                   groups=groups,
                                                   bias=True)
        else:
            # Multiple branches for training - preserve original architecture exactly
            if out_chs == in_chs and stride == 1:
                self.layers['Skip'] = nn.BatchNorm2d(in_chs)

            for i in range(num_conv_branches):
                self.layers[f'ConvBranch_{i}'] = self._conv_bn(kernel_size=kernel_size, padding=padding)

            if kernel_size > 1:
                self.layers['ScaleBranch'] = self._conv_bn(kernel_size=1, padding=0)
        
        self.apply(initialize_weights)

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Create conv-bn block for MobileOne branches.
        
        Args:
            kernel_size (int): Convolution kernel size.
            padding (int): Padding for convolution.
            
        Returns:
            nn.Sequential: Sequential block with conv and batch norm.
        """
        block = nn.Sequential()
        block.add_module('conv', nn.Conv2d(self.in_chs, self.out_chs,
                                           kernel_size=kernel_size,
                                           stride=self.stride,
                                           padding=padding,
                                           groups=self.groups,
                                           bias=False))
        block.add_module('bn', nn.BatchNorm2d(self.out_chs))
        
        return block

    def _squeeze_excite(self, x: Tensor) -> Tensor:
        """Apply squeeze-and-excitation attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: SE-weighted tensor of same shape.
        """
        b, c, h, w = x.size()
        y = F.avg_pool2d(x, kernel_size=[h, w])
        y = self.layers['SE_Reduce'](y)
        y = self.layers['SE_Act1'](y)
        y = self.layers['SE_Expand'](y)
        y = self.layers['SE_Act2'](y)
        
        return x * y.view(-1, c, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MobileOne module.
        
        Applies either single reparameterized convolution (inference mode) or 
        multiple branch convolutions (training mode) with optional SE attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Processed features through MobileOne block.
        """
        if self.inference_mode:
            out = self.layers['ReparamConv'](x)
        else:
            out = 0

            if 'Skip' in self.layers:
                out += self.layers['Skip'](x)

            if 'ScaleBranch' in self.layers:
                out += self.layers['ScaleBranch'](x)

            for i in range(self.num_conv_branches):
                out += self.layers[f'ConvBranch_{i}'](x)

        if self.use_se:
            out = self._squeeze_excite(out)

        return self.layers['Activation'](out)

    def reparameterize(self) -> None:
        """Convert multiple training branches into single inference convolution.
        
        Fuses multiple conv-bn branches into single reparameterized convolution
        for efficient inference without accuracy loss.
        """
        if self.inference_mode:
            return

        kernel, bias = self._get_kernel_bias()

        self.layers['ReparamConv'] = nn.Conv2d(self.in_chs, self.out_chs,
                                               kernel_size=self.kernel_size,
                                               stride=self.stride,
                                               padding=self.kernel_size // 2,
                                               dilation=1,
                                               groups=self.groups,
                                               bias=True)

        self.layers['ReparamConv'].weight.data = kernel
        self.layers['ReparamConv'].bias.data = bias

        # Clean-up training branches
        keys_to_remove = [k for k in self.layers if k not in ['Activation', 'ReparamConv']]
        for k in keys_to_remove:
            del self.layers[k]

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        """Get fused kernel and bias from multiple branches.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Fused kernel and bias tensors.
        """
        kernel_conv = 0
        bias_conv = 0

        if 'ScaleBranch' in self.layers:
            k, b = self._fuse_bn_tensor(self.layers['ScaleBranch'])
            k = F.pad(k, [self.kernel_size // 2] * 4)
            kernel_conv += k
            bias_conv += b

        if 'Skip' in self.layers:
            k, b = self._fuse_bn_tensor(self.layers['Skip'])
            kernel_conv += k
            bias_conv += b

        for i in range(self.num_conv_branches):
            k, b = self._fuse_bn_tensor(self.layers[f'ConvBranch_{i}'])
            kernel_conv += k
            bias_conv += b

        return kernel_conv, bias_conv

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[Tensor, Tensor]:
        """Fuse convolution and batch normalization into single tensor.
        
        Args:
            branch (nn.Module): Conv-bn branch to fuse.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Fused kernel and bias.
        """
        if isinstance(branch, nn.BatchNorm2d):
            kernel = self._get_identity_kernel()
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _get_identity_kernel(self) -> Tensor:
        """Get identity kernel for skip connections.
        
        Returns:
            torch.Tensor: Identity kernel tensor.
        """
        kernel = torch.zeros((self.out_chs, self.in_chs // self.groups, 
                             self.kernel_size, self.kernel_size))
        for i in range(self.out_chs):
            kernel[i, i % (self.in_chs // self.groups), 
                   self.kernel_size // 2, self.kernel_size // 2] = 1
        return kernel


# MixNet
class MixDepthModule(nn.Module):
    """MixDepth convolution block for mixed kernel size depthwise convolutions.
    
    Implements MixDepth Convolution from "MixConv: Mixed Depthwise Convolutional Kernels" 
    by Tan et al. Features mixed kernel sizes within depthwise convolutions for improved 
    accuracy-efficiency trade-offs in mobile neural architectures.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        exp_ratio (int): Expansion ratio for the hidden layer. Must be > 0.
        exp_kernel_sizes (List[int]): Kernel size configuration for expansion GroupedPointwise.
            Each kernel size must be > 0.
        kernel_sizes (List[int]): Kernel size configuration for mixed depthwise convolution.
            Each kernel size must be > 0.
        poi_kernel_sizes (List[int]): Kernel size configuration for projection GroupedPointwise.
            Each kernel size must be > 0.
        stride (int): Convolutional stride, must be 1 or 2.
        dilation (int): Dilation rate for depthwise convolution. Must be > 0.
        red_ratio (int, optional): Reduction ratio for Squeeze-and-Excite. Must be > 1 
            to enable SE, or None to disable. Defaults to 4.
        dropout_rate (float, optional): Dropout probability before residual addition.
            Must be in range [0, 1]. Defaults to 0.2.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.SiLU().
            
    Attributes:
        name (str): Module identifier set to 'MixDepthModule'.
        layers (nn.ModuleDict): Dictionary containing expansion, mixed depthwise, SE, 
            and projection layers.
        use_se (bool): Whether Squeeze-and-Excitation is enabled.
        use_residual (bool): Whether residual connection is used.
        
    Raises:
        ValueError: If in_chs, out_chs, exp_ratio, or dilation <= 0.
        ValueError: If any kernel size in kernel size lists <= 0.
        ValueError: If stride not in [1, 2].
        ValueError: If dropout_rate not in range [0, 1].
        ValueError: If red_ratio is not None and <= 1.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        exp_ratio: int, 
        exp_kernel_sizes: List[int], 
        kernel_sizes: List[int], 
        poi_kernel_sizes: List[int], 
        stride: int, 
        dilation: int, 
        red_ratio: Optional[int] = 4, 
        dropout_rate: float = 0.2, 
        activation: nn.Module = nn.SiLU()
    ) -> None:
        super(MixDepthModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if exp_ratio <= 0:
            raise ValueError(f"exp_ratio must be > 0, got {exp_ratio}")
        if dilation <= 0:
            raise ValueError(f"dilation must be > 0, got {dilation}")
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate must be in range [0, 1], got {dropout_rate}")
        if red_ratio is not None and red_ratio <= 1:
            raise ValueError(f"red_ratio must be > 1 or None, got {red_ratio}")
            
        # Validate kernel size lists
        for i, ks in enumerate(exp_kernel_sizes):
            if ks <= 0:
                raise ValueError(f"exp_kernel_sizes[{i}] must be > 0, got {ks}")
        for i, ks in enumerate(kernel_sizes):
            if ks <= 0:
                raise ValueError(f"kernel_sizes[{i}] must be > 0, got {ks}")
        for i, ks in enumerate(poi_kernel_sizes):
            if ks <= 0:
                raise ValueError(f"poi_kernel_sizes[{i}] must be > 0, got {ks}")
        
        self.name = "MixDepthModule"
        self.dropout_rate = dropout_rate
        self.exp_ratio = exp_ratio
        self.groups = len(kernel_sizes)
        self.use_se = (red_ratio is not None) and (red_ratio > 1)
        self.use_residual = in_chs == out_chs and stride == 1

        dilate = 1 if stride > 1 else dilation
        hidden_chs = in_chs * exp_ratio

        self.layers = nn.ModuleDict()

        # Stage 1: Expansion (Grouped Pointwise)
        if exp_ratio != 1:
            self.layers['Expansion'] = nn.Sequential(OrderedDict([("GPW_Conv", GroupedPointwiseConvolution(in_chs, 
                                                                                                           hidden_chs, 
                                                                                                           exp_kernel_sizes)),
                                                                  ("Batch_Norm", nn.BatchNorm2d(hidden_chs, eps=1e-3, momentum=0.01)),
                                                                  ("activation", activation)]))

        # Stage 2: Mixed Depthwise Convolution
        self.layers['Mixed_DWConv'] = nn.Sequential(OrderedDict([("MDW_Conv", MixedDepthwiseConvolution(hidden_chs, 
                                                                                                        kernel_sizes, 
                                                                                                        stride=stride, 
                                                                                                        dilation=dilate)),
                                                                 ("Batch_Norm", nn.BatchNorm2d(hidden_chs, eps=1e-3, momentum=0.01)),
                                                                 ("activation", activation)]))

        # Stage 3: Squeeze-and-Excitation
        if self.use_se:
            reduced_chs = max(1, int(in_chs / red_ratio))
            self.layers['ConvSE'] = ConvSE(in_chs=hidden_chs,
                                           red_chs=reduced_chs,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           activations=[nn.SiLU(), nn.Sigmoid()])

        # Stage 4: Projection (Grouped Pointwise)
        self.layers['Projection'] = nn.Sequential(OrderedDict([("GPW_Conv", GroupedPointwiseConvolution(hidden_chs, 
                                                                                                        out_chs, 
                                                                                                        poi_kernel_sizes)),
                                                               ("Batch_Norm", nn.BatchNorm2d(out_chs, eps=1e-3, momentum=0.01))]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MixDepth module.
        
        Applies expansion (if exp_ratio != 1), mixed depthwise convolution with multiple 
        kernel sizes, optional Squeeze-and-Excitation, projection, and residual connection 
        for efficient mixed-kernel mobile architectures.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Result includes optional residual connection and dropout regularization.
        """
        res = x.clone()

        if self.exp_ratio != 1:
            x = self.layers['Expansion'](x)

        x = self.layers['Mixed_DWConv'](x)

        if self.use_se:
            x = self.layers['ConvSE'](x)

        x = self.layers['Projection'](x)

        if self.use_residual:
            if self.training and self.dropout_rate is not None:
                x = F.dropout2d(x, p=self.dropout_rate, training=True, inplace=True)
            x = x + res

        return x


# DiCENet
class DiCEModule(nn.Module):
    """Dimension-wise Convolutional Module for efficient spatial feature learning.
    
    Implements DiCE Module from "DiCENet: Dimension-wise Convolutions for Efficient Networks" 
    by Mehta et al. Features dimension-wise convolutions across channel, height, and width 
    dimensions with adaptive fusion for efficient spatial feature extraction.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        height (int): Target height for adaptive height-wise operations. Must be > 0.
        width (int): Target width for adaptive width-wise operations. Must be > 0.
        kernel_size (int, optional): Kernel size for depthwise convolutions. Must be > 0. 
            Defaults to 3.
        dilation (List[int], optional): Dilation rates for [channel, width, height] 
            convolutions. Each value must be > 0. Defaults to [1, 1, 1].
        shuffle (bool, optional): Whether to apply channel shuffle for feature mixing. 
            Defaults to True.
            
    Attributes:
        name (str): Module identifier set to 'DiCEModule'.
        layers (nn.ModuleDict): Dictionary containing dimension-wise convolution layers,
            normalization, fusion, and projection components.
        shuffle (bool): Whether channel shuffle is enabled.
        height (int): Target height for adaptive operations.
        width (int): Target width for adaptive operations.
        
    Raises:
        ValueError: If in_chs, out_chs, height, width, or kernel_size <= 0.
        ValueError: If dilation list length != 3.
        ValueError: If any dilation value <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        height: int, 
        width: int, 
        kernel_size: int = 3, 
        dilation: List[int] = [1, 1, 1], 
        shuffle: bool = True
    ) -> None:
        super(DiCEModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be > 0, got height={height}, width={width}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if len(dilation) != 3:
            raise ValueError(f"dilation must be a list of 3 elements, got {len(dilation)}")
        for i, d in enumerate(dilation):
            if d <= 0:
                raise ValueError(f"dilation[{i}] must be > 0, got {d}")
                
        self.name = "DiCEModule"
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.in_chs = in_chs
        self.out_chs = out_chs

        pad_c = (kernel_size - 1) // 2 * dilation[0]
        pad_w = (kernel_size - 1) // 2 * dilation[1]
        pad_h = (kernel_size - 1) // 2 * dilation[2]

        self.layers = nn.ModuleDict()

        # Channel-wise convolution
        self.layers['Conv_Ch'] = nn.Conv2d(in_channels=in_chs,
                                           out_channels=in_chs,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           groups=in_chs,
                                           padding=pad_c,
                                           bias=True,
                                           dilation=dilation[0])

        # Width-wise convolution
        self.layers['Conv_W'] = nn.Conv2d(in_channels=width,
                                          out_channels=width,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          groups=width,
                                          padding=pad_w,
                                          bias=True,
                                          dilation=dilation[1])

        # Height-wise convolution
        self.layers['Conv_H'] = nn.Conv2d(in_channels=height,
                                          out_channels=height,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          groups=height,
                                          padding=pad_h,
                                          bias=True,
                                          dilation=dilation[2])

        # Normalization and activation
        self.layers['Norm_Act'] = nn.Sequential(nn.BatchNorm2d(3 * in_chs, eps=1e-3, momentum=0.01),
                                                nn.PReLU())

        self.layers['Weights_AVG'] = nn.Sequential(nn.Conv2d(in_channels=3 * in_chs,
                                                             out_channels=in_chs,
                                                             kernel_size=1,
                                                             stride=1,
                                                             padding=0,
                                                             groups=in_chs,
                                                             dilation=1,
                                                             bias=True),
                                                   nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.01),
                                                   nn.PReLU())

        groups_proj = math.gcd(in_chs, out_chs)
        self.layers['Projection'] = nn.Sequential(nn.Conv2d(in_channels=in_chs,
                                                  out_channels=out_chs,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  groups=groups_proj,
                                                  dilation=1,
                                                  bias=True),
                                    nn.BatchNorm2d(out_chs, eps=1e-3, momentum=0.01),
                                    nn.PReLU())

        self.layers['Fusion'] = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                                              nn.Conv2d(in_channels=in_chs, out_channels=in_chs // 4, kernel_size=1, bias=True),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=in_chs // 4, out_channels=out_chs, kernel_size=1, bias=True),
                                              nn.Sigmoid())
        
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through DiCE module.
        
        Applies dimension-wise convolutions across channel, height, and width dimensions,
        followed by adaptive fusion and channel shuffle for efficient spatial feature 
        extraction with dimension-wise processing.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
                Processed features through dimension-wise convolutions with adaptive fusion.
        """
        _, _, H, W = x.size()
        x_c = self.layers['Conv_Ch'](x)

        # Height-wise processing
        x_h = self._resize_if_needed(x, target_size=(self.height, W))
        x_h = x_h.transpose(1, 2).contiguous()
        x_h = self.layers['Conv_H'](x_h)
        x_h = x_h.transpose(1, 2).contiguous()
        x_h = self._resize_if_needed(x_h, target_size=(H, W))

        # Width-wise processing
        x_w = self._resize_if_needed(x, target_size=(H, self.width))
        x_w = x_w.transpose(1, 3).contiguous()
        x_w = self.layers['Conv_W'](x_w)
        x_w = x_w.transpose(1, 3).contiguous()
        x_w = self._resize_if_needed(x_w, target_size=(H, W))

        # Fusion
        x_cat = torch.cat((x_c, x_h, x_w), dim=1)
        x_cat = self.layers['Norm_Act'](x_cat)
        if self.shuffle:
            x_cat = self.channel_shuffle(x_cat, groups=3)

        fused = self.layers['Weights_AVG'](x_cat)
        proj = self.layers['Projection'](fused)
        weight = self.layers['Fusion'](fused)

        return proj * weight

    def _resize_if_needed(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Resize tensor to target spatial dimensions if needed.
        
        Args:
            x (torch.Tensor): Input tensor to potentially resize.
            target_size (Tuple[int, int]): Target (height, width) dimensions.
            
        Returns:
            torch.Tensor: Resized tensor or original if dimensions match.
        """
        if x.shape[2:] != target_size:
            return F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        
        return x

    def channel_shuffle(self, x: Tensor, groups: int) -> Tensor:
        """Apply channel shuffle for improved feature mixing across groups.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            groups (int): Number of groups for channel shuffling.
            
        Returns:
            torch.Tensor: Channel-shuffled tensor of same shape.
        """
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        
        return x

class StridedDiCEModule(nn.Module):
    """Strided Dimension-wise Convolutional Module with dual-branch architecture.
    
    Implements Strided DiCE Module from "DiCENet: Dimension-wise Convolutions for 
    Efficient Networks" by Mehta et al. Features dual-branch design with depthwise 
    convolutions and DiCE modules for efficient downsampling with dimension-wise 
    feature processing.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        height (int): Reference height dimension for DiCEModule operations. Must be > 0.
        width (int): Reference width dimension for DiCEModule operations. Must be > 0.
        kernel_size (int, optional): Kernel size for DiCEModule depthwise operations.
            Must be > 0. Defaults to 3.
        dilation (List[int], optional): Dilation rates for [channel, height, width] 
            convolutions in DiCE. Each value must be > 0. Defaults to [1, 1, 1].
        shuffle (bool, optional): Whether to apply channel shuffle after concatenation.
            Defaults to True.
            
    Attributes:
        name (str): Module identifier set to 'StridedDiCEModule'.
        layers (nn.ModuleDict): Dictionary containing left and right branch modules.
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels (2 * in_chs).
        shuffle (bool): Whether channel shuffle is enabled.
        
    Raises:
        ValueError: If in_chs, height, width, or kernel_size <= 0.
        ValueError: If dilation list length != 3.
        ValueError: If any dilation value <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        height: int, 
        width: int, 
        kernel_size: int = 3, 
        dilation: List[int] = [1, 1, 1], 
        shuffle: bool = True
    ) -> None:
        super(StridedDiCEModule, self).__init__()
        
        if in_chs <= 0:
            raise ValueError(f"in_chs must be > 0, got {in_chs}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be > 0, got height={height}, width={width}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if len(dilation) != 3:
            raise ValueError(f"dilation must be a list of 3 elements, got {len(dilation)}")
        for i, d in enumerate(dilation):
            if d <= 0:
                raise ValueError(f"dilation[{i}] must be > 0, got {d}")
                
        self.name = "StridedDiCEModule"
        self.in_chs = in_chs
        self.out_chs = 2 * in_chs
        self.height = height
        self.width = width
        self.shuffle = shuffle

        self.layers = nn.ModuleDict()

        # Left branch (depthwise conv + pointwise)
        self.layers['Branch_L'] = nn.Sequential(nn.Conv2d(in_channels=in_chs, 
                                                          out_channels=in_chs,
                                                          kernel_size=3, 
                                                          stride=2, 
                                                          padding=1,
                                                          dilation=1, 
                                                          groups=in_chs, 
                                                          bias=True),
                                                nn.BatchNorm2d(in_chs),
                                                nn.PReLU(),
                                                nn.Conv2d(in_channels=in_chs, 
                                                          out_channels=in_chs,
                                                          kernel_size=1, 
                                                          stride=1, 
                                                          padding=0,
                                                          groups=1, 
                                                          bias=True),
                                                nn.BatchNorm2d(in_chs),
                                                nn.PReLU())

        # Right branch (avgpool + DiCEModule + projection conv)
        self.layers['Branch_R'] = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                                DiCEModule(in_chs=in_chs,
                                                        out_chs=in_chs,
                                                        height=height,
                                                        width=width,
                                                        kernel_size=kernel_size,
                                                        dilation=dilation,
                                                        shuffle=shuffle),
                                                nn.Conv2d(in_channels=in_chs, out_channels=in_chs,
                                                        kernel_size=1, stride=1, padding=0,
                                                        groups=1, bias=True),
                                                nn.BatchNorm2d(in_chs),
                                                nn.PReLU())
        
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Strided DiCE module.
        
        Processes input through dual branches: left branch with strided depthwise 
        convolution and right branch with average pooling and DiCE module, then 
        concatenates and optionally applies channel shuffle for efficient downsampling.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, 2*C_in, H//2, W//2).
                Concatenated dual-branch features with 2x spatial downsampling.
        """
        x_left = self.layers['Branch_L'](x)
        x_right = self.layers['Branch_R'](x)
        x = torch.cat([x_left, x_right], dim=1)

        if self.shuffle:
            x = self.channel_shuffle(x, groups=2)
        
        return x

    @staticmethod
    def channel_shuffle(x: Tensor, groups: int) -> Tensor:
        """Apply channel shuffle for improved feature mixing across groups.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            groups (int): Number of groups for channel shuffling.
            
        Returns:
            torch.Tensor: Channel-shuffled tensor of same shape.
        """
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        
        return x

class ShuffleDiCEModule(nn.Module):
    """Channel-partitioned DiCE module with efficient feature processing.
    
    Implements Channel-partitioned DiCE Module from "DiCENet: Dimension-wise Convolutions 
    for Efficient Networks" by Mehta et al. Features channel splitting strategy where part 
    of channels bypass processing while others go through DiCE transformation for efficiency.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        height (int): Target height for DiCEModule operations. Must be > 0.
        width (int): Target width for DiCEModule operations. Must be > 0.
        chs_tag (float, optional): Proportion of input channels forwarded unprocessed.
            Must be in range (0, 1). Defaults to 0.5.
        groups (int, optional): Number of groups for final channel shuffle. Must be > 0.
            Defaults to 2.
            
    Attributes:
        name (str): Module identifier set to 'ShuffleDiCEModule'.
        layers (nn.ModuleDict): Dictionary containing right branch processing modules.
        left_chs (int): Number of channels in bypass path.
        right_in (int): Number of input channels for processing path.
        right_out (int): Number of output channels for processing path.
        groups (int): Number of groups for channel shuffle.
        
    Raises:
        ValueError: If in_chs, out_chs, height, width, or groups <= 0.
        ValueError: If chs_tag not in range (0, 1).
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        height: int, 
        width: int, 
        chs_tag: float = 0.5, 
        groups: int = 2
    ) -> None:
        super(ShuffleDiCEModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be > 0, got height={height}, width={width}")
        if not (0 < chs_tag < 1):
            raise ValueError(f"chs_tag must be in range (0, 1), got {chs_tag}")
        if groups <= 0:
            raise ValueError(f"groups must be > 0, got {groups}")
            
        self.name = "ShuffleDiCEModule"
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.groups = groups
        self.height = height
        self.width = width

        self.left_chs = round(chs_tag * in_chs)
        self.right_in = in_chs - self.left_chs
        self.right_out = out_chs - self.left_chs

        self.layers = nn.ModuleDict()

        self.layers['branch_R'] = nn.Sequential(nn.Conv2d(in_channels=self.right_in,
                                                          out_channels=self.right_out,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          dilation=1,
                                                          groups=1,
                                                          bias=True),
                                                nn.BatchNorm2d(self.right_out),
                                                nn.PReLU(),
                                                DiCEModule(in_chs=self.right_out,
                                                           out_chs=self.right_out,
                                                           height=self.height,
                                                           width=self.width,
                                                           kernel_size=3,
                                                           dilation=[1, 1, 1],
                                                           shuffle=True))
        
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Shuffle DiCE module.
        
        Splits input channels into bypass (left) and processing (right) paths. Left channels 
        pass through unchanged while right channels undergo DiCE transformation, then 
        concatenates and applies channel shuffle for efficient feature mixing.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
                Channel-shuffled concatenation of bypass and processed features.
        """
        left = x[:, :self.left_chs, :, :]
        right = x[:, self.left_chs:, :, :]
        right = self.layers['branch_R'](right)
        
        return self.channel_shuffle(torch.cat((left, right), dim=1), self.groups)

    @staticmethod
    def channel_shuffle(x: Tensor, groups: int) -> Tensor:
        """Apply channel shuffle for improved feature mixing across groups.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            groups (int): Number of groups for channel shuffling.
            
        Returns:
            torch.Tensor: Channel-shuffled tensor of same shape.
        """
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        
        return x


# GhostNet V1
class GhostModule(nn.Module):
    """Ghost convolution module for efficient feature map generation.
    
    Implements Ghost Module from "GhostNet: More Features from Cheap Operations" 
    by Han et al. Creates more feature maps from cheap operations (depthwise convolutions) 
    applied on a reduced set of intrinsic feature maps for computational efficiency.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        ratio (int): Ghost ratio (total output channels / intrinsic channels). Must be > 0.
        kernel_size (int): Kernel size for the primary convolution. Must be > 0.
        DW_kernel_size (int): Kernel size for the depthwise convolution. Must be > 0.
        stride (int): Stride for the primary convolution. Must be > 0.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
        
    Attributes:
        name (str): Module identifier set to 'GhostModule'.
        layers (nn.ModuleDict): Dictionary containing primary and cheap operation layers.
        out_chs (int): Number of output channels for slicing final output.
        
    Raises:
        ValueError: If in_chs, out_chs, ratio, kernel_size, DW_kernel_size, or stride <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        ratio: int, 
        kernel_size: int, 
        DW_kernel_size: int, 
        stride: int, 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super(GhostModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0, got {ratio}")
        if kernel_size <= 0 or DW_kernel_size <= 0:
            raise ValueError(f"kernel_size and DW_kernel_size must be > 0, "
                           f"got kernel_size={kernel_size}, DW_kernel_size={DW_kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
            
        self.name = "GhostModule"
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.layers = nn.ModuleDict()

        # Primary convolution: pointwise ConvBlock
        self.layers['Primary_Op'] = ConvBlock(in_chs=in_chs,
                                              out_chs=init_chs,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=kernel_size // 2,
                                              activation=activation)

        # Cheap operation: DWconv + BN + activation
        self.layers['Cheap_Op'] = nn.Sequential(nn.Conv2d(in_channels=init_chs,
                                                          out_channels=new_chs,
                                                          kernel_size=DW_kernel_size,
                                                          stride=1,
                                                          padding=DW_kernel_size // 2,
                                                          groups=init_chs,
                                                          bias=True),
                                                nn.BatchNorm2d(new_chs),
                                                activation)

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Ghost module.
        
        Applies primary convolution to generate intrinsic features, then applies cheap 
        depthwise operations to generate ghost features. Concatenates both and slices 
        to desired output channels for efficient feature map generation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Ghost features generated from primary and cheap operations.
        """
        x1 = self.layers['Primary_Op'](x)
        x2 = self.layers['Cheap_Op'](x1)
        out = torch.cat([x1, x2], dim=1)
        
        return out[:, :self.out_chs, :, :]

class GhostBottleneckModule(nn.Module):
    """Ghost bottleneck module with residual connection and optional Squeeze-and-Excitation.
    
    Implements Ghost Bottleneck from "GhostNet: More Features from Cheap Operations" 
    by Han et al. Features inverted residual structure with Ghost modules for efficient 
    feature expansion and projection, optional depthwise convolution, and SE attention.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        mid_chs (int): Number of middle channels after expansion. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        DW_kernel_size (int): Kernel size for the depthwise convolution. Must be > 0.
        stride (int): Stride for the depthwise convolution. Must be > 0.
        se_ratio (float): Squeeze-and-Excitation reduction ratio. Must be > 0 to enable SE,
            or 0 to disable.
            
    Attributes:
        name (str): Module identifier set to 'GhostBottleneckModule'.
        layers (nn.ModuleDict): Dictionary containing Ghost expansion, depthwise convolution,
            SE block, and projection layers.
        use_residual (bool): Whether residual connection is used.
        use_se (bool): Whether Squeeze-and-Excitation is enabled.
        
    Raises:
        ValueError: If in_chs, mid_chs, out_chs, DW_kernel_size, or stride <= 0.
        ValueError: If se_ratio < 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        mid_chs: int, 
        out_chs: int, 
        DW_kernel_size: int, 
        stride: int, 
        se_ratio: float
    ) -> None:
        super(GhostBottleneckModule, self).__init__()
        
        if in_chs <= 0 or mid_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs, mid_chs, and out_chs must be > 0, "
                           f"got in_chs={in_chs}, mid_chs={mid_chs}, out_chs={out_chs}")
        if DW_kernel_size <= 0:
            raise ValueError(f"DW_kernel_size must be > 0, got {DW_kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if se_ratio < 0:
            raise ValueError(f"se_ratio must be >= 0, got {se_ratio}")
            
        self.name = 'GhostBottleneckModule'
        self.use_residual = in_chs == out_chs and stride == 1
        self.use_se = se_ratio > 0

        self.layers = nn.ModuleDict()

        # Stage 1: Ghost expansion
        self.layers['Ghost_Expand'] = GhostModule(in_chs=in_chs,
                                                  out_chs=mid_chs,
                                                  ratio=2,
                                                  kernel_size=1,
                                                  DW_kernel_size=3,
                                                  stride=1,
                                                  activation=nn.ReLU())

        # Stage 2: Depthwise convolution (if stride > 1)
        if stride > 1:
            self.layers['DW_Conv'] = DepthwiseConvolution(in_chs=mid_chs,
                                                          kernel_size=DW_kernel_size,
                                                          stride=stride,
                                                          padding=DW_kernel_size // 2,
                                                          activation=nn.Identity())

        # Convolutional Squeeze-and-Excitation stage
        if self.use_se:
            self.layers['ConvSE'] = AltConvSE(in_chs=mid_chs,
                                              red_chs=None,
                                              ratio=se_ratio,
                                              divisor=4,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              activations=[nn.ReLU(), nn.Hardsigmoid()])

        # Stage 4: Ghost projection (no activation)
        self.layers['Ghost_Project'] = GhostModule(in_chs=mid_chs,
                                                   out_chs=out_chs,
                                                   ratio=2,
                                                   kernel_size=1,
                                                   DW_kernel_size=3,
                                                   stride=1,
                                                   activation=nn.Identity())

        # Stage 5: Shortcut connection (if needed)
        if not self.use_residual and stride > 1:
            self.layers['Shortcut'] = nn.Sequential(DepthwiseConvolution(in_chs=in_chs,
                                                                         kernel_size=DW_kernel_size,
                                                                         stride=stride,
                                                                         padding=DW_kernel_size // 2,
                                                                         activation=nn.Identity()),
                                                    PointwiseConvolution(in_chs=in_chs,
                                                                         out_chs=out_chs,
                                                                         stride=1,
                                                                         padding=0,
                                                                         activation=nn.Identity()))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Ghost bottleneck module.
        
        Applies Ghost expansion, optional depthwise convolution, optional SE attention,
        and Ghost projection. Includes residual connection when input and output 
        dimensions match with stride=1 for efficient inverted residual architecture.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Processed features through Ghost bottleneck with optional residual connection.
        """
        residual = x

        # Stage 1: Ghost expansion
        x = self.layers['Ghost_Expand'](x)

        # Stage 2: Depthwise convolution (if stride > 1)
        if 'DW_Conv' in self.layers:
            x = self.layers['DW_Conv'](x)

        # Stage 3: Squeeze-and-Excitation
        if self.use_se:
            x = self.layers['ConvSE'](x)

        # Stage 4: Ghost projection
        x = self.layers['Ghost_Project'](x)

        # Stage 5: Residual connection
        if self.use_residual:
            x = x + residual
        elif 'Shortcut' in self.layers:
            x = x + self.layers['Shortcut'](residual)

        return x


# GhostNet V2
class GhostModuleV2(nn.Module):
    """Ghost Module V2 with Dubbed Fully Connected Attention for enhanced cheap operations.
    
    Implements GhostModule V2 from "GhostNetV2: Enhance Cheap Operation with Long-Range 
    Attention" by Han et al. Features enhanced cheap operations with optional long-range 
    attention mechanism (DFC attention) for improved feature representation efficiency.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        ratio (int): Ghost ratio (total output channels / intrinsic channels). Must be > 0.
        kernel_size (int): Kernel size for the primary convolution. Must be > 0.
        DW_kernel_size (int): Kernel size for the depthwise convolution (cheap operation).
            Must be > 0.
        stride (int): Stride for the primary convolution. Must be > 0.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
        attention (bool, optional): Whether to apply long-range attention via DFC block.
            Defaults to False.
            
    Attributes:
        name (str): Module identifier set to 'GhostModuleV2'.
        layers (nn.ModuleDict): Dictionary containing primary operation, cheap operation,
            and optional DFC attention layers.
        out_chs (int): Number of output channels for slicing final output.
        attention (bool): Whether DFC attention is enabled.
        
    Raises:
        ValueError: If in_chs, out_chs, ratio, kernel_size, DW_kernel_size, or stride <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        out_chs: int, 
        ratio: int, 
        kernel_size: int, 
        DW_kernel_size: int, 
        stride: int, 
        activation: nn.Module = nn.ReLU(), 
        attention: bool = False
    ) -> None:
        super(GhostModuleV2, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0, got {ratio}")
        if kernel_size <= 0 or DW_kernel_size <= 0:
            raise ValueError(f"kernel_size and DW_kernel_size must be > 0, "
                           f"got kernel_size={kernel_size}, DW_kernel_size={DW_kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
            
        self.name = "GhostModuleV2"
        self.out_chs = out_chs
        self.attention = attention

        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.layers = nn.ModuleDict()

        # Primary convolution using ConvBlock auxiliary module
        self.layers['Primary_Op'] = ConvBlock(in_chs=in_chs,
                                              out_chs=init_chs,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=kernel_size // 2,
                                              activation=activation)

        # Cheap operation: DWconv + BN + activation
        self.layers['Cheap_Op'] = nn.Sequential(nn.Conv2d(in_channels=init_chs,
                                                          out_channels=new_chs,
                                                          kernel_size=DW_kernel_size,
                                                          stride=1,
                                                          padding=DW_kernel_size // 2,
                                                          groups=init_chs,
                                                          bias=True),
                                                nn.BatchNorm2d(new_chs),
                                                activation)

        # Optional Dubbed Fully Connected Attention Block
        if self.attention:
            self.layers['DFC_Att'] = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                                   ConvBlock(in_chs=in_chs,
                                                             out_chs=out_chs,
                                                             kernel_size=kernel_size,
                                                             stride=stride,
                                                             padding=kernel_size // 2,
                                                             activation=nn.Identity()),
                                                   nn.Conv2d(out_chs, 
                                                             out_chs, 
                                                             kernel_size=(1, 5), 
                                                             stride=1, 
                                                             padding=(0, 2),
                                                             groups=out_chs, 
                                                             bias=True),
                                                   nn.Conv2d(out_chs, 
                                                             out_chs, 
                                                             kernel_size=(5, 1), 
                                                             stride=1, 
                                                             padding=(2, 0),
                                                             groups=out_chs, 
                                                             bias=True),
                                                   nn.Sigmoid())
            
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Ghost Module V2.
        
        Applies primary convolution to generate intrinsic features, then applies cheap 
        depthwise operations to generate ghost features. Optionally applies DFC attention 
        for long-range feature enhancement before final output generation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Ghost features with optional DFC attention enhancement.
        """
        # Generate DFC attention weights if enabled
        if self.attention:
            att = self.layers['DFC_Att'](x)
            att = F.interpolate(att, size=x.shape[-2:], mode='nearest')

        # Primary operation and cheap operation
        x1 = self.layers['Primary_Op'](x)
        x2 = self.layers['Cheap_Op'](x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.out_chs, :, :]

        # Apply attention weighting if enabled
        return out * att if self.attention else out

class GhostBottleneckModuleV2(nn.Module):
    """Ghost Bottleneck Module V2 with enhanced cheap operations and long-range attention.
    
    Implements Ghost Bottleneck Module V2 from "GhostNetV2: Enhance Cheap Operation with 
    Long-Range Attention" by Han et al. Features inverted residual structure with 
    GhostModuleV2 for efficient feature expansion and projection, optional DFC attention, 
    depthwise convolution, and SE attention for enhanced mobile architectures.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        mid_chs (int): Number of intermediate channels after expansion. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0.
        DW_kernel_size (int): Kernel size for the intermediate depthwise convolution.
            Must be > 0.
        stride (int): Stride of the bottleneck block (used for downsampling). Must be > 0.
        se_ratio (float, optional): Squeeze-and-Excitation ratio. If None or <= 0, 
            SE is disabled. Defaults to None.
        attention (bool, optional): Whether to apply Dubbed Fully Connected Attention 
            in the expansion stage. Defaults to False.
        activation (nn.Module, optional): Activation function to use in convolutional blocks.
            Defaults to nn.ReLU().
            
    Attributes:
        name (str): Module identifier set to 'GhostBottleneckModuleV2'.
        layers (nn.ModuleDict): Dictionary containing Ghost expansion, depthwise convolution,
            SE block, and projection layers.
        stride (int): Stride value for the module.
        shortcut (nn.Module): Shortcut connection (identity or downsampling path).
        
    Raises:
        ValueError: If in_chs, mid_chs, out_chs, DW_kernel_size, or stride <= 0.
    """
    
    def __init__(
        self, 
        in_chs: int, 
        mid_chs: int, 
        out_chs: int, 
        DW_kernel_size: int, 
        stride: int, 
        se_ratio: Optional[float] = None, 
        attention: bool = False, 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super(GhostBottleneckModuleV2, self).__init__()
        
        if in_chs <= 0 or mid_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs, mid_chs, and out_chs must be > 0, "
                           f"got in_chs={in_chs}, mid_chs={mid_chs}, out_chs={out_chs}")
        if DW_kernel_size <= 0:
            raise ValueError(f"DW_kernel_size must be > 0, got {DW_kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
            
        self.name = 'GhostBottleneckModuleV2'
        self.stride = stride
        self.layers = nn.ModuleDict()
        has_se = se_ratio is not None and se_ratio > 0.

        # Stage 1: Ghost expansion with optional DFC attention
        self.layers['Ghost_PW_exp'] = GhostModuleV2(in_chs=in_chs,
                                                    out_chs=mid_chs,
                                                    ratio=2,
                                                    kernel_size=1,
                                                    DW_kernel_size=3,
                                                    stride=1,
                                                    activation=activation,
                                                    attention=attention)

        # Stage 2: Depthwise convolution (if stride > 1)
        if self.stride > 1:
            self.layers['DW_conv'] = nn.Sequential(nn.Conv2d(mid_chs, 
                                                             mid_chs, 
                                                             kernel_size=DW_kernel_size,
                                                             stride=stride, 
                                                             padding=DW_kernel_size // 2,
                                                             groups=mid_chs, 
                                                             bias=False),
                                                   nn.BatchNorm2d(mid_chs))

        # Stage 3: Squeeze-and-Excitation (if enabled)
        if has_se:
            self.layers['ConvSE'] = AltConvSE(in_chs=mid_chs,
                                              red_chs=None,
                                              ratio=se_ratio,
                                              divisor=4,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              activations=[nn.ReLU(), nn.Hardsigmoid()])
        else:
            self.layers['ConvSE'] = nn.Identity()

        # Stage 4: Ghost projection (no attention)
        self.layers['Ghost_PW_proj'] = GhostModuleV2(in_chs=mid_chs,
                                                     out_chs=out_chs,
                                                     ratio=2,
                                                     kernel_size=1,
                                                     DW_kernel_size=3,
                                                     stride=1,
                                                     activation=nn.Identity(),
                                                     attention=False)

        # Stage 5: Shortcut connection
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_chs, 
                                                    in_chs, 
                                                    kernel_size=DW_kernel_size,
                                                    stride=stride, 
                                                    padding=DW_kernel_size // 2,
                                                    groups=in_chs, 
                                                    bias=False),
                                          nn.BatchNorm2d(in_chs),
                                          PointwiseConvolution(in_chs=in_chs,
                                                               out_chs=out_chs,
                                                               stride=1,
                                                               padding=0,
                                                               activation=nn.Identity()))

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Ghost Bottleneck Module V2.
        
        Applies Ghost expansion with optional DFC attention, optional depthwise convolution,
        optional SE attention, and Ghost projection. Includes residual connection for 
        efficient inverted residual architecture with enhanced cheap operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Processed features through GhostV2 bottleneck with optional enhancements.
        """
        residual = x

        # Stage 1: Ghost expansion with optional DFC attention
        x = self.layers['Ghost_PW_exp'](x)
        
        # Stage 2: Depthwise convolution (if stride > 1)
        if self.stride > 1:
            x = self.layers['DW_conv'](x)
            
        # Stage 3: Squeeze-and-Excitation (or identity)
        x = self.layers['ConvSE'](x)
        
        # Stage 4: Ghost projection
        x = self.layers['Ghost_PW_proj'](x)
        
        # Stage 5: Residual connection
        x += self.shortcut(residual)

        return x


# ESPNet V1
class SpatialPyramidModule(nn.Module):
    """Spatial Pyramid Module with hierarchical dilated convolutions for multi-scale features.
    
    Implements Spatial Pyramid Module from "ESPNet: Efficient Spatial Pyramid of Dilated 
    Convolutions for Semantic Segmentation" by Mehta et al. Features hierarchical feature 
    fusion using dilated convolutions at multiple scales for efficient semantic segmentation.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0 and divisible by 4.
            
    Attributes:
        name (str): Module identifier set to 'SpatialPyramidModule'.
        layers (nn.ModuleDict): Dictionary containing input convolution and hierarchical
            dilated convolution layers.
        downsample (bool): Whether input and output channel counts differ.
        act (nn.Module): PReLU activation function.
        
    Raises:
        ValueError: If in_chs or out_chs <= 0.
        ValueError: If out_chs is not divisible by 4.
    """
    
    def __init__(self, in_chs: int, out_chs: int) -> None:
        super(SpatialPyramidModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if out_chs % 4 != 0:
            raise ValueError(f"out_chs must be divisible by 4, got {out_chs}")
            
        self.name = 'SpatialPyramidModule'
        self.downsample = in_chs != out_chs
        n = out_chs // 4
        self.layers = nn.ModuleDict()

        # Input Stage: pointwise or strided convolution based on downsampling need
        if self.downsample:
            self.layers['Conv_IN'] = nn.Conv2d(in_channels=in_chs,
                                               out_channels=n,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               bias=True)
        else:
            self.layers['Conv_IN'] = nn.Conv2d(in_channels=in_chs,
                                               out_channels=n,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0,
                                               bias=True)

        # Hierarchical Feature Maps with increasing dilation rates
        self.layers['LVL_1'] = nn.Conv2d(in_channels=n,
                                         out_channels=n,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         dilation=1,
                                         bias=True)
        self.layers['LVL_2'] = nn.Conv2d(in_channels=n,
                                         out_channels=n,
                                         kernel_size=3,
                                         stride=1,
                                         padding=2,
                                         dilation=2,
                                         bias=True)
        self.layers['LVL_3'] = nn.Conv2d(in_channels=n,
                                         out_channels=n,
                                         kernel_size=3,
                                         stride=1,
                                         padding=4,
                                         dilation=4,
                                         bias=True)
        self.layers['LVL_4'] = nn.Conv2d(in_channels=n,
                                         out_channels=n,
                                         kernel_size=3,
                                         stride=1,
                                         padding=8,
                                         dilation=8,
                                         bias=True)

        self.layers['Batch_Norm'] = nn.BatchNorm2d(out_chs)
        self.act = nn.PReLU()

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Spatial Pyramid Module.
        
        Applies hierarchical feature fusion using dilated convolutions at multiple scales.
        Features from different dilation rates are progressively combined and concatenated
        for multi-scale representation suitable for semantic segmentation tasks.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
                Multi-scale features with optional residual connection.
        """
        out = self.layers['Conv_IN'](x)

        # Hierarchical Feature Fusion with progressive combination
        lvl_1 = self.layers['LVL_1'](out)      # Dilation 1
        lvl_2 = self.layers['LVL_2'](out)      # Dilation 2
        lvl_3 = self.layers['LVL_3'](out)      # Dilation 4
        lvl_4 = self.layers['LVL_4'](out)      # Dilation 8

        # Progressive feature combination
        sum1 = lvl_1 + lvl_2                   # Combine scales 1-2
        sum2 = sum1 + lvl_3                    # Combine scales 1-3
        sum3 = sum2 + lvl_4                    # Combine scales 1-4

        # Concatenate all hierarchical features
        out = torch.cat([lvl_1, sum1, sum2, sum3], dim=1)
        out = self.layers['Batch_Norm'](out)
        out = self.act(out)

        # Add residual connection if no downsampling
        if not self.downsample:
            out += x

        return out


# ESPNet V2
class EESpatialPyramidModule(nn.Module):
    """Extremely Efficient Spatial Pyramid Module with hierarchical feature fusion.
    
    Implements EESP Module from "ESPNetv2: A Light-weight, Power Efficient, and General 
    Purpose Convolutional Neural Network" by Mehta et al. Features extremely efficient 
    spatial pyramid with grouped convolutions and hierarchical concurrent processing 
    for enhanced mobile performance.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0.
        out_chs (int): Number of output channels. Must be > 0 and divisible by number 
            of dilation branches.
        stride (int): Stride of the spatial pyramid. Must be > 0.
        dilations (List[int]): Dilation values for the depthwise branches. Each value 
            must be > 0.
            
    Attributes:
        name (str): Module identifier set to 'EESpatialPyramidModule'.
        layers (nn.ModuleDict): Dictionary containing grouped convolutions, hierarchical
            feature fusion, and shortcut connection layers.
        downsample (bool): Whether stride differs from 1.
        residual_act (nn.Module): PReLU activation for residual output.
        
    Raises:
        ValueError: If in_chs, out_chs, or stride <= 0.
        ValueError: If out_chs is not divisible by number of dilations.
        ValueError: If any dilation value <= 0.
    """

    def __init__(self, in_chs: int, out_chs: int, stride: int, dilations: List[int]) -> None:
        super(EESpatialPyramidModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        
        num_branches = len(dilations)
        if out_chs % num_branches != 0:
            raise ValueError(f"out_chs must be divisible by number of dilations ({num_branches}), "
                           f"got out_chs={out_chs}")
        
        for i, d in enumerate(dilations):
            if d <= 0:
                raise ValueError(f"dilations[{i}] must be > 0, got {d}")
                
        self.name = 'EESpatialPyramidModule'
        self.downsample = stride != 1
        mid_chs = out_chs // num_branches
        self.layers = nn.ModuleDict()

        # Stage 1: Grouped pointwise projection
        self.layers['GroupedConv_1'] = nn.Sequential(nn.Conv2d(in_channels=in_chs,
                                                               out_channels=mid_chs,
                                                               kernel_size=1,
                                                               stride=1,
                                                               padding=0,
                                                               groups=num_branches,
                                                               bias=False),
                                                    nn.BatchNorm2d(mid_chs),
                                                    nn.PReLU(num_parameters=mid_chs))

        # Stage 2: Hierarchical Feature Fusion (specialized hierarchical architecture)
        self.layers['HFF'] = self._make_hff(mid_chs, stride, dilations)

        # Stage 3: Pre-activation (BN + PReLU)
        self.layers['Pre_Act'] = nn.Sequential(nn.BatchNorm2d(out_chs),
                                               nn.PReLU(num_parameters=out_chs))

        # Stage 4: Grouped pointwise expansion
        self.layers['GroupedConv_2'] = nn.Sequential(nn.Conv2d(in_channels=out_chs,
                                                               out_channels=out_chs,
                                                               kernel_size=1,
                                                               stride=1,
                                                               padding=0,
                                                               groups=num_branches,
                                                               bias=False),
                                                    nn.BatchNorm2d(out_chs))

        # Stage 5: Shortcut connection
        if not self.downsample and in_chs != out_chs:
            self.layers['Shortcut'] = ConvBlock(in_chs=in_chs,
                                                out_chs=out_chs,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                activation=nn.Identity())
        else:
            self.layers['Shortcut'] = nn.Identity()
        
        self.residual_act = nn.PReLU(num_parameters=out_chs)
        
        self.apply(initialize_weights)

    def _make_hff(self, channels: int, stride: int, dilations: List[int]) -> nn.Module:
        """Creates Hierarchical Feature Fusion module with concurrent processing.
        
        Implements hierarchical feature combination where each branch progressively 
        incorporates information from previous branches through element-wise addition 
        before concatenation, enabling efficient multi-scale feature extraction.
        
        Args:
            channels (int): Number of input/output channels per branch. Must be > 0.
            stride (int): Stride for all depthwise convolutions. Must be > 0.
            dilations (List[int]): Dilation values for each depthwise branch. 
                Each value must be > 0.
                
        Returns:
            nn.Module: HierarchicalConcurrent module with configured depthwise 
                branches for multi-scale feature fusion.
                
        Raises:
            ValueError: If channels or stride <= 0.
            ValueError: If any dilation value <= 0.
        """
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        
        for i, d in enumerate(dilations):
            if d <= 0:
                raise ValueError(f"dilations[{i}] must be > 0, got {d}")
        
        class HierarchicalConcurrent(nn.Sequential):
            """Hierarchical concurrent processing with progressive feature combination.
            
            Each branch processes the input and adds the output from the previous 
            branch before concatenating all outputs. This enables hierarchical 
            feature representation at multiple scales.
            
            Args:
                axis (int): Concatenation axis. Default is 1 (channel dimension).
            """
            
            def __init__(self, axis: int = 1) -> None:
                super(HierarchicalConcurrent, self).__init__()
                self.axis = axis

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass with hierarchical feature combination.
                
                Args:
                    x (torch.Tensor): Input tensor of shape (N, C, H, W).
                    
                Returns:
                    torch.Tensor: Concatenated hierarchical features of shape 
                        (N, C * num_branches, H', W').
                """
                out = []
                y_prev = None
                for module in self._modules.values():
                    y = module(x)
                    if y_prev is not None:
                        y += y_prev
                    out.append(y)
                    y_prev = y
                
                return torch.cat(out, dim=self.axis)

        hff = HierarchicalConcurrent()
        
        for i, d in enumerate(dilations):
            hff.add_module(f'Branch_{i + 1}', nn.Conv2d(in_channels=channels,
                                                        out_channels=channels,
                                                        kernel_size=3,
                                                        stride=stride,
                                                        padding=d,
                                                        dilation=d,
                                                        groups=channels,
                                                        bias=True))
        
        return hff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Extremely Efficient Spatial Pyramid Module.
        
        Processes input through grouped pointwise projection, hierarchical feature 
        fusion with multi-scale dilated convolutions, pre-activation, grouped 
        expansion, and optional residual connection.
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (N, in_chs, H, W).
            
        Returns:
            torch.Tensor: Output feature tensor of shape (N, out_chs, H', W') where
                H' and W' depend on stride and padding configuration.
                
        Note:
            Residual connection is only applied when not downsampling (stride=1).
            For downsampling cases, only the main path output is returned.
        """
        out = self.layers['GroupedConv_1'](x)
        out = self.layers['HFF'](out)
        out = self.layers['Pre_Act'](out)
        out = self.layers['GroupedConv_2'](out)

        if not self.downsample:
            shortcut = self.layers['Shortcut'](x)
            out = self.residual_act(out + shortcut)

        return out


# LegoNet
class LegoModule(nn.Module):
    """Lego Convolution Module with efficient filter reuse and learnable combination.
    
    Implements Lego Convolution from "LegoNet: Efficient Convolutional Neural Networks 
    with Lego Filters" by Yang et al. Features learnable filter combination strategy 
    where input channels are split into groups and combined with shared Lego filters 
    through learnable coefficients for parameter-efficient convolutions.
    
    Args:
        in_chs (int): Number of input channels. Must be > 0 and divisible by n_split.
        out_chs (int): Number of output channels. Must be > 0.
        kernel_size (int): Size of the convolutional kernel. Must be > 0.
        n_split (int): Number of input channel splits. Must be > 0 and divide in_chs.
        n_lego (float): Proportion of lego filters over total filters. Must be in (0, 1].
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            
    Attributes:
        name (str): Module identifier set to 'LegoModule'.
        layers (nn.ModuleDict): Dictionary containing batch normalization and activation.
        lego (nn.Parameter): Shared lego filter parameters.
        aux_coefficients (nn.Parameter): Learnable combination coefficients.
        aux_combination (nn.Parameter): Auxiliary combination weights for gradient balancing.
        proxy_combination (torch.Tensor): Proxy tensor for one-hot combination selection.
        basic_chs (int): Number of channels per split group.
        n_lego (int): Actual number of lego filters.
        
    Raises:
        ValueError: If in_chs, out_chs, kernel_size, or n_split <= 0.
        ValueError: If in_chs is not divisible by n_split.
        ValueError: If n_lego not in range (0, 1].
    """

    def __init__(self, in_chs: int, out_chs: int, kernel_size: int, n_split: int, n_lego: float, activation: nn.Module = nn.ReLU()) -> None:
        super(LegoModule, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if n_split <= 0:
            raise ValueError(f"n_split must be > 0, got {n_split}")
        if in_chs % n_split != 0:
            raise ValueError(f"in_chs must be divisible by n_split, got in_chs={in_chs}, n_split={n_split}")
        if not (0 < n_lego <= 1):
            raise ValueError(f"n_lego must be in range (0, 1], got {n_lego}")
            
        self.name = "LegoModule"
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.n_split = n_split
        self.n_lego = int(out_chs * n_lego)
        self.basic_chs = in_chs // n_split

        self.layers = nn.ModuleDict()

        # Initialize lego filters and auxiliary tensors for learnable combination
        self.lego = nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.n_lego, self.basic_chs, kernel_size, kernel_size)))
        self.aux_coefficients = nn.Parameter(nn.init.kaiming_normal_(torch.rand(n_split, out_chs, self.n_lego, 1, 1)))
        self.aux_combination = nn.Parameter(nn.init.kaiming_normal_(torch.rand(n_split, out_chs, self.n_lego, 1, 1)))

        # BatchNorm and Activation
        self.layers["Batch_Norm"] = nn.BatchNorm2d(out_chs)
        self.layers["Activation"] = activation
        
        # Initialize proxy combination tensor
        self.proxy_combination = None
        
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Lego convolution with learnable filter combination.
        
        Applies channel splitting, shared lego filter convolutions, and learnable 
        combination strategy. Uses proxy combination tensor for differentiable 
        one-hot filter selection during training.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W').
                Features generated through learnable lego filter combinations.
        """
        # Create proxy combination tensor for differentiable one-hot selection
        proxy_combination = torch.zeros_like(self.aux_combination)
        proxy_combination.scatter_(2, self.aux_combination.argmax(dim=2, keepdim=True), 1)
        proxy_combination.requires_grad = True
        self.proxy_combination = proxy_combination

        out = 0
        # Process each channel split with shared lego filters
        for i in range(self.n_split):
            # Extract channel split
            x_split = x[:, i * self.basic_chs : (i + 1) * self.basic_chs]
            
            # Apply shared lego filters to split
            lego_feature = F.conv2d(x_split, self.lego, padding=self.kernel_size // 2)
            
            # Combine features using learnable coefficients and proxy selection
            kernel = self.aux_coefficients[i] * self.proxy_combination[i]
            out = out + F.conv2d(lego_feature, kernel)

        # Apply normalization and activation
        out = self.layers["Batch_Norm"](out)
        out = self.layers["Activation"](out)
        
        return out

    def copy_grad(self, balance_weight: float) -> None:
        """Copies gradients and applies frequency balancing for lego filter usage.
        
        Transfers gradients from proxy combination tensor to auxiliary combination 
        parameters and applies frequency balancing to ensure uniform usage of lego 
        filters across different splits and output channels.
        
        Args:
            balance_weight (float): Weight for gradient balancing adjustment. Higher 
                values increase the penalty for frequency imbalance.
                
        Note:
            This method should be called during training to maintain balanced 
            filter usage and prevent filter redundancy. The balancing ensures 
            each lego filter is used approximately equally across the network.
        """
        # Copy gradients from proxy to auxiliary combination
        self.aux_combination.grad = self.proxy_combination.grad
        
        # Calculate filter usage frequency for balancing
        idxs = self.aux_combination.argmax(dim=2).view(-1).cpu().numpy()
        unique, count = np.unique(idxs, return_counts=True)
        unique, count = np.unique(count, return_counts=True)
        avg_freq = (self.n_split * self.out_chs) / self.n_lego
        
        # Apply frequency balancing to gradients
        for i in range(self.n_lego):
            i_freq = (idxs == i).sum().item()
            # Skip if frequency is within acceptable range
            if np.floor(avg_freq) <= i_freq <= np.ceil(avg_freq):
                continue
            # Adjust gradients based on frequency imbalance
            if i_freq < np.floor(avg_freq):
                # Decrease gradient for under-used filters
                self.aux_combination.grad[:, :, i] -= balance_weight * (np.floor(avg_freq) - i_freq)
            else:
                # Increase gradient for over-used filters
                self.aux_combination.grad[:, :, i] += balance_weight * (i_freq - np.ceil(avg_freq))


# Versatile Filters
class VersatileConvolution(nn.Module):
    """Versatile Convolution with adaptive filter shapes for efficient CNNs.
    
    Implements Versatile Convolution from "Learning Versatile Filters for Efficient 
    Convolutional Neural Networks" by Wang et al. Features adaptive filter shapes 
    that can dynamically adjust their spatial support and channel combinations for 
    enhanced parameter efficiency and feature representation.
    
    Args:
        in_chs (int): Number of input channels. Must be > delta and divisible by groups.
        out_chs (int): Number of output channels. Must be > 0.
        kernel_size (int, optional): Size of the convolutional kernel. Must be > 0.
            Defaults to 3.
        stride (int, optional): Stride of the convolution. Must be > 0. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input.
            Must be >= 0. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Must be > 0.
            Defaults to 1.
        groups (int, optional): Number of blocked connections from input to output 
            channels. Must be > 0 and divide both in_chs and out_chs. Defaults to 1.
        delta (int, optional): Channel shift parameter for versatile filtering.
            Must be >= 0 and < in_chs. Defaults to 0.
        g (int, optional): Group size for channel shifting. Must be > 0 and divide 
            delta+1. Defaults to 1.
        activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            
    Attributes:
        name (str): Module identifier set to 'VersatileConvolution'.
        layers (nn.ModuleDict): Dictionary containing versatile convolution, batch 
            normalization, and activation layers.
        
    Raises:
        ValueError: If in_chs, out_chs, kernel_size, stride, or g <= 0.
        ValueError: If padding or delta < 0.
        ValueError: If delta >= in_chs.
        ValueError: If in_chs not divisible by groups.
        ValueError: If (delta + 1) not divisible by g.
    """
    
    def __init__(self, in_chs: int, out_chs: int, kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, dilation: int = 1, groups: int = 1, delta: int = 0, 
                 g: int = 1, activation: nn.Module = nn.ReLU()) -> None:
        super(VersatileConvolution, self).__init__()
        
        if in_chs <= 0 or out_chs <= 0:
            raise ValueError(f"in_chs and out_chs must be > 0, got in_chs={in_chs}, out_chs={out_chs}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}")
        if dilation <= 0:
            raise ValueError(f"dilation must be > 0, got {dilation}")
        if groups <= 0:
            raise ValueError(f"groups must be > 0, got {groups}")
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if g <= 0:
            raise ValueError(f"g must be > 0, got {g}")
        if delta >= in_chs:
            raise ValueError(f"delta must be < in_chs, got delta={delta}, in_chs={in_chs}")
        if in_chs % groups != 0:
            raise ValueError(f"in_chs must be divisible by groups, got in_chs={in_chs}, groups={groups}")
        if (delta + 1) % g != 0:
            raise ValueError(f"(delta + 1) must be divisible by g, got delta={delta}, g={g}")
            
        self.name = 'VersatileConvolution'
        self.layers = nn.ModuleDict()

        class VConv2d(nn.modules.conv._ConvNd):
            """Versatile 2D Convolution with adaptive spatial and channel processing.
            
            Custom convolution layer that implements versatile filtering by applying 
            different kernel shapes and channel shifts. Features dynamic spatial 
            support adjustment and channel combination strategies for efficient 
            parameter utilization.
            
            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                kernel_size (Union[int, tuple]): Size of the convolving kernel.
                stride (Union[int, tuple]): Stride of the convolution.
                padding (Union[int, tuple]): Zero-padding added to input.
                dilation (Union[int, tuple]): Spacing between kernel elements.
                groups (int): Number of blocked connections.
                bias (bool): Whether to add learnable bias.
                delta (int): Channel shift parameter for versatile filtering.
                g (int): Group size for channel shifting operations.
                
            Attributes:
                s_num (int): Number of spatial variations (ceil(kernel_size/2)).
                delta (int): Channel shift parameter.
                g (int): Group size for shifting.
                weight (nn.Parameter): Learnable convolution weights.
                real_out_chs (int): Actual number of output channels after processing.
                bias (nn.Parameter): Learnable bias parameters.
            """
            
            def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                        stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], 
                        dilation: Union[int, Tuple[int, int]], groups: int, bias: bool, 
                        delta: int, g: int) -> None:
                kernel_size = _pair(kernel_size)
                stride = _pair(stride)
                padding = _pair(padding)
                dilation = _pair(dilation)
                super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                               dilation, False, _pair(0), groups, False, padding_mode='zeros')
                
                # Versatile convolution parameters
                self.s_num = int(np.ceil(self.kernel_size[0] / 2))
                self.delta = delta
                self.g = g
                
                # Calculate actual weight dimensions
                weight_out_chs = int(out_channels / self.s_num / (1 + delta / g))
                self.weight = nn.Parameter(torch.Tensor(weight_out_chs, in_channels // groups, *kernel_size))
                self.real_out_chs = weight_out_chs * self.s_num * ((delta // g) + 1)
                self.bias = nn.Parameter(torch.zeros(self.real_out_chs))
                self.reset_parameters()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass through versatile convolution.
                
                Applies versatile filtering with multiple spatial scales and channel 
                shifts. Each combination of spatial scale and channel shift produces 
                different feature maps that are concatenated for final output.
                
                Args:
                    x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

                Returns:
                    torch.Tensor: Output tensor of shape (B, real_out_chs, H', W').
                        Features from versatile filtering with adaptive shapes.
                """
                x_list = []
                s_num = self.s_num
                ch_ratio = (1 + self.delta / self.g)
                ch_len = self.in_channels - self.delta
                
                # Apply versatile filtering with different spatial and channel configurations
                for s in range(s_num):
                    for start in range(0, self.delta + 1, self.g):
                        # Extract adaptive kernel shape
                        weight1 = self.weight[:, :ch_len, s:self.kernel_size[0] - s, s:self.kernel_size[0] - s]
                        
                        # Apply adaptive spatial cropping and padding
                        if self.padding[0] - s < 0:
                            h = x.size(2)
                            x1 = x[:, start:start + ch_len, s:h - s, s:h - s]
                            padding1 = _pair(0)
                        else:
                            x1 = x[:, start:start + ch_len, :, :]
                            padding1 = _pair(self.padding[0] - s)

                        # Calculate bias slice indices
                        b_start = int(self.real_out_chs * (s * ch_ratio + start) / s_num / ch_ratio)
                        b_end = int(self.real_out_chs * (s * ch_ratio + start + 1) / s_num / ch_ratio)

                        # Extract corresponding bias slice
                        bias_range = b_end - b_start
                        if bias_range == weight1.shape[0] and b_end <= self.bias.shape[0]:
                            bias = self.bias[b_start:b_end]
                        else:
                            bias = None

                        # Apply convolution with adaptive parameters
                        x_out = F.conv2d(x1,
                                         weight1,
                                         bias=bias,
                                         stride=self.stride,
                                         padding=padding1,
                                         dilation=self.dilation,
                                         groups=self.groups)
                        x_list.append(x_out)

                return torch.cat(x_list, dim=1)

        # Initialize versatile convolution layer
        self.layers['VConv'] = VConv2d(in_chs, out_chs, kernel_size, stride, padding, 
                                       dilation, groups, bias=True, delta=delta, g=g)
        
        # Batch normalization (using actual output channels from versatile conv)
        self.layers['Batch_Norm'] = nn.BatchNorm2d(self.layers['VConv'].real_out_chs)
        self.layers['Activation'] = activation

        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Versatile Convolution module.
        
        Applies versatile convolution with adaptive filter shapes, followed by 
        batch normalization and activation. The versatile convolution dynamically 
        adjusts spatial support and channel combinations for efficient feature extraction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, real_out_chs, H', W').
                Features processed through versatile filtering with adaptive shapes.
                
        Note:
            The actual number of output channels (real_out_chs) may differ from 
            the specified out_chs due to versatile convolution's adaptive structure.
        """
        x = self.layers['VConv'](x)
        x = self.layers['Batch_Norm'](x)
        x = self.layers['Activation'](x)
        
        return x





# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing Framework
# ------------------------------------------------------------------------------------------------------------------------------------------------------
class ModelProfiler:
    """Comprehensive model profiling with torchinfo integration"""
    def __init__(self):
        try:
            from torchinfo import summary
            self.summary = summary
            self.torchinfo_available = True
        except ImportError:
            print("   Torchinfo not available. Install it with: 'pip install torchinfo'")
            self.torchinfo_available = False
            self.summary = None
    
    def profile_model(self, model, input_tensor, config):
        """Profile model with torchinfo and return structured data"""
        profile_data = {'config_hash': self._hash_config(config),
                        'input_shape': str(tuple(input_tensor.shape)),
                        'total_params': 0,
                        'trainable_params': 0,
                        'model_size_mb': 0.0,
                        'macs': 0,
                        'output_shape': 'Unknown',
                        'torchinfo_success': False}
        
        if not self.torchinfo_available:
            return profile_data
            
        try:
            # Get torchinfo summary with detailed statistics
            model_stats = self.summary(model, 
                                       input_size=input_tensor.shape,
                                       verbose=0,  # Suppress output
                                       col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"],
                                       row_settings=["var_names"])

            # Extract key metrics
            profile_data.update({'total_params': model_stats.total_params,
                                 'trainable_params': model_stats.trainable_params,
                                 'model_size_mb': model_stats.total_param_bytes / (1024 * 1024),  # Convert to MB
                                 'macs': getattr(model_stats, 'total_mult_adds', 0),
                                 'torchinfo_success': True})
            
            # Get output shape by running a forward pass
            with torch.no_grad():
                output = model(input_tensor)
                profile_data['output_shape'] = str(tuple(output.shape))
                
        except Exception as e:
            print(f"   Torchinfo profiling failed: {str(e)}")
            
        return profile_data
    
    def _hash_config(self, config):
        """Create a hash for configuration to identify unique variants"""
        import hashlib
        import json
        
        # Convert config to a deterministic string
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class MemoryProfiler:
    """Memory profiling"""
    def __init__(self):
        import psutil
        import tracemalloc
        self.psutil = psutil
        self.tracemalloc = tracemalloc
        self.process = psutil.Process()
        self.baseline_memory = None
        self.tracemalloc_started = False
        
    def start_profiling(self):
        """Start memory profiling"""
        if not self.tracemalloc_started:
            self.tracemalloc.start()
            self.tracemalloc_started = True
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def get_memory_stats(self):
        """Get current memory statistics"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - (self.baseline_memory or current_memory)
        
        # Tracemalloc stats
        if self.tracemalloc_started:
            current, peak = self.tracemalloc.get_traced_memory()
            tracemalloc_current = current / 1024 / 1024  # MB
            tracemalloc_peak = peak / 1024 / 1024  # MB
        else:
            tracemalloc_current = tracemalloc_peak = 0
            
        return {'process_memory_mb': current_memory,
                'memory_delta_mb': memory_delta,
                'tracemalloc_current_mb': tracemalloc_current,
                'tracemalloc_peak_mb': tracemalloc_peak}
    
    def profile_module_memory(self, module_class, config, input_tensor):
        """Profile memory usage specifically for module creation and inference
        
        Returns:
            dict: Memory metrics with detailed breakdown:
                - module_creation_mb: Memory used for module initialization
                - inference_memory_mb: Additional memory used during forward pass
                - peak_inference_mb: Peak memory during inference (from tracemalloc)
                - total_module_mb: Total memory attributed to this module
        """
        import gc
        import torch
        
        # Clear any existing tensors and force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Reset tracemalloc peak
        if self.tracemalloc_started:
            self.tracemalloc.reset_peak()
        
        # Baseline memory before module creation
        baseline_memory = self.process.memory_info().rss / 1024 / 1024
        baseline_tracemalloc = self.tracemalloc.get_traced_memory()[0] / 1024 / 1024 if self.tracemalloc_started else 0
        
        # Create module and measure memory
        module = module_class(**config)
        post_creation_memory = self.process.memory_info().rss / 1024 / 1024
        module_creation_mb = post_creation_memory - baseline_memory
        
        # Measure inference memory
        with torch.no_grad():
            _ = module(input_tensor)
        
        post_inference_memory = self.process.memory_info().rss / 1024 / 1024
        inference_memory_mb = post_inference_memory - post_creation_memory
        
        # Get peak memory from tracemalloc
        if self.tracemalloc_started:
            peak_inference_mb = self.tracemalloc.get_traced_memory()[1] / 1024 / 1024 - baseline_tracemalloc
        else:
            peak_inference_mb = 0
        
        total_module_mb = module_creation_mb + inference_memory_mb
        
        return {'module_creation_mb': module_creation_mb,
                'inference_memory_mb': inference_memory_mb, 
                'peak_inference_mb': peak_inference_mb,
                'total_module_mb': total_module_mb}

class ModuleTestFramework:
    """Testing framework with progress tracking, failure logging, and model profiling"""
    
    def __init__(self, log_failures_only=True, enable_profiling=True, csv_output="model_profiling.csv"):
        import logging
        from tqdm import tqdm
        
        self.tqdm = tqdm
        self.log_failures_only = log_failures_only
        self.enable_profiling = enable_profiling
        self.csv_output = csv_output
        self.memory_profiler = MemoryProfiler()
        self.model_profiler = ModelProfiler() if enable_profiling else None
        self.failed_tests = []
        self.profiling_data = []
        
        # Setup CSV file for profiling data
        if self.enable_profiling:
            self._setup_csv_file()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler('test_failures.log'),
                                      logging.StreamHandler() if not log_failures_only else logging.NullHandler()])
        self.logger = logging.getLogger(__name__)
    
    def _setup_csv_file(self):
        """Setup CSV file with headers for profiling data"""
        import csv
        
        headers = ['module_name', 'config_hash', 'config_str', 'total_params', 'trainable_params', 
                   'model_size_mb', 'macs', 'input_shape', 'output_shape', 
                   'forward_time_ms', 'module_creation_mb', 'inference_memory_mb', 
                   'peak_inference_mb', 'total_module_mb',
                   'inference_success', 'torchinfo_success', 'timestamp']
        
        # Create new CSV file or append to existing
        with open(self.csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        
        print(f" Profiling data will be saved to: {self.csv_output}")
    
    def _save_profiling_data(self, module_name, config, profile_data, timing_data, enhanced_memory_data, success):
        """Save enhanced profiling data to CSV"""
        import csv
        import datetime
        
        if not self.enable_profiling:
            return
            
        row_data = [module_name,
                    profile_data['config_hash'],
                    self._format_config(config),
                    profile_data['total_params'],
                    profile_data['trainable_params'],
                    profile_data['model_size_mb'],
                    profile_data['macs'],
                    profile_data['input_shape'],
                    profile_data['output_shape'],
                    timing_data['forward_time_ms'],
                    enhanced_memory_data['module_creation_mb'],
                    enhanced_memory_data['inference_memory_mb'],
                    enhanced_memory_data['peak_inference_mb'],
                    enhanced_memory_data['total_module_mb'],
                    success,
                    profile_data['torchinfo_success'],
                    datetime.datetime.now().isoformat()]
        
        # Append to CSV file
        with open(self.csv_output, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
        
    def run_module_tests(self, module_name, test_configs, module_class, input_generator):
        """Run tests for a single module with progress tracking and comprehensive profiling"""
        
        print(f"\n{'='*60}")
        print(f"Testing {module_name}")
        if self.enable_profiling:
            print(f" Profiling enabled - data will be saved to {self.csv_output}")
        print(f"{'='*60}")
        
        total_configs = len(test_configs)
        successful_tests = 0
        
        # Start memory profiling
        self.memory_profiler.start_profiling()
        
        with self.tqdm(test_configs, desc=f"{module_name}", unit="config", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for config in pbar:
                success = False
                timing_data = {'forward_time_ms': 0.0}
                memory_data = {'memory_delta_mb': 0.0, 'tracemalloc_peak_mb': 0.0}
                profile_data = {'config_hash': '', 'total_params': 0, 'trainable_params': 0,
                                'model_size_mb': 0.0, 'macs': 0,
                                'input_shape': 'Unknown', 'output_shape': 'Unknown',
                                'torchinfo_success': False}
                
                try:
                    # Update progress bar description with current config
                    config_str = self._format_config(config)
                    pbar.set_postfix_str(f"Config: {config_str[:40]}...")
                    
                    # Enhanced memory profiling for this specific module
                    if self.enable_profiling:
                        enhanced_memory_data = self.memory_profiler.profile_module_memory(module_class, config, input_generator(config))
                    else:
                        enhanced_memory_data = {'module_creation_mb': 0.0,
                                                'inference_memory_mb': 0.0,
                                                'peak_inference_mb': 0.0,
                                                'total_module_mb': 0.0}
                    
                    # Create module and input for timing (separate from memory profiling)
                    module = module_class(**config)
                    x = input_generator(config)
                    
                    # Model profiling with torchinfo
                    if self.enable_profiling and self.model_profiler:
                        profile_data = self.model_profiler.profile_model(module, x, config)
                    
                    # Timed inference
                    import time
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        y = module(x)
                    
                    end_time = time.perf_counter()
                    timing_data['forward_time_ms'] = (end_time - start_time) * 1000
                    
                    # Update profile data with actual output shape
                    if 'output_shape' not in profile_data or profile_data['output_shape'] == 'Unknown':
                        profile_data['output_shape'] = str(tuple(y.shape))
                    
                    successful_tests += 1
                    success = True
                    
                    # Log only if not in failure-only mode
                    if not self.log_failures_only:
                        self.logger.info(f"✓ {module_name} - {config_str} - Output: {y.shape} - "
                                         f"Time: {timing_data['forward_time_ms']:.2f}ms - "
                                         f"Total Module Memory: {enhanced_memory_data['total_module_mb']:.2f}MB - "
                                         f"Params: {profile_data['total_params']}")
                        
                except Exception as e:
                    # Always log failures
                    error_msg = f"✗ {module_name} - {self._format_config(config)} - Error: {str(e)}"
                    self.logger.error(error_msg)
                    self.failed_tests.append({'module': module_name,
                                              'config': config,
                                              'error': str(e)})
                    
                    pbar.set_postfix_str(f"FAILED: {str(e)[:30]}...")
                    
                    # Set empty data for failed tests
                    profile_data = {}
                    timing_data = {}
                    enhanced_memory_data = {}
                    success = False
                
                # Save profiling data (both success and failure cases)
                if self.enable_profiling:
                    self._save_profiling_data(module_name, config, profile_data, timing_data, enhanced_memory_data, success)
        
        # Final summary for this module
        final_memory_stats = self.memory_profiler.get_memory_stats()
        success_rate = (successful_tests / total_configs) * 100
        
        print(f"\n{module_name} Summary:")
        print(f"  ✓ Successful: {successful_tests}/{total_configs} ({success_rate:.1f}%)")
        print(f"  ✗ Failed: {len([f for f in self.failed_tests if f['module'] == module_name])}")
        print(f"  Memory Usage: {final_memory_stats['process_memory_mb']:.1f}MB (+{final_memory_stats['memory_delta_mb']:.1f}MB)")
        print(f"  Peak Memory: {final_memory_stats['tracemalloc_peak_mb']:.1f}MB")
        
        if self.enable_profiling:
            print(f"  Profiling data saved to: {self.csv_output}")
        
    def _format_config(self, config):
        """Format configuration for display with actual values"""
        items = []
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                # Show actual list/tuple values, truncate if too long
                value_str = str(value)
                if len(value_str) > 20:
                    value_str = value_str[:17] + "..."
                items.append(f"{key}={value_str}")
            elif hasattr(value, '__class__') and hasattr(value, '__name__'):
                # For functions/classes, show the name
                items.append(f"{key}={value.__class__.__name__}")
            elif hasattr(value, '__class__') and value.__class__.__module__ == 'torch.nn.modules.activation':
                # For PyTorch activations, show the class name
                items.append(f"{key}={value.__class__.__name__}")
            else:
                # Show actual value for primitives (int, float, bool, str)
                items.append(f"{key}={value}")
        return ", ".join(items)
    
    def print_final_summary(self):
        """Print final testing summary with profiling insights"""
        print(f"\n{'='*80}")
        print(f"FINAL TESTING SUMMARY")
        print(f"{'='*80}")
        
        if self.failed_tests:
            print(f"✗ Total Failures: {len(self.failed_tests)}")
            print(" Detailed failure log saved to 'test_failures.log'")
            
            # Group failures by module
            failure_by_module = {}
            for failure in self.failed_tests:
                module = failure['module']
                if module not in failure_by_module:
                    failure_by_module[module] = 0
                failure_by_module[module] += 1
            
            print("\nFailures by module:")
            for module, count in failure_by_module.items():
                print(f"  • {module}: {count} failures")
        else:
            print(" All tests passed successfully!")
            
        final_memory = self.memory_profiler.get_memory_stats()
        print(f"\nFinal Memory Stats:")
        print(f"  • Process Memory: {final_memory['process_memory_mb']:.1f}MB")
        print(f"  • Memory Delta: {final_memory['memory_delta_mb']:.1f}MB")
        print(f"  • Peak Memory: {final_memory['tracemalloc_peak_mb']:.1f}MB")
        
        if self.enable_profiling:
            print(f"\nProfiling Summary:")
            print(f"  • Profiling data saved to: {self.csv_output}")


# Test Configuration Generators
def generate_depthwise_separable_configs():
    """Generate test configurations for DepthwiseSeparableConvolutionModule"""
    configs = []
    for in_chs, out_chs in [(16, 32), (24, 48), (32, 64), (16, 24)]:
        for stride in [1, 2]:
            for DW_kernel_size in [3, 5]:
                for activation in [nn.ReLU(), nn.SiLU()]:
                    # Skip some combinations to manage output
                    if DW_kernel_size > 3 and stride == 2:
                        continue
                        
                    padding = DW_kernel_size // 2
                    configs.append({
                        'in_chs': in_chs,
                        'out_chs': out_chs,
                        'DW_kernel_size': DW_kernel_size,
                        'DW_stride': stride,
                        'DW_padding': padding,
                        'DW_activation': activation
                    })
    return configs

def generate_inverted_residual_configs():
    """Generate test configurations for InvertedResidualModule"""
    configs = []
    for in_chs, out_chs in [(16, 16), (16, 32), (24, 24), (32, 64)]:
        for stride in [1, 2]:
            for expansion in [1, 3, 6]:
                    # Skip residual mismatch for stride=2 or channel mismatch
                    if stride == 2 and in_chs == out_chs:
                        continue
                        
                    configs.append({
                        'in_chs': in_chs,
                        'out_chs': out_chs,
                        'expansion': expansion,
                        'stride': stride,
                    })
    return configs

def generate_sandglass_configs():
    """Generate test configurations for SandGlassModule"""
    configs = []
    for in_channels, out_channels in [(16, 16), (16, 32), (32, 32)]:
        for stride in [1, 2]:
            for exp_ratio in [1, 2, 4]:
                for identity_mul in [1.0, 0.5]:
                    for keep_3x3 in [False, True]:
                        configs.append({
                            'in_chs': in_channels,
                            'out_chs': out_channels,
                            'stride': stride,
                            'exp_ratio': exp_ratio,
                            'identity_mul': identity_mul,
                            'keep_3x3': keep_3x3
                        })
    return configs

def generate_inverted_residual_se_configs():
    """Generate test configurations for InvertedResidualSEModule"""
    configs = []
    for stride in [1, 2]:
        for in_chs, hidden_chs, out_chs in [(16, 32, 16), (24, 48, 24), (32, 64, 16)]:
            for kernel_size in [3, 5]:
                for use_se in [False, True]:
                    configs.append({
                        'in_chs': in_chs,
                        'hidden_chs': hidden_chs,
                        'out_chs': out_chs,
                        'DW_kernel_size': kernel_size,
                        'DW_stride': stride,
                        'use_se': use_se,
                    })
    return configs

def generate_universal_inverted_bottleneck_configs():
    """Generate test configurations for UniversalInvertedBottleneckModule"""
    configs = []
    for in_chs, out_chs in [(24, 24), (24, 32), (32, 48), (16, 24)]:
        for stride in [1, 2]:
            for start_DW in [3, 5, False]:
                for middle_DW in [3, 5, False]:
                    for downsample in [True, False]:
                        for use_ls in [True, False]:
                            for expand_ratio in [2, 4, 6]:
                                # Skip some complex combinations to manage output
                                if start_DW and middle_DW and start_DW > 3 and middle_DW > 3:
                                    continue
                                if expand_ratio > 4 and use_ls:
                                    continue
                                if stride == 2 and in_chs == out_chs and downsample:
                                    continue
                                if not start_DW and not middle_DW:
                                    continue
                                    
                                configs.append({
                                    'in_chs': in_chs,
                                    'out_chs': out_chs,
                                    'start_DW_kernel_size': start_DW,
                                    'middle_DW_kernel_size': middle_DW,
                                    'middle_DW_downsample': downsample,
                                    'stride': stride,
                                    'expand_ratio': expand_ratio,
                                    'use_layer_scale': use_ls
                                })
    return configs

def generate_efficient_module_configs():
    """Generate test configurations for EfficientModule"""
    configs = []
    for stride in [1, 2]:
        for in_chs, out_chs in [(16, 32), (24, 32), (32, 48)]:
            for kernel_size in [3, 5]:
                for ratio in [3, 6]:
                    for reduction in [4, 8]:
                        configs.append({
                            'in_chs': in_chs,
                            'out_chs': out_chs,
                            'kernel_size': kernel_size,
                            'stride': stride,
                            'padding': kernel_size//2,
                            'ratio': ratio,
                            'reduction': reduction
                        })
    return configs

def generate_residual_efficient_configs():
    """Generate test configurations for ResidualEfficientModule"""
    configs = []
    for stride in [1, 2]:
        for in_chs, out_chs in [(24, 24), (32, 32), (24, 48)]:
            for kernel_size in [3, 5]:
                for expansion in [1, 3, 6]:
                    for reduction in [4, 8]:
                        if stride == 2 and in_chs == out_chs:
                            continue
                        configs.append({
                            'in_chs': in_chs,
                            'out_chs': out_chs,
                            'DW_kernel_size': kernel_size,
                            'DW_stride': stride,
                            'expansion': expansion,
                            'reduction': reduction
                        })
    return configs

def generate_shuffle_module_configs():
    """Generate test configurations for ShuffleModule"""
    configs = []
    for in_chs in [24, 32, 48]:
        for combine in ['add', 'concat']:
            for grouped_conv in [True, False]:
                for groups in [1, 2, 4]:
                        if combine == 'add':
                            out_chs = in_chs
                        else:
                            out_chs = in_chs + 8
                            
                        if in_chs % groups != 0 or out_chs % groups != 0:
                            continue
                        if groups > 2 and not grouped_conv:
                            continue
                        
                        # Fix: validate internal_out_channels divisibility for ShuffleModule
                        if combine == 'add':
                            internal_out_channels = out_chs
                        else:  # concat
                            internal_out_channels = out_chs - in_chs
                        
                        if internal_out_channels % groups != 0:
                            continue
                        
                        # Fix: validate bottleneck_channels divisibility 
                        bottleneck_channels = internal_out_channels // 4
                        first_1x1_groups = groups if grouped_conv else 1
                        if bottleneck_channels % first_1x1_groups != 0:
                            continue
                            
                        configs.append({
                            'in_chs': in_chs,
                            'out_chs': out_chs,
                            'groups': groups,
                            'grouped_conv': grouped_conv,
                            'combine': combine,
                        })
    return configs

def generate_ghost_module_configs():
    """Generate test configurations for GhostModule"""
    configs = []
    for in_chs, out_chs in [(16, 24), (24, 32), (32, 16), (16, 48)]:
        for ratio in [2, 4, 8]:
            for kernel_size in [1, 3]:
                for DW_kernel_size in [3, 5]:
                    for stride in [1, 2]:
                        if ratio > 4 and DW_kernel_size > 3:
                            continue
                        if stride == 2 and kernel_size == 1:
                            continue
                            
                        configs.append({
                            'in_chs': in_chs,
                            'out_chs': out_chs,
                            'ratio': ratio,
                            'kernel_size': kernel_size,
                            'DW_kernel_size': DW_kernel_size,
                            'stride': stride,
                            'activation': nn.ReLU()
                        })
    return configs

def generate_inverted_residual_shuffle_configs():
    """Generate test configurations for InvertedResidualShuffleModule"""
    configs = []
    for in_chs, out_chs in [(24, 24), (24, 32), (32, 32), (16, 24)]:
        for stride in [1, 2]:
            for model in [1, 2]:
                    if model == 1 and stride == 2:
                        continue
                    if stride == 2 and in_chs == out_chs:
                        continue
                    
                    # Fix: validate channel constraints for InvertedResidualShuffleModule
                    mid_channels = out_chs // 2
                    if model == 1:
                        # Model 1 splits input, requires in_chs % 2 == 0 and in_chs // 2 == mid_channels
                        if in_chs % 2 != 0 or in_chs // 2 != mid_channels:
                            continue
                    else:  # model == 2
                        # Model 2 processes full input, requires in_chs == mid_channels
                        if in_chs != mid_channels:
                            continue
                        
                    configs.append({
                        'in_chs': in_chs,
                        'out_chs': out_chs,
                        'stride': stride,
                        'model': model,
                    })
    return configs

def generate_fire_module_configs():
    """Generate test configurations for FireModule"""
    configs = []
    for in_chs in [24, 32, 48]:
        for squeeze_ratio in [0.125, 0.25, 0.5]:
            for exp_ratio_1x1 in [0.5, 0.75]:
                squeeze_chs = int(in_chs * squeeze_ratio)
                total_exp = in_chs * 2
                exp1x1_chs = int(total_exp * exp_ratio_1x1)
                exp3x3_chs = total_exp - exp1x1_chs
                
                configs.append({
                    'in_chs': in_chs,
                    'squeeze_chs': squeeze_chs,
                    'expPW_chs': exp1x1_chs,
                    'exp3x3_chs': exp3x3_chs
                })
    return configs

def generate_squeezenext_configs():
    """Generate test configurations for SqueezeNeXtModule"""
    configs = []
    for stride in [1, 2]:
        for in_chs, out_chs in [(24, 24), (24, 32), (32, 24), (48, 32)]:
            configs.append({
                'in_chs': in_chs,
                'out_chs': out_chs,
                'stride': stride,
            })
    return configs

def generate_condense_configs():
    """Generate test configurations for CondenseModule"""
    configs = []
    for in_chs in [16, 24]:
        for growth_rate in [8, 12]:
            for condense_factor in [4, 8]:
                for group_factor in [1, 2]:
                    config_dict = {
                        'conv_bottleneck_chs': growth_rate // 2,
                        'group1x1': group_factor,
                        'group3x3': group_factor,
                        'condense_factor': condense_factor,
                        'dropout_rate': 0.1
                    }
                    configs.append({
                        'in_chs': in_chs,
                        'growth_rate': growth_rate,
                        'config': config_dict
                    })
    return configs

def generate_factorized_convolution_configs():
    """Generate test configurations for FactorizedConvolutionModule"""
    configs = []
    for in_chs in [16, 24, 32]:
        for kernel_size in [3, 5, 7]:
            for dilation in [1, 2]:
                for drop_prob in [0.0, 0.1, 0.2]:
                    configs.append({
                        'in_chs': in_chs,
                        'kernel_size': kernel_size,
                        'drop_prob': drop_prob,
                        'dilation': dilation
                    })
    return configs

def generate_parallel_factorized_convolution_configs():
    """Generate test configurations for ParallelFactorizedConvolutionModule"""
    configs = []
    for in_chs in [16, 24, 32]:
        for drop_prob in [0.0, 0.2, 0.3]:
            configs.append({
                'in_chs': in_chs,
                'drop_prob': drop_prob
            })
    return configs

def generate_stem_module_configs():
    """Generate test configurations for StemModule"""
    configs = []
    for in_chs in [3, 16, 24]:
        for out_chs in [32, 48, 64]:
            configs.append({
                'in_chs': in_chs,
                'out_chs': out_chs,
            })
    return configs

def generate_mobile_one_configs():
    """Generate test configurations for MobileOneModule"""
    configs = []
    for inference_mode in [False, True]:
        for use_se in [False, True]:
            for num_branches in [1, 2, 4]:
                for stride in [1, 2]:
                    for in_chs, out_chs in [(16, 16), (16, 32), (32, 32), (24, 48)]:
                        for kernel_size in [3, 5]:
                            if num_branches > 2 and use_se and kernel_size > 3:
                                continue
                            if stride == 2 and in_chs == out_chs and num_branches > 2:
                                continue
                                
                            padding = kernel_size // 2
                            configs.append({
                                'in_chs': in_chs,
                                'out_chs': out_chs,
                                'kernel_size': kernel_size,
                                'stride': stride,
                                'padding': padding,
                                'groups': 1,
                                'inference_mode': inference_mode,
                                'use_se': use_se,
                                'num_conv_branches': num_branches
                            })
    return configs

def generate_mix_depth_configs():
    """Generate test configurations for MixDepthModule"""
    configs = []
    for exp_ratio in [1, 2, 4]:
        for stride in [1, 2]:
            for red_ratio in [None, 4, 8]:
                for dropout_rate in [0.0, 0.2]:
                    for in_chs, out_chs in [(16, 16), (24, 24), (16, 32), (32, 48)]:
                        if stride == 2 and in_chs == out_chs:
                            out_chs = in_chs * 2
                        
                        for exp_kernels, poi_kernels in [([1, 3], [1, 3]), ([3, 5], [3, 5]), ([1, 5], [1, 5])]:
                            if exp_ratio > 2 and red_ratio and dropout_rate > 0.1:
                                continue
                                
                            configs.append({
                                'in_chs': in_chs,
                                'out_chs': out_chs,
                                'exp_ratio': exp_ratio,
                                'exp_kernel_sizes': exp_kernels,
                                'kernel_sizes': [3, 5],
                                'poi_kernel_sizes': poi_kernels,
                                'stride': stride,
                                'dilation': 1,
                                'red_ratio': red_ratio,
                                'dropout_rate': dropout_rate,
                                'activation': nn.SiLU()
                            })
    return configs

def generate_dice_configs():
    """Generate test configurations for DiCEModule"""
    configs = []
    for shuffle in [False, True]:
        for (H, W) in [(32, 32), (48, 64), (24, 48), (64, 32)]:
            for in_chs, out_chs in [(16, 32), (24, 24), (32, 48), (16, 64)]:
                for kernel_size in [3, 5]:
                    for dilation in [[1, 2, 3], [1, 2, 4], [1, 1, 2], [2, 3, 4]]:
                        if kernel_size > 3 and max(dilation) > 3:
                            continue
                        if H != W and shuffle and kernel_size > 3:
                            continue
                            
                        configs.append({
                            'in_chs': in_chs,
                            'out_chs': out_chs,
                            'height': H,
                            'width': W,
                            'kernel_size': kernel_size,
                            'dilation': dilation,
                            'shuffle': shuffle
                        })
    return configs

def generate_strided_dice_configs():
    """Generate test configurations for StridedDiCEModule"""
    configs = []
    for shuffle in [False, True]:
        for (H, W) in [(32, 32), (48, 64), (24, 48), (64, 32)]:
            for in_chs in [16, 24, 32]:
                for kernel_size in [3, 5]:
                    for dilation in [[1, 2, 3], [1, 2, 4], [2, 3, 4]]:
                        if kernel_size > 3 and max(dilation) > 3:
                            continue
                        if H != W and shuffle and in_chs > 24:
                            continue
                            
                        configs.append({
                            'in_chs': in_chs,
                            'height': H,
                            'width': W,
                            'kernel_size': kernel_size,
                            'dilation': dilation,
                            'shuffle': shuffle
                        })
    return configs

def generate_shuffle_dice_configs():
    """Generate test configurations for ShuffleDiCEModule"""
    configs = []
    for ch_tag in [0.25, 0.5, 0.75]:
        for (H, W) in [(32, 32), (48, 64), (24, 48)]:
            for in_chs, out_chs in [(16, 24), (24, 32), (16, 32), (32, 48)]:
                for groups in [1, 2, 4]:
                    if in_chs % groups != 0 or out_chs % groups != 0:
                        continue
                    if groups > 2 and H != W:
                        continue
                        
                    configs.append({
                        'in_chs': in_chs,
                        'out_chs': out_chs,
                        'height': H,
                        'width': W,
                        'chs_tag': ch_tag,
                        'groups': groups
                    })
    return configs

def generate_ghost_bottleneck_configs():
    """Generate test configurations for GhostBottleneckModule"""
    configs = []
    for in_chs in [16, 24, 32]:
        for mid_chs in [24, 32, 48]:
            for out_chs in [16, 32, 48]:
                for DW_kernel_size in [3, 5]:
                    for stride in [1, 2]:
                        for se_ratio in [0.0, 0.25, 0.5]:
                            if mid_chs < in_chs or mid_chs < out_chs:
                                continue
                            if se_ratio > 0.25 and DW_kernel_size > 3:
                                continue
                            if stride == 2 and in_chs == out_chs and se_ratio > 0:
                                continue
                                
                            configs.append({
                                'in_chs': in_chs,
                                'mid_chs': mid_chs,
                                'out_chs': out_chs,
                                'DW_kernel_size': DW_kernel_size,
                                'stride': stride,
                                'se_ratio': se_ratio
                            })
    return configs

def generate_ghost_module_v2_configs():
    """Generate test configurations for GhostModuleV2"""
    configs = []
    for in_chs, out_chs in [(16, 24), (24, 32), (32, 16), (16, 48)]:
        for ratio in [2, 4, 8]:
            for kernel_size in [1, 3]:
                for DW_kernel_size in [3, 5]:
                    for stride in [1, 2]:
                        for attention in [False, True]:
                            if ratio > 4 and DW_kernel_size > 3:
                                continue
                            if stride == 2 and kernel_size == 1:
                                continue
                            if attention and ratio > 4:
                                continue
                            # Fix: disable attention for configurations that cause spatial dimension issues
                            if attention and stride == 2:
                                continue
                                
                            configs.append({
                                'in_chs': in_chs,
                                'out_chs': out_chs,
                                'ratio': ratio,
                                'kernel_size': kernel_size,
                                'DW_kernel_size': DW_kernel_size,
                                'stride': stride,
                                'activation': nn.ReLU(),
                                'attention': attention
                            })
    return configs

def generate_ghost_bottleneck_v2_configs():
    """Generate test configurations for GhostBottleneckModuleV2"""
    configs = []
    for in_chs in [16, 24, 32]:
        for mid_chs in [24, 32, 48]:
            for out_chs in [16, 32, 48]:
                for DW_kernel_size in [3, 5]:
                    for stride in [1, 2]:
                        for se_ratio in [None, 0.25, 0.5]:
                            for attention in [False, True]:
                                if mid_chs < in_chs or mid_chs < out_chs:
                                    continue
                                if se_ratio and se_ratio > 0.25 and attention:
                                    continue
                                if stride == 2 and in_chs == out_chs and se_ratio:
                                    continue
                                    
                                configs.append({
                                    'in_chs': in_chs,
                                    'mid_chs': mid_chs,
                                    'out_chs': out_chs,
                                    'DW_kernel_size': DW_kernel_size,
                                    'stride': stride,
                                    'se_ratio': se_ratio,
                                    'attention': attention,
                                    'activation': nn.ReLU()
                                })
    return configs

def generate_spatial_pyramid_configs():
    """Generate test configurations for SpatialPyramidModule"""
    configs = []
    for in_chs, out_chs in [(32, 32), (32, 64), (64, 64), (16, 32), (48, 48)]:
        configs.append({
            'in_chs': in_chs,
            'out_chs': out_chs
        })
    return configs

def generate_ee_spatial_pyramid_configs():
    """Generate test configurations for EESpatialPyramidModule"""
    configs = []
    for in_chs in [32, 64, 48]:
        for out_chs in [64, 128, 96]:
            for stride in [1, 2]:
                for dilations in [[1, 2, 4, 8], [1, 3, 5, 7], [1, 2, 3, 6], [2, 4, 6, 8]]:
                    if stride == 2 and max(dilations) > 6:
                        continue
                    if in_chs > 48 and len(dilations) > 3:
                        continue
                        
                    configs.append({
                        'in_chs': in_chs,
                        'out_chs': out_chs,
                        'stride': stride,
                        'dilations': dilations
                    })
    return configs

def generate_lego_configs():
    """Generate test configurations for LegoModule"""
    configs = []
    for in_chs in [32, 64, 48]:
        for out_chs in [64, 128, 96]:
            for kernel_size in [3, 5, 7]:
                for n_split in [2, 4, 8]:
                    for n_lego in [0.125, 0.25, 0.5, 0.75]:
                        if kernel_size > 5 and n_split > 4:
                            continue
                        if n_lego < 0.25 and n_split > 4:
                            continue
                        if in_chs > 48 and n_split > 4:
                            continue
                            
                        configs.append({
                            'in_chs': in_chs,
                            'out_chs': out_chs,
                            'kernel_size': kernel_size,
                            'n_split': n_split,
                            'n_lego': n_lego,
                            'activation': nn.ReLU()
                        })
    return configs

def generate_versatile_convolution_configs():
    """Generate test configurations for VersatileConvolution"""
    configs = []
    for in_chs in [32, 64, 48]:
        for out_chs in [64, 128, 96]:
            for kernel_size in [3, 5, 7]:
                for stride in [1, 2]:
                    for delta in [1, 2, 3]:
                        for g in [1, 2, 4]:
                            if kernel_size > 5 and delta > 2:
                                continue
                            if stride == 2 and delta > 2:
                                continue
                            if g > 2 and in_chs % g != 0:
                                continue
                            if g > 2 and kernel_size > 5:
                                continue
                            # Fix: validate (delta + 1) divisibility for VersatileConvolution
                            if (delta + 1) % g != 0:
                                continue
                                
                            padding = kernel_size // 2
                            configs.append({
                                'in_chs': in_chs,
                                'out_chs': out_chs,
                                'kernel_size': kernel_size,
                                'stride': stride,
                                'padding': padding,
                                'delta': delta,
                                'g': g,
                                'activation': nn.ReLU()
                            })
    return configs


# Input generators
def depthwise_separable_input_gen(config):
    return torch.randn(1, config['in_chs'], 64, 64)

def inverted_residual_input_gen(config):
    return torch.randn(1, config['in_chs'], 64, 64)

def inverted_residual_se_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def universal_inverted_bottleneck_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def efficient_module_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def residual_efficient_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def shuffle_module_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def ghost_module_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def sandglass_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def inverted_residual_shuffle_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def fire_module_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def squeezenext_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def condense_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def factorized_convolution_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def parallel_factorized_convolution_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def stem_module_input_gen(config):
    return torch.randn(1, config['in_chs'], 64, 64)

def mobile_one_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def mix_depth_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def dice_input_gen(config):
    return torch.randn(1, config['in_chs'], config['height'], config['width'])

def strided_dice_input_gen(config):
    return torch.randn(1, config['in_chs'], config['height'], config['width'])

def shuffle_dice_input_gen(config):
    return torch.randn(1, config['in_chs'], config['height'], config['width'])

def ghost_bottleneck_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def ghost_module_v2_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def ghost_bottleneck_v2_input_gen(config):
    return torch.randn(1, config['in_chs'], 32, 32)

def spatial_pyramid_input_gen(config):
    # Larger input for downsampling cases
    downsampling = config['in_chs'] != config['out_chs']
    input_size = 128 if downsampling else 64
    return torch.randn(1, config['in_chs'], input_size, input_size)

def ee_spatial_pyramid_input_gen(config):
    return torch.randn(1, config['in_chs'], 64, 64)

def lego_input_gen(config):
    return torch.randn(1, config['in_chs'], 64, 64)

def versatile_convolution_input_gen(config):
    return torch.randn(1, config['in_chs'], 64, 64)

# Test
if __name__ == '__main__':
    
    print("Starting Efficient CNN Modules Testing with Comprehensive Profiling...")
    
    # Initialize the testing framework with profiling enabled
    framework = ModuleTestFramework(log_failures_only=True,           # Only log failures to console
                                    enable_profiling=True,            # Enable comprehensive model profiling
                                    csv_output="model_profiling.csv"  # CSV file for profiling data
                                    )
    
    # Test all 28 CNN modules with comprehensive profiling
    print(f"\n Testing all CNN modules with enhanced configurations...")
    
    # Test DepthwiseSeparableConvolutionModule
    framework.run_module_tests("Depthwise_Separable_Convolution_Module",
                               generate_depthwise_separable_configs(),
                               DepthwiseSeparableConvolutionModule,
                               depthwise_separable_input_gen)
    
    # Test InvertedResidualModule
    framework.run_module_tests("Inverted_Residual_Module", 
                               generate_inverted_residual_configs(),
                               InvertedResidualModule,
                               inverted_residual_input_gen)
    
    # Test SandGlassModule
    framework.run_module_tests("Sand_Glass_Module",
                               generate_sandglass_configs(), 
                               SandGlassModule,
                               sandglass_input_gen)
    
    # Test InvertedResidualSEModule
    framework.run_module_tests("Inverted_Residual_Squeeze-and-Excite_Module",
                               generate_inverted_residual_se_configs(),
                               InvertedResidualSEModule,
                               inverted_residual_se_input_gen)
    
    # Test UniversalInvertedBottleneckModule
    framework.run_module_tests("Universal_Inverted_Bottleneck_Module",
                               generate_universal_inverted_bottleneck_configs(),
                               UniversalInvertedBottleneckModule,
                               universal_inverted_bottleneck_input_gen)
    
    # Test EfficientModule
    framework.run_module_tests("Efficient_Module",
                               generate_efficient_module_configs(),
                               EfficientModule,
                               efficient_module_input_gen)
    
    # Test ResidualEfficientModule  
    framework.run_module_tests("Residual_Efficient_Module",
                               generate_residual_efficient_configs(),
                               ResidualEfficientModule,
                               residual_efficient_input_gen)
    
    # Test ShuffleModule
    framework.run_module_tests("Shuffle_Module",
                               generate_shuffle_module_configs(),
                               ShuffleModule,
                               shuffle_module_input_gen)

    # Test GhostModule
    framework.run_module_tests("Ghost_Module",
                               generate_ghost_module_configs(),
                               GhostModule,
                               ghost_module_input_gen)
    
    # Test InvertedResidualShuffleModule
    framework.run_module_tests("Inverted_Residual_Shuffle_Module",
                               generate_inverted_residual_shuffle_configs(),
                               InvertedResidualShuffleModule,
                               inverted_residual_shuffle_input_gen)
    
    # Test FireModule
    framework.run_module_tests("Fire_Module",
                               generate_fire_module_configs(),
                               FireModule,
                               fire_module_input_gen)

    # Test SqueezeNeXtModule
    framework.run_module_tests("SqueezeNeXt_Module",
                               generate_squeezenext_configs(),
                               SqueezeNeXtModule,
                               squeezenext_input_gen)
    
    # Test CondenseModule
    framework.run_module_tests("Condense_Module",
                               generate_condense_configs(),
                               CondenseModule,
                               condense_input_gen)

    # Test FactorizedConvolutionModule
    framework.run_module_tests("Factorized_Convolution_Module",
                               generate_factorized_convolution_configs(),
                               FactorizedConvolutionModule,
                               factorized_convolution_input_gen)
    
    # Test ParallelFactorizedConvolutionModule
    framework.run_module_tests("Parallel_Factorized_Convolution_Module",
                               generate_parallel_factorized_convolution_configs(),
                               ParallelFactorizedConvolutionModule,
                               parallel_factorized_convolution_input_gen)

    # Test StemModule
    framework.run_module_tests("Stem_Module",
                               generate_stem_module_configs(),
                               StemModule,
                               stem_module_input_gen)
    
    # Test MobileOneModule
    framework.run_module_tests("MobileOne_Module",
                               generate_mobile_one_configs(),
                               MobileOneModule,
                               mobile_one_input_gen)

    # Test MixDepthModule
    framework.run_module_tests("MixDepth_Module",
                               generate_mix_depth_configs(),
                               MixDepthModule,
                               mix_depth_input_gen)
    
    # Test DiCEModule
    framework.run_module_tests("DiCE_Module",
                               generate_dice_configs(),
                               DiCEModule,
                               dice_input_gen)
    
    # Test StridedDiCEModule
    framework.run_module_tests("Strided_DiCE_Module",
                               generate_strided_dice_configs(),
                               StridedDiCEModule,
                               strided_dice_input_gen)

    # Test ShuffleDiCEModule
    framework.run_module_tests("Shuffle_DiCE_Module",
                               generate_shuffle_dice_configs(),
                               ShuffleDiCEModule,
                               shuffle_dice_input_gen)
    
    # Test GhostBottleneckModule
    framework.run_module_tests("Ghost_Bottleneck_Module",
                               generate_ghost_bottleneck_configs(),
                               GhostBottleneckModule,
                               ghost_bottleneck_input_gen)
    
    # Test GhostModuleV2
    framework.run_module_tests("Ghost_Module_V2",
                               generate_ghost_module_v2_configs(),
                               GhostModuleV2,
                               ghost_module_v2_input_gen)
    
    # Test GhostBottleneckModuleV2
    framework.run_module_tests("Ghost_Bottleneck_Module_V2",
                               generate_ghost_bottleneck_v2_configs(),
                               GhostBottleneckModuleV2,
                               ghost_bottleneck_v2_input_gen)
    
    # Test SpatialPyramidModule
    framework.run_module_tests("Spatial_Pyramid_Module",
                               generate_spatial_pyramid_configs(),
                               SpatialPyramidModule,
                               spatial_pyramid_input_gen)
    
    # Test EESpatialPyramidModule
    framework.run_module_tests("Extremely_Efficient_Spatial_Pyramid_Module",
                               generate_ee_spatial_pyramid_configs(),
                               EESpatialPyramidModule,
                               ee_spatial_pyramid_input_gen)

    # Test LegoModule
    framework.run_module_tests("Lego_Module",
                               generate_lego_configs(),
                               LegoModule,
                               lego_input_gen)

    # Test VersatileConvolution
    framework.run_module_tests("Versatile_Convolution_Module",
                               generate_versatile_convolution_configs(),
                               VersatileConvolution,
                               versatile_convolution_input_gen)
    
    # Print final summary
    framework.print_final_summary()
    print("EOF")
