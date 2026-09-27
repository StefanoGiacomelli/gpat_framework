"""
YAMNet (Yet Another Mobile Network for Audio)
==============================================
MobileNet-style CNN for audio classification trained on AudioSet.

Original repository: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
PyTorch port: https://github.com/w-hc/torch_audioset

This is a standalone implementation of YAMNet for AudioSet tagging.

Note: The original YAMNet was trained on 521 AudioSet classes (6 classes removed).
This implementation expands the output to 527 classes for compatibility with other
AudioSet models. The 6 missing classes have randomly initialized weights.
"""

import numpy as np
from typing import Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio.transforms as T


# ============================================
# CONSTANTS
# ============================================
SAMPLE_RATE = 16000
CLASSES_NUM = 527  # Full AudioSet (expanded from original 521)
EMBED_DIM = 1024

# Original YAMNet had 521 classes - these 6 AudioSet indices were NOT trained
# They will have randomly initialized weights
NOT_PRETRAINED_CLASSES: Set[int] = {1, 2, 12, 32, 33, 277}
PRETRAINED_OUTPUT_INDICES = tuple(i for i in range(CLASSES_NUM) if i not in NOT_PRETRAINED_CLASSES)
# idx=1: Male speech, man speaking
# idx=2: Female speech, woman speaking  
# idx=12: Battle cry
# idx=32: Male singing
# idx=33: Female singing
# idx=277: Funny music

# Mel spectrogram params
NUM_MEL_BINS = 64
NUM_FRAMES = 96  # 0.96 s analysis window
PATCH_HOP_FRAMES = 48  # 0.48 s YAMNet patch hop
STFT_WINDOW_SECONDS = 0.025
STFT_HOP_SECONDS = 0.010
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001
BATCHNORM_EPSILON = 1e-4

# Derived params
WINDOW_LENGTH_SAMPLES = int(round(SAMPLE_RATE * STFT_WINDOW_SECONDS))  # 400
HOP_LENGTH_SAMPLES = int(round(SAMPLE_RATE * STFT_HOP_SECONDS))  # 160
FFT_LENGTH = 512


# ============================================
# YAMNET TO AUDIOSET INDEX MAPPING
# ============================================
# YAMNet output[i] -> AudioSet index
# This mapping is needed because YAMNet removed 6 classes and re-indexed
YAMNET_TO_AUDIOSET = {
    0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11,
    10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19, 17: 20, 18: 21, 19: 22,
    20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 45, 41: 46, 42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54,
    50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63, 59: 64,
    60: 65, 61: 66, 62: 67, 63: 68, 64: 69, 65: 70, 66: 71, 67: 72, 68: 73, 69: 74,
    70: 75, 71: 76, 72: 77, 73: 78, 74: 79, 75: 80, 76: 81, 77: 82, 78: 83, 79: 84,
    80: 85, 81: 86, 82: 87, 83: 88, 84: 89, 85: 90, 86: 91, 87: 92, 88: 93, 89: 94,
    90: 95, 91: 96, 92: 97, 93: 98, 94: 99, 95: 100, 96: 101, 97: 102, 98: 103, 99: 104,
    100: 105, 101: 106, 102: 107, 103: 108, 104: 109, 105: 110, 106: 111, 107: 112, 108: 113, 109: 114,
    110: 115, 111: 116, 112: 117, 113: 118, 114: 119, 115: 120, 116: 121, 117: 122, 118: 123, 119: 124,
    120: 125, 121: 126, 122: 127, 123: 128, 124: 129, 125: 130, 126: 131, 127: 132, 128: 133, 129: 134,
    130: 135, 131: 136, 132: 137, 133: 138, 134: 139, 135: 140, 136: 141, 137: 142, 138: 143, 139: 144,
    140: 145, 141: 146, 142: 147, 143: 148, 144: 149, 145: 150, 146: 151, 147: 152, 148: 153, 149: 154,
    150: 155, 151: 156, 152: 157, 153: 158, 154: 159, 155: 160, 156: 161, 157: 162, 158: 163, 159: 164,
    160: 165, 161: 166, 162: 167, 163: 168, 164: 169, 165: 170, 166: 171, 167: 172, 168: 173, 169: 174,
    170: 175, 171: 176, 172: 177, 173: 178, 174: 179, 175: 180, 176: 181, 177: 182, 178: 183, 179: 184,
    180: 185, 181: 186, 182: 187, 183: 188, 184: 189, 185: 190, 186: 191, 187: 192, 188: 193, 189: 194,
    190: 195, 191: 196, 192: 197, 193: 198, 194: 199, 195: 200, 196: 201, 197: 202, 198: 203, 199: 204,
    200: 205, 201: 206, 202: 207, 203: 208, 204: 209, 205: 210, 206: 211, 207: 212, 208: 213, 209: 214,
    210: 215, 211: 216, 212: 217, 213: 218, 214: 219, 215: 220, 216: 221, 217: 222, 218: 223, 219: 224,
    220: 225, 221: 226, 222: 227, 223: 228, 224: 229, 225: 230, 226: 231, 227: 232, 228: 233, 229: 234,
    230: 235, 231: 236, 232: 237, 233: 238, 234: 239, 235: 240, 236: 241, 237: 242, 238: 243, 239: 244,
    240: 245, 241: 246, 242: 247, 243: 248, 244: 249, 245: 250, 246: 251, 247: 252, 248: 253, 249: 254,
    250: 255, 251: 256, 252: 257, 253: 258, 254: 259, 255: 260, 256: 261, 257: 262, 258: 263, 259: 264,
    260: 265, 261: 266, 262: 267, 263: 268, 264: 269, 265: 270, 266: 271, 267: 272, 268: 273, 269: 274,
    270: 275, 271: 276, 272: 278, 273: 279, 274: 280, 275: 281, 276: 282, 277: 283, 278: 284, 279: 285,
    280: 286, 281: 287, 282: 288, 283: 289, 284: 290, 285: 291, 286: 292, 287: 293, 288: 294, 289: 295,
    290: 296, 291: 297, 292: 298, 293: 299, 294: 300, 295: 301, 296: 302, 297: 303, 298: 304, 299: 305,
    300: 306, 301: 307, 302: 308, 303: 309, 304: 310, 305: 311, 306: 312, 307: 313, 308: 314, 309: 315,
    310: 316, 311: 317, 312: 318, 313: 319, 314: 320, 315: 321, 316: 322, 317: 323, 318: 324, 319: 325,
    320: 326, 321: 327, 322: 328, 323: 329, 324: 330, 325: 331, 326: 332, 327: 333, 328: 334, 329: 335,
    330: 336, 331: 337, 332: 338, 333: 339, 334: 340, 335: 341, 336: 342, 337: 343, 338: 344, 339: 345,
    340: 346, 341: 347, 342: 348, 343: 349, 344: 350, 345: 351, 346: 352, 347: 353, 348: 354, 349: 355,
    350: 356, 351: 357, 352: 358, 353: 359, 354: 360, 355: 361, 356: 362, 357: 363, 358: 364, 359: 365,
    360: 366, 361: 367, 362: 368, 363: 369, 364: 370, 365: 371, 366: 372, 367: 373, 368: 374, 369: 375,
    370: 376, 371: 377, 372: 378, 373: 379, 374: 380, 375: 381, 376: 382, 377: 383, 378: 384, 379: 385,
    380: 386, 381: 387, 382: 388, 383: 389, 384: 390, 385: 391, 386: 392, 387: 393, 388: 394, 389: 395,
    390: 396, 391: 397, 392: 398, 393: 399, 394: 400, 395: 401, 396: 402, 397: 403, 398: 404, 399: 405,
    400: 406, 401: 407, 402: 408, 403: 409, 404: 410, 405: 411, 406: 412, 407: 413, 408: 414, 409: 415,
    410: 416, 411: 417, 412: 418, 413: 419, 414: 420, 415: 421, 416: 422, 417: 423, 418: 424, 419: 425,
    420: 426, 421: 427, 422: 428, 423: 429, 424: 430, 425: 431, 426: 432, 427: 433, 428: 434, 429: 435,
    430: 436, 431: 437, 432: 438, 433: 439, 434: 440, 435: 441, 436: 442, 437: 443, 438: 444, 439: 445,
    440: 446, 441: 447, 442: 448, 443: 449, 444: 450, 445: 451, 446: 452, 447: 453, 448: 454, 449: 455,
    450: 456, 451: 457, 452: 458, 453: 459, 454: 460, 455: 461, 456: 462, 457: 463, 458: 464, 459: 465,
    460: 466, 461: 467, 462: 468, 463: 469, 464: 470, 465: 471, 466: 472, 467: 473, 468: 474, 469: 475,
    470: 476, 471: 477, 472: 478, 473: 479, 474: 480, 475: 481, 476: 482, 477: 483, 478: 484, 479: 485,
    480: 486, 481: 487, 482: 488, 483: 489, 484: 490, 485: 491, 486: 492, 487: 493, 488: 494, 489: 495,
    490: 496, 491: 497, 492: 498, 493: 499, 494: 500, 495: 501, 496: 502, 497: 503, 498: 504, 499: 505,
    500: 506, 501: 507, 502: 508, 503: 509, 504: 510, 505: 511, 506: 512, 507: 513, 508: 514, 509: 515,
    510: 516, 511: 517, 512: 518, 513: 519, 514: 520, 515: 521, 516: 522, 517: 523, 518: 524, 519: 525,
    520: 526,
}


# ============================================
# Preprocessing
# ============================================
class YAMNetLogMelSpectrogram(nn.Module):
    """Log Mel Spectrogram following YAMNet preprocessing.
    
    Note: The order of operations is important:
    1. STFT (power spectrogram)
    2. sqrt (convert power to magnitude)
    3. Mel filterbank
    4. log compression
    
    This matches the original TensorFlow YAMNet implementation.
    """
    
    def __init__(self):
        super().__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=FFT_LENGTH,
            win_length=WINDOW_LENGTH_SAMPLES,
            hop_length=HOP_LENGTH_SAMPLES,
            power=2.0
        )
        self.mel_scale = T.MelScale(
            n_mels=NUM_MEL_BINS,
            sample_rate=SAMPLE_RATE,
            n_stft=FFT_LENGTH // 2 + 1,
            f_min=MEL_MIN_HZ,
            f_max=MEL_MAX_HZ
        )
    
    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Step 1: Power spectrogram
        spec = self.spectrogram(waveform)
        
        # Step 2: Convert power to magnitude (sqrt)
        spec = spec ** 0.5
        
        # Step 3: Apply mel filterbank
        mel = self.mel_scale(spec)
        
        # Step 4: Log compression
        log_mel = torch.log(mel + LOG_OFFSET)
        
        # Transpose to (batch, 1, time, freq) for CNN input
        log_mel = log_mel.unsqueeze(1).transpose(2, 3)
        
        return log_mel


class AudioPreprocessor(nn.Module):
    """Preprocess waveform to YAMNet input patches."""
    
    def __init__(self):
        super().__init__()
        self.log_mel = YAMNetLogMelSpectrogram()
        self.resample = None
        self._cached_sample_rate = None
    
    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        
        if sample_rate != SAMPLE_RATE:
            if self._cached_sample_rate != sample_rate:
                self.resample = T.Resample(sample_rate, SAMPLE_RATE).to(waveform.device)
                self._cached_sample_rate = sample_rate
            waveform = self.resample(waveform)
        
        log_mel = self.log_mel(waveform)
        
        batch_size = log_mel.shape[0]
        time_frames = log_mel.shape[2]

        if time_frames < NUM_FRAMES:
            log_mel = F.pad(log_mel, (0, 0, 0, NUM_FRAMES - time_frames))

        # YAMNet uses overlapping 0.96 s patches with a 0.48 s hop.
        patches = log_mel.unfold(
            dimension=2,
            size=NUM_FRAMES,
            step=PATCH_HOP_FRAMES,
        )
        # [B, 1, N, F, T] -> [B*N, 1, T, F]
        patches = patches.permute(0, 2, 1, 4, 3).contiguous()
        num_patches = patches.shape[1]
        patches = patches.reshape(
            batch_size * num_patches,
            1,
            NUM_FRAMES,
            NUM_MEL_BINS,
        )

        return patches


# ============================================
# Helper Functions
# ============================================
class Conv2d_tf(nn.Conv2d):
    """Conv2d with TensorFlow 'SAME' padding behavior."""
    
    def __init__(self, *args, **kwargs):
        kwargs.pop("padding", None)
        super().__init__(*args, **kwargs)
    
    def _compute_padding(self, input_size: int, dim: int) -> Tuple[int, int]:
        filter_size = self.kernel_size[dim]
        stride = self.stride[dim] if isinstance(self.stride, tuple) else self.stride
        dilation = self.dilation[dim] if isinstance(self.dilation, tuple) else self.dilation
        
        effective_kernel_size = (filter_size - 1) * dilation + 1
        out_size = (input_size + stride - 1) // stride
        total_padding = max(0, (out_size - 1) * stride + effective_kernel_size - input_size)
        
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before
        return pad_before, pad_after
    
    def forward(self, x: Tensor) -> Tensor:
        pad_h = self._compute_padding(x.size(2), 0)
        pad_w = self._compute_padding(x.size(3), 1)
        
        if pad_h[1] > 0 or pad_w[1] > 0:
            x = F.pad(x, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]])
        else:
            x = F.pad(x, [pad_w[0], pad_w[0], pad_h[0], pad_h[0]])
        
        return F.conv2d(x, self.weight, self.bias, self.stride, 
                       padding=0, dilation=self.dilation, groups=self.groups)


# ============================================
# Building Blocks
# ============================================
class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU."""
    
    def __init__(self, conv: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels, eps=BATCHNORM_EPSILON)
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    """Standard convolution block."""
    
    def __init__(self, kernel: list, stride: int, input_dim: int, output_dim: int):
        super().__init__()
        self.fused = ConvBnRelu(
            Conv2d_tf(input_dim, output_dim, kernel_size=kernel, stride=stride, bias=False)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fused(x)


class SeparableConv(nn.Module):
    """Depthwise separable convolution block."""
    
    def __init__(self, kernel: list, stride: int, input_dim: int, output_dim: int):
        super().__init__()
        self.depthwise_conv = ConvBnRelu(
            Conv2d_tf(input_dim, input_dim, kernel_size=kernel, stride=stride, 
                     groups=input_dim, bias=False)
        )
        self.pointwise_conv = ConvBnRelu(
            Conv2d_tf(input_dim, output_dim, kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# ============================================
# Backbone
# ============================================
class YAMNetBackbone(nn.Module):
    """MobileNet-style backbone for YAMNet."""
    
    def __init__(self):
        super().__init__()
        
        # Network configuration: (layer_type, kernel, stride, output_channels)
        net_configs = [
            (Conv,          [3, 3], 2,   32),
            (SeparableConv, [3, 3], 1,   64),
            (SeparableConv, [3, 3], 2,  128),
            (SeparableConv, [3, 3], 1,  128),
            (SeparableConv, [3, 3], 2,  256),
            (SeparableConv, [3, 3], 1,  256),
            (SeparableConv, [3, 3], 2,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 2, 1024),
            (SeparableConv, [3, 3], 1, 1024)
        ]
        
        input_dim = 1
        self.layer_names = []
        for i, (layer_mod, kernel, stride, output_dim) in enumerate(net_configs):
            name = f'layer{i + 1}'
            self.add_module(name, layer_mod(kernel, stride, input_dim, output_dim))
            input_dim = output_dim
            self.layer_names.append(name)
    
    def forward(self, x: Tensor) -> Tensor:
        for name in self.layer_names:
            x = getattr(self, name)(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.shape[0], -1)
        return x


# ============================================
# Main Model
# ============================================
class YAMNet(nn.Module):
    """
    YAMNet wrapper with standard API for audio classification.
    
    Note: Output is expanded to 527 classes for AudioSet compatibility.
    The original YAMNet had 521 classes - 6 classes have random weights.
    See NOT_PRETRAINED_CLASSES for the list of affected class indices.
    
    Args:
        sample_rate: Expected input sample rate (default: 16000)
    
    Standard API:
        - forward(waveform) -> probs (527 AudioSet classes)
        - forward_with_embedding(waveform) -> (probs, embedding)
        - get_embedding(waveform) -> embedding (1024 dim)
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor()
        self.backbone = YAMNetBackbone()
        # Expanded to 527 classes for AudioSet compatibility
        self.classifier = nn.Linear(EMBED_DIM, CLASSES_NUM, bias=True)
        # Preserve the 527-unit head, but expose which rows are actually pretrained.
        self.evaluation_valid_output_indices = PRETRAINED_OUTPUT_INDICES
        self.evaluation_classifier_pretrained = True
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained YAMNet checkpoint with weight remapping.
        
        The original checkpoint has 521 output classes. This method maps
        the weights to our expanded 527-class classifier.
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Initialize classifier with Xavier (for the 6 missing classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Map backbone weights
        backbone_state = {}
        classifier_weight_521 = None
        classifier_bias_521 = None
        
        for k, v in state_dict.items():
            if k == 'classifier.weight':
                classifier_weight_521 = v
            elif k == 'classifier.bias':
                classifier_bias_521 = v
            else:
                backbone_state[f'backbone.{k}'] = v
        
        # Load backbone weights
        self.load_state_dict(backbone_state, strict=False)
        
        # Remap classifier weights from 521 -> 527
        if classifier_weight_521 is not None:
            for yamnet_idx, audioset_idx in YAMNET_TO_AUDIOSET.items():
                self.classifier.weight.data[audioset_idx] = classifier_weight_521[yamnet_idx]
                if classifier_bias_521 is not None:
                    self.classifier.bias.data[audioset_idx] = classifier_bias_521[yamnet_idx]
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def forward(self, waveform: Tensor) -> Tensor:
        """
        Forward pass returning AudioSet class probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at any sample rate
        Returns:
            probs: (batch, 527) AudioSet class probabilities
        """
        probs, _ = self.forward_with_embedding(waveform)
        return probs
    
    def forward_with_embedding(self, waveform: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at any sample rate
        Returns:
            probs: (batch, 527) AudioSet class probabilities
            embedding: (batch, 1024) feature embedding
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        
        # Preprocess to patches
        patches = self.preprocessor(waveform, self.sample_rate)
        
        # Get embeddings for each patch
        embeddings = self.backbone(patches)
        
        # Get logits for each patch
        logits = self.classifier(embeddings)
        
        # YAMNet averages patch-level scores, not logits, at clip level.
        batch_size = waveform.shape[0]
        num_patches = patches.shape[0] // batch_size
        embeddings = embeddings.reshape(batch_size, num_patches, -1)
        patch_probs = torch.sigmoid(logits).reshape(batch_size, num_patches, -1)

        clip_embedding = embeddings.mean(dim=1)
        probs = patch_probs.mean(dim=1)

        return probs, clip_embedding
    
    def get_embedding(self, waveform: Tensor) -> Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at any sample rate
        Returns:
            embedding: (batch, 1024) feature embedding
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
    print("YAMNet (MobileNet-style Audio CNN) - Demo")
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
        labels = {int(row[0]): row[2] for row in reader}
    print(f"   Loaded {len(labels)} labels")
    
    # 3. Load audio
    print("\n3. Loading audio...")
    audio_path = "/Users/stefano/Documents/PhD_main_project/utils/R9_ZSCveAHg_7s.wav"
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    
    print(f"   Duration: {waveform.shape[0]/SAMPLE_RATE:.2f}s")
    waveform = waveform.to(device)
    
    # 4. Create model and load checkpoint
    print("\n4. Creating model...")
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/yamnet/yamnet.pth"
    model = YAMNet(sample_rate=SAMPLE_RATE)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 5. Inference
    print("\n5. Running inference...")
    with torch.no_grad():
        probs = model(waveform)
    print(f"   Output shape: {probs.shape}")
    
    # 6. Top-10 predictions (filtering out not-pretrained classes)
    print("\n6. Top-10 predictions (pretrained classes only):")
    print(f"   Note: Filtering out {len(NOT_PRETRAINED_CLASSES)} classes with random weights")
    print(f"   (indices: {sorted(NOT_PRETRAINED_CLASSES)})")
    probs_np = probs[0].cpu().numpy()
    
    # Filter out not-pretrained classes
    pretrained_indices = [i for i in range(CLASSES_NUM) if i not in NOT_PRETRAINED_CLASSES]
    pretrained_probs = [(i, probs_np[i]) for i in pretrained_indices]
    pretrained_probs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (idx, prob) in enumerate(pretrained_probs[:10]):
        print(f"   {i+1:2d}. {labels[idx]:<40} {prob:.4f}")
    
    # 7. Embedding extraction
    print("\n7. Extracting embedding...")
    with torch.no_grad():
        embedding = model.get_embedding(waveform)
    print(f"   Embedding shape: {embedding.shape}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
