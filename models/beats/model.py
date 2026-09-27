"""
BEATs (Audio Pre-Training with Acoustic Tokenizers)
====================================================
Transformer encoder with acoustic tokenizer pre-training for audio classification.

Original repository: https://github.com/microsoft/unilm/tree/master/beats

This is a standalone implementation of BEATs for AudioSet tagging.
"""

import math
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi

from models.audioset_labels import canonical_reorder_index, load_canonical_audioset_mids


# ============================================
# CONSTANTS (hardcoded for BEATs iter3+ AS2M)
# ============================================
SAMPLE_RATE = 16000
CLASSES_NUM = 527
EMBED_DIM = 768

# Mel spectrogram params (Kaldi-style fbank)
NUM_MEL_BINS = 128
FRAME_LENGTH = 25  # ms
FRAME_SHIFT = 10   # ms

# BEATs normalization stats
FBANK_MEAN = 15.41663
FBANK_STD = 6.55582

# Architecture params (iter3+ AS2M finetuned)
INPUT_PATCH_SIZE = 16
ENCODER_LAYERS = 12
ENCODER_EMBED_DIM = 768
ENCODER_FFN_EMBED_DIM = 3072
ENCODER_ATTENTION_HEADS = 12


# ============================================
# Helper Functions
# ============================================
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super().__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = nn.SiLU()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()

        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)

    def forward(self, x):
        x = self.linear(x)
        if self.glu_type == "bilinear":
            x = x[:, :, :self.output_dim] * x[:, :, self.output_dim:]
        else:
            x = x[:, :, :self.output_dim] * self.glu_act(x[:, :, self.output_dim:])
        return x


def get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError(f"activation {activation} not supported")


def quant_noise(module, p, block_size):
    """Quantization noise wrapper (no-op for inference with p=0)."""
    if p <= 0:
        return module
    return module


# ============================================
# MULTIHEAD ATTENTION
# ============================================
class MultiheadAttention(nn.Module):
    """Multi-headed attention with relative position bias."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        self_attention=True,
        has_relative_attention_bias=False,
        num_buckets=32,
        max_distance=128,
        gru_rel_pos=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_module = nn.Dropout(dropout)

        self.gru_rel_pos = gru_rel_pos
        if gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, 
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(
                bsz * self.num_heads, tgt_len, src_len
            )

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = q * self.scaling
        alpha = 32
        q = q * (1 / alpha)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.head_dim) * alpha / self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias

            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn, None, position_bias


# ============================================
# TRANSFORMER ENCODER LAYER
# ============================================
class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        deep_norm: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        gru_rel_pos: bool = False,
        encoder_layers: int = 12,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)

        self.self_attn = MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(embedding_dim)

        if activation_fn == "glu":
            self.fc1 = GLU_Linear(embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

        self.final_layer_norm = LayerNorm(embedding_dim)

        self.deep_norm = deep_norm
        if deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
        self,
        x: Tensor,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        pos_bias: Optional[Tensor] = None,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, _, pos_bias = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, _, pos_bias = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x
            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, None, pos_bias


# ============================================
# TRANSFORMER ENCODER
# ============================================
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        std = math.sqrt(4.0 / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.parametrizations.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.relative_position_embedding = getattr(args, 'relative_position_embedding', False)
        self.num_buckets = getattr(args, 'num_buckets', 0)
        self.max_distance = getattr(args, 'max_distance', 0)

        self.layers = nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                deep_norm=args.deep_norm,
                has_relative_attention_bias=self.relative_position_embedding,
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
                gru_rel_pos=args.gru_rel_pos,
                encoder_layers=args.encoder_layers,
            )
            for _ in range(args.encoder_layers)
        ])

        if self.relative_position_embedding:
            for i in range(1, args.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[0].self_attn.relative_attention_bias

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(self._init_bert_params)

        if args.deep_norm:
            deep_norm_beta = math.pow(8 * args.encoder_layers, -1 / 4)
            for i in range(args.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)

        self.layer_wise_gradient_decay_ratio = getattr(args, 'layer_wise_gradient_decay_ratio', 1.0)

    def _init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            dropout_probability = np.random.random() if self.training else 0
            if not self.training or dropout_probability > self.layerdrop:
                x, _, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, pos_bias=pos_bias)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


# ============================================
# BEATS CONFIG
# ============================================
class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = 16
        self.embed_dim: int = 512
        self.conv_bias: bool = False

        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"

        self.layer_wise_gradient_decay_ratio: float = 1.0
        self.layer_norm_first: bool = False
        self.deep_norm: bool = False

        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.encoder_layerdrop: float = 0.0
        self.dropout_input: float = 0.0

        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16

        self.relative_position_embedding: bool = False
        self.num_buckets: int = 320
        self.max_distance: int = 1280
        self.gru_rel_pos: bool = False

        self.finetuned_model: bool = False
        self.predictor_dropout: float = 0.1
        self.predictor_class: int = 527

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


# ============================================
# BEATS MODEL
# ============================================
class BEATsModel(nn.Module):
    def __init__(self, cfg: BEATsConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1, self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias,
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(self, features: Tensor, padding_mask: Tensor) -> Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(
        self,
        source: Tensor,
        fbank_mean: float = FBANK_MEAN,
        fbank_std: float = FBANK_STD,
    ) -> Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=NUM_MEL_BINS,
                sample_frequency=SAMPLE_RATE,
                frame_length=FRAME_LENGTH,
                frame_shift=FRAME_SHIFT,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(
        self,
        source: Tensor,
        padding_mask: Optional[Tensor] = None,
    ):
        fbank = self.preprocess(source)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)
        x, _ = self.encoder(x, padding_mask=padding_mask)

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            return logits, x.mean(dim=1)
        else:
            return x, x.mean(dim=1)


# ============================================
# Main Model
# ============================================
class BEATs(nn.Module):
    """
    BEATs wrapper class with standard interface.
    
    Usage:
        model = BEATs()
        model.load_pretrained('checkpoint.pt')
        probs = model(waveform)  # sigmoid probabilities
        embedding = model.get_embedding(waveform)
    """
    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.model = None
        self.label_dict = None
        self.source_label_dict = None
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATsModel(cfg)
        self.model.load_state_dict(checkpoint['model'])
        
        if 'label_dict' not in checkpoint:
            raise RuntimeError("BEATs checkpoint does not contain label_dict; class-order fidelity cannot be established")

        self.source_label_dict = checkpoint['label_dict']
        source_mids = [
            self.source_label_dict[index]
            if index in self.source_label_dict
            else self.source_label_dict[str(index)]
            for index in range(CLASSES_NUM)
        ]
        canonical_mids = load_canonical_audioset_mids()
        permutation = canonical_reorder_index(source_mids, canonical_mids=canonical_mids)

        if self.model.predictor is None or self.model.predictor.out_features != CLASSES_NUM:
            raise RuntimeError("BEATs checkpoint does not expose the expected 527-class predictor")

        with torch.no_grad():
            weight = self.model.predictor.weight.detach().clone()
            bias = self.model.predictor.bias.detach().clone() if self.model.predictor.bias is not None else None
            self.model.predictor.weight.copy_(weight.index_select(0, permutation))
            if bias is not None:
                self.model.predictor.bias.copy_(bias.index_select(0, permutation))

        # Public GP-AT outputs now follow datasets/AudioSet_meta/class_labels_indices.csv.
        self.label_dict = {index: mid for index, mid in enumerate(canonical_mids)}
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print("Reordered BEATs predictor from checkpoint MID order to canonical AudioSet order")
    
    def forward(self, waveform: Tensor) -> Tensor:
        """
        Forward pass returning sigmoid probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at 16kHz
        Returns:
            probs: (batch, 527) sigmoid probabilities
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(next(self.model.parameters()).device)
        logits, _ = self.model.extract_features(waveform)
        return torch.sigmoid(logits)
    
    def forward_with_embedding(self, waveform: Tensor):
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at 16kHz
        Returns:
            probs: (batch, 527) sigmoid probabilities
            embedding: (batch, 768) feature embedding
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(next(self.model.parameters()).device)
        logits, embedding = self.model.extract_features(waveform)
        return torch.sigmoid(logits), embedding
    
    def get_embedding(self, waveform: Tensor) -> Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at 16kHz
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
    print("BEATs (Audio Pre-Training with Acoustic Tokenizers) - Demo")
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
    checkpoint_path = "/Users/stefano/Documents/PhD_main_project/models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    model = BEATs(sample_rate=SAMPLE_RATE)
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
