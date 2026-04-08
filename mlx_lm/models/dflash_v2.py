# Copyright © 2025 Apple Inc.

# DFlash Draft Model - Ported from reference implementation
# Reference: https://github.com/jianc99/dflash

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .dflash_cache import CroppableKVCache, DFlashCacheManager
from .base import scaled_dot_product_attention
from .activations import swiglu


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Build target_layer_ids for DFlash draft model.

    Selects layer indices spread across the target model to extract features from.
    """
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]

def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    # cos/sin: [seq_len, head_dim/2] - duplicate for full head_dim
    # Expand dims for broadcasting: [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2]
    cos = mx.expand_dims(cos, 0)
    cos = mx.expand_dims(cos, 0)
    sin = mx.expand_dims(sin, 0)
    sin = mx.expand_dims(sin, 0)

    # Concatenate cos/sin with themselves to match head_dim
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)

    q_len = q.shape[-2]
    k_len = k.shape[-2]
    cos_q = cos[..., -q_len:, :]
    sin_q = sin[..., -q_len:, :]
    cos_k = cos[..., -k_len:, :]
    sin_k = sin[..., -k_len:, :]

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

    return q_embed, k_embed


@dataclass
class ModelArgs:
    model_type: str = "qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000
    rope_traditional: bool = False
    attention_bias: bool = False
    vocab_size: int = 248320
    tie_word_embeddings: bool = True
    # DFlash-specific
    block_size: int = 16
    num_target_layers: int = 32
    target_layer_ids: Optional[List[int]] = None
    mask_token_id: Optional[int] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any], weights: Dict[str, mx.array] = None):
        # Extract DFlash config
        dflash_config = params.pop("dflash_config", None)
        if dflash_config:
            params.setdefault("block_size", dflash_config.get("block_size", 16))
            params.setdefault("num_target_layers", dflash_config.get("num_target_layers", 32))
            params.setdefault("target_layer_ids", dflash_config.get("target_layer_ids"))
            params.setdefault("mask_token_id", dflash_config.get("mask_token_id"))

        # Detect actual dimensions from weights if available
        # The HuggingFace configs have wrong hidden_size - detect from actual weights
        if weights is not None and "layers.0.self_attn.q_proj.weight" in weights:
            q_proj_shape = weights["layers.0.self_attn.q_proj.weight"].shape
            # Shape is (out_dim, in_dim)
            # out_dim = num_heads * head_dim (total Q size)
            # in_dim = hidden_size (input hidden size)
            actual_hidden_size = q_proj_shape[1]
            q_out_dim = q_proj_shape[0]
            # Compute head_dim from q_out_dim and num_heads
            if "num_attention_heads" in params:
                head_dim = q_out_dim // params["num_attention_heads"]
                # Override hidden_size to match actual weights
                params["hidden_size"] = actual_hidden_size

        # Filter out unsupported parameters
        filtered_params = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_params)


class DFlashAttention(nn.Module):
    """DFlash dual attention layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size  # = 2560
        self.num_heads = args.num_attention_heads  # = 32
        self.num_key_value_heads = args.num_key_value_heads  # = 8

        # DFlash uses head_dim = 128, not hidden_size // num_heads
        # Q: 4096 = 32 * 128
        # K/V: 1024 = 8 * 128
        self.head_dim = 128
        self.q_proj_size = self.num_heads * self.head_dim  # = 4096
        self.kv_proj_size = self.num_key_value_heads * self.head_dim  # = 1024

        self.scaling = self.head_dim ** -0.5

        # Projections have different output dimensions
        self.q_proj = nn.Linear(self.hidden_size, self.q_proj_size, bias=args.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_proj_size, bias=args.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_proj_size, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.q_proj_size, self.hidden_size, bias=args.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def kv_proj_only(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        """Project hidden_states to K/V only (skip Q).

        Used by DFlash to materialize ctx tokens into the draft KV cache.

        Args:
            hidden_states: [B, L, D] - Input hidden states

        Returns:
            (k, v) - K/V projections [n_kv_heads, B, L, head_dim] each
        """
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape and transpose for multi-head attention (like in __call__)
        B, L = k.shape[:2]
        k = k.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        return k, v

    def apply_k_norm(self, k: mx.array) -> mx.array:
        """Apply K RMS norm without Q dependency.

        Args:
            k: [n_kv_heads, B, L, head_dim] - Key projections

        Returns:
            Normalized k in same shape
        """
        original_shape = k.shape
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.reshape(original_shape)

    def apply_k_rope(self, positions: mx.array, k: mx.array, rotary_emb: nn.Module) -> mx.array:
        """Apply RoPE to K using the model's RoPE module.

        Args:
            positions: [B, L] - Position indices
            k: [n_kv_heads, B, L, head_dim] - Key projections
            rotary_emb: The RoPE module from the model

        Returns:
            RoPE-applied k in same shape
        """
        n_kv_heads, B, L, head_dim = k.shape

        # Create dummy query to get cos/sin from RoPE module
        dummy_q = mx.zeros((B, L, 1, head_dim), k.dtype)

        # Use the actual RoPE module to compute cos/sin
        cos, sin = rotary_emb(dummy_q, positions)

        # Expand cos/sin for broadcasting with k
        # cos, sin have shape (L, head_dim/2), need to duplicate and expand
        cos = mx.concatenate([cos, cos], axis=-1)  # (L, head_dim)
        sin = mx.concatenate([sin, sin], axis=-1)  # (L, head_dim)

        # Expand dims for broadcasting: (L, head_dim) -> (1, L, 1, head_dim)
        cos = mx.expand_dims(cos, 0)  # (1, L, head_dim)
        cos = mx.expand_dims(cos, 2)  # (1, L, 1, head_dim)
        sin = mx.expand_dims(sin, 0)  # (1, L, head_dim)
        sin = mx.expand_dims(sin, 2)  # (1, L, 1, head_dim)

        # Transpose k to [B, L, n_kv_heads, head_dim] for RoPE application
        k_for_rope = k.transpose(1, 2, 0, 3)

        # Apply RoPE to K
        k_rot = (k_for_rope * cos) + (rotate_half(k_for_rope) * sin)

        # Transpose back to [n_kv_heads, B, L, head_dim]
        return k_rot.transpose(2, 0, 1, 3)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """Simple standard attention - cache contains materialized context.

        Args:
            hidden_states: Input embeddings [B, L, D]
            position_embeddings: (cos, sin) for RoPE
            mask: Attention mask (unused, kept for API compatibility)
            cache: KV cache (contains materialized target context + previous tokens)

        Returns:
            Output embeddings [B, L, D]
        """
        B, L, D = hidden_states.shape

        # Standard QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        # Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        cos, sin = position_embeddings
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        # Update cache (CroppableKVCache or standard)
        if cache is not None:
            k_embed, v = cache.update_and_fetch(k_embed, v)

        # Scaled dot-product attention
        # mask=None provides non-causal attention (attend to all positions including cache)
        output = scaled_dot_product_attention(
            q_embed, k_embed, v, cache=None, scale=self.scaling, mask=None
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DFlashAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Self-attention (cache contains materialized target context)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Handle cache - pass layer_idx for DFlashCacheManager
        cache_to_use = None
        if cache is not None:
            if isinstance(cache, DFlashCacheManager):
                cache_to_use = cache.get_layer_cache(self.layer_idx)
            else:
                cache_to_use = cache

        hidden_states = self.self_attn(hidden_states, position_embeddings, mask, cache_to_use)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # DFlash uses fixed head_dim of 128, not hidden_size // num_heads
        self.head_dim = 128

    def __call__(self, hidden_states: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute cos/sin for RoPE.

        Args:
            hidden_states: [B, L, D]
            position_ids: [B, L] - absolute positions

        Returns:
            (cos, sin) for RoPE application - shape (seq_len, rotary_dim)
        """
        seq_len = position_ids.shape[-1]
        position_ids = position_ids.astype(mx.float32)

        # Base frequency computation - rotary_dim is head_dim
        rotary_dim = self.head_dim
        inv_freq = 1.0 / (self.args.rope_theta ** (mx.arange(0, rotary_dim, 2) / rotary_dim))

        # Compute position indices - shape (seq_len,)
        position_ids = position_ids.reshape(-1)

        # Compute rotary embeddings
        t = position_ids[:, None]  # (seq_len, 1)
        freqs = t * inv_freq[None, :]  # (seq_len, rotary_dim/2)

        emb = mx.concatenate([mx.cos(freqs), mx.sin(freqs)], axis=-1)  # (seq_len, rotary_dim)

        cos = emb[:, : emb.shape[-1] // 2]  # (seq_len, rotary_dim/2)
        sin = emb[:, emb.shape[-1] // 2 :]  # (seq_len, rotary_dim/2)

        return cos, sin


class Model(nn.Module):
    """DFlash draft model for MLX-LM."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Expose key attributes
        self.block_size = args.block_size
        self.mask_token_id = args.mask_token_id
        # Build target_layer_ids if not specified in config
        if args.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(
                args.num_target_layers, args.num_hidden_layers
            )
        else:
            self.target_layer_ids = args.target_layer_ids
        self.model_type = args.model_type

        # Create decoder layers
        self.layers = [
            DFlashDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]

        # Feature compression
        num_layers = len(self.target_layer_ids)
        self.fc = nn.Linear(num_layers * args.hidden_size, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Final normalization
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # RoPE
        self.rotary_emb = RoPE(args)

    def make_cache(self):
        """Create a DFlash cache manager with cropping support."""
        return DFlashCacheManager(self.args.num_hidden_layers, self.args.block_size)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Clean up weights before loading.

        The HuggingFace DFlash models have incorrect config - using actual weight shapes.
        """
        # No reshaping needed - from_dict now detects actual dimensions from weights
        return weights

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        """Project concatenated target-layer hidden states into draft hidden_size.

        Args:
            target_hidden: [B, ctx_len, num_layers * D] - raw target features

        Returns:
            Compressed hidden states [B, ctx_len, D]
        """
        B, ctx_len, num_layers_times_D = target_hidden.shape
        target_hidden_flat = target_hidden.reshape(B * ctx_len, num_layers_times_D)
        compressed_target_flat = self.hidden_norm(self.fc(target_hidden_flat))
        return compressed_target_flat.reshape(B, ctx_len, -1)

    def materialize_target_hidden(
        self,
        target_hidden: mx.array,
        cache: Any,  # DFlashCacheManager or list
        position_ids: Optional[mx.array] = None,
    ) -> None:
        """Materialize target hidden states into draft KV cache.

        This projects the target model's hidden states and adds them as K/V
        to the draft model's cache, giving the draft model proper context.

        Args:
            target_hidden: [B, ctx_len, num_layers * D] - raw target features
            cache: DFlashCacheManager or list of caches to update
            position_ids: [B, ctx_len] - position IDs for target context (optional)
        """
        B, ctx_len, _ = target_hidden.shape

        # Project target hidden states to draft dimension
        compressed_hidden = self.project_target_hidden(target_hidden)

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = mx.arange(ctx_len)[None, :]

        # Compute position embeddings for target context
        position_embeddings = self.rotary_emb(compressed_hidden, position_ids)

        # Materialize into each layer's cache
        for layer_idx, layer in enumerate(self.layers):
            # Get the appropriate cache for this layer
            if isinstance(cache, DFlashCacheManager):
                layer_cache = cache.get_layer_cache(layer_idx)
            else:
                layer_cache = cache[layer_idx]

            # Get K/V projections only (skip Q since we're just materializing context)
            k, v = layer.self_attn.kv_proj_only(compressed_hidden)

            # Apply K normalization
            k = layer.self_attn.apply_k_norm(k)

            # Apply RoPE to K
            k = layer.self_attn.apply_k_rope(position_ids.reshape(-1), k, self.rotary_emb)

            # Update cache with target context K/V
            layer_cache.update_and_fetch(k, v)

    def __call__(
        self,
        position_ids: mx.array,
        noise_embedding: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Args:
            position_ids: [B, L] - absolute positions for noise tokens
            noise_embedding: [B, L, D] - embeddings from mask tokens
            cache: DFlashCacheManager or list of caches (should already contain materialized target context)

        Returns:
            Hidden states [B, L, D]
        """
        # Compute position embeddings
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)

        # Process through decoder layers
        hidden_states = noise_embedding
        if cache is None:
            caches = [None] * len(self.layers)
        elif isinstance(cache, DFlashCacheManager):
            # DFlashCacheManager - each layer will fetch its own cache
            caches = [cache] * len(self.layers)
        else:
            # List of caches
            caches = cache

        for layer, c in zip(self.layers, caches):
            hidden_states = layer(hidden_states, position_embeddings, cache=c)

        return self.norm(hidden_states)
