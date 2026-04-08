# Copyright © 2025 Apple Inc.

"""DFlash draft model cache implementations.

Contains:
- DFlashDraftLayerCache: Per-layer cache for cross-attention that separates
  context K/V (replaced each iteration) from noise K/V (accumulated).
- CroppableKVCache: Simple KV cache with cropping support (legacy).
- DFlashCacheManager: Manages CroppableKVCache instances (legacy).
"""

from typing import Optional, Tuple, List, Any
import mlx.core as mx


class DFlashDraftLayerCache:
    """Per-layer cache for DFlash draft model cross-attention.

    DFlash attention concatenates K/V from target_hidden (context) and
    noise_embedding (noise). Standard KVCache doesn't work well because
    context changes each iteration.

    This cache stores verified NOISE K/V across iterations (the draft model's
    own historical processing), while context K/V are replaced each iteration.

    The attention sees: cached_noise + new_context + new_noise
    """

    def __init__(self):
        self.cached_k = None  # [B, H, cached_len, D]
        self.cached_v = None  # [B, H, cached_len, D]
        self.cached_len = 0

    def combine(self, new_k, new_v):
        """Combine cached noise K/V with new K/V for attention.

        New K/V already have RoPE applied. Cached K/V already have RoPE
        from their original positions.

        Args:
            new_k: [B, H, ctx_len + noise_len, D] - new K with RoPE
            new_v: [B, H, ctx_len + noise_len, D] - new V

        Returns:
            Combined (K, V) for attention computation.
        """
        if self.cached_k is not None:
            k = mx.concatenate([self.cached_k, new_k], axis=2)
            v = mx.concatenate([self.cached_v, new_v], axis=2)
        else:
            k = new_k
            v = new_v
        return k, v

    def commit(self, noise_k, noise_v, num_to_cache):
        """Save accepted noise K/V to the cache after verification.

        Args:
            noise_k: [B, H, noise_len, D] - noise portion of K (with RoPE)
            noise_v: [B, H, noise_len, D] - noise portion of V
            num_to_cache: number of accepted entries to cache
        """
        if num_to_cache <= 0:
            return
        accepted_k = noise_k[:, :, :num_to_cache, :]
        accepted_v = noise_v[:, :, :num_to_cache, :]
        if self.cached_k is not None:
            self.cached_k = mx.concatenate([self.cached_k, accepted_k], axis=2)
            self.cached_v = mx.concatenate([self.cached_v, accepted_v], axis=2)
        else:
            self.cached_k = accepted_k
            self.cached_v = accepted_v
        self.cached_len += num_to_cache


def make_dflash_draft_cache(num_layers):
    """Create a list of DFlashDraftLayerCache, one per draft model layer."""
    return [DFlashDraftLayerCache() for _ in range(num_layers)]


class CroppableKVCache:
    """KV cache that supports cropping/removing tokens."""

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    def update(self, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache with new keys and values.

        Args:
            k: Keys [n_kv_heads, B, L, head_dim]
            v: Values [n_kv_heads, B, L, head_dim]

        Returns:
            Tuple of (k, v) with cache appended
        """
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)

        return self.keys, self.values

    def fetch(self) -> Tuple[mx.array, mx.array]:
        """Fetch all keys and values from cache."""
        if self.keys is None:
            return None, None
        return self.keys, self.values

    def crop(self, keep_length: int) -> None:
        """Crop cache to keep only first keep_length tokens.

        Args:
            keep_length: Number of tokens to keep from the beginning
        """
        if self.keys is not None:
            self.keys = self.keys[:, :, :keep_length, :]
            self.values = self.values[:, :, :keep_length, :]

    def size(self) -> int:
        """Get current cache size (number of tokens)."""
        if self.keys is None:
            return 0
        return self.keys.shape[2]

    def trim(self, num_tokens: int) -> int:
        """Trim the last num_tokens from the cache.

        Args:
            num_tokens: Number of tokens to trim from the end

        Returns:
            Actual number of tokens trimmed
        """
        if self.keys is None:
            return 0
        current_size = self.keys.shape[2]
        actual_trim = min(num_tokens, current_size)
        keep_size = current_size - actual_trim
        if keep_size > 0:
            self.keys = self.keys[:, :, :keep_size, :]
            self.values = self.values[:, :, :keep_size, :]
        else:
            # Keep at least one token
            self.keys = self.keys[:, :, :1, :]
            self.values = self.values[:, :, :1, :]
        return actual_trim

    def update_and_fetch(self, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache and fetch current state (API compatible with KVCache)."""
        return self.update(k, v)

    def to_axes(self, kv_pos: int, axis: int) -> Tuple[int, ...]:
        """API compatibility for prompt cache construction."""
        return (kv_pos, axis)


class DFlashCacheManager:
    """Manages caches for DFlash with cropping support.

    This class is designed to be compatible with the expected list-of-caches
    interface used by MLX-LM's generation code.
    """

    def __init__(self, num_layers: int, block_size: int):
        self.num_layers = num_layers
        self.block_size = block_size
        self.caches = [CroppableKVCache() for _ in range(num_layers)]
        self.noise_start = 0  # Position where noise tokens start

    def __iter__(self):
        """Make the manager iterable like a list of caches."""
        return iter(self.caches)

    def __len__(self):
        """Return the number of layers (for list compatibility)."""
        return self.num_layers

    def __getitem__(self, index):
        """Allow indexing like a list."""
        return self.caches[index]

    def update_layer(self, layer_idx: int, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        """Update a specific layer's cache."""
        return self.caches[layer_idx].update(k, v)

    def crop_noise_tokens(self) -> None:
        """Crop all noise tokens from previous iteration, keep only context."""
        for cache in self.caches:
            cache.crop(self.noise_start)

    def update_noise_start(self, new_start: int) -> None:
        """Update the position where noise tokens start."""
        self.noise_start = new_start

    def get_layer_cache(self, layer_idx: int) -> CroppableKVCache:
        """Get cache for a specific layer."""
        return self.caches[layer_idx]

    def total_size(self) -> int:
        """Get total cache size (same for all layers)."""
        return self.caches[0].size() if self.caches else 0

    def trim(self, num_tokens: int) -> int:
        """Trim the last num_tokens from all layer caches.

        Args:
            num_tokens: Number of tokens to trim from the end

        Returns:
            Actual number of tokens trimmed
        """
        if not self.caches:
            return 0
        return self.caches[0].trim(num_tokens)  # All layers have same size
