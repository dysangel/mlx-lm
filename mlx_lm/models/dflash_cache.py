# Copyright © 2025 Apple Inc.

"""Custom KV cache with cropping support for DFlash."""

from typing import Optional, Tuple
import mlx.core as mx


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

    def update_and_fetch(self, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache and fetch current state (API compatible with KVCache)."""
        return self.update(k, v)


class DFlashCacheManager:
    """Manages caches for DFlash with cropping support."""

    def __init__(self, num_layers: int, block_size: int):
        self.num_layers = num_layers
        self.block_size = block_size
        self.caches = [CroppableKVCache() for _ in range(num_layers)]
        self.noise_start = 0  # Position where noise tokens start

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
