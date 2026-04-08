#!/usr/bin/env python3
"""Test if ArraysCache updates inside the layer."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load


def test_layer_cache_update():
    """Test ArraysCache update at layer level."""
    print("\n=== Layer-level ArraysCache Update Test ===")

    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Get a single linear layer
    inner_model = model
    if hasattr(inner_model, 'language_model'):
        inner_model = inner_model.language_model
    if hasattr(inner_model, 'model'):
        inner_model = inner_model.model

    layer = inner_model.layers[0]  # First layer is linear
    print(f"Layer type: {type(layer).__name__}")
    print(f"Is linear: {layer.is_linear}")

    # Create input (hidden_size = 2560 for Qwen3.5-4B)
    x = mx.random.uniform(shape=(1, 3, 2560))

    # Create cache
    cache = model.make_cache()[0]
    print(f"\nInitial cache: type={type(cache).__name__}")
    print(f"Initial cache[0]: {cache.cache[0] if cache.cache[0] is not None else None}")

    # First forward pass (3 tokens)
    h1 = layer(x, mask=None, cache=cache)
    mx.eval(h1)
    print(f"\nAfter first pass (3 tokens):")
    print(f"cache[0] shape: {cache.cache[0].shape if cache.cache[0] is not None else None}")
    print(f"cache[1] shape: {cache.cache[1].shape if cache.cache[1] is not None else None}")

    # Second forward pass (1 token)
    x2 = mx.random.uniform(shape=(1, 1, 2560))
    h2 = layer(x2, mask=None, cache=cache)
    mx.eval(h2)
    print(f"\nAfter second pass (1 new token):")
    print(f"cache[0] shape: {cache.cache[0].shape if cache.cache[0] is not None else None}")
    print(f"cache[1] shape: {cache.cache[1].shape if cache.cache[1] is not None else None}")

    # Check if size increased
    expected_size = 4  # n_keep
    actual_size = cache.cache[0].shape[1]
    print(f"\nExpected size: {expected_size}, Actual: {actual_size}")
    print(f"Cache update {'SUCCESS' if actual_size == expected_size else 'FAILED'}")


if __name__ == "__main__":
    test_layer_cache_update()
