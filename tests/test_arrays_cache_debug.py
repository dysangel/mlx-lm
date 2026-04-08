#!/usr/bin/env python3
"""Debug ArraysCache state during DFlash generation - with save/restore."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, extract_context_feature


def test_arrays_cache_with_restore():
    """Inspect ArraysCache state during save/restore cycle."""
    print("\n=== ArraysCache Save/Restore Debug ===")

    # Load models
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Setup
    prompt = "What is 2+2?"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_layer_ids = [1, 8, 15, 22, 29]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    # Prefill
    cache = model.make_cache()
    output = wrapped_model(tokens, cache=cache)
    mx.eval(output.logits)

    print(f"\n--- After prefill (3 tokens) ---")
    for i, c in enumerate(cache[:3]):
        print(f"Cache {i}: offset={c.offset}, shape={c.cache[0].shape if c.cache[0] is not None else None}")

    # Save cache
    saved_cache = []
    for c in cache:
        if hasattr(c, 'cache'):
            cache_copy = [mx.array(x) if x is not None else None for x in c.cache]
            lengths_copy = mx.array(c.lengths) if c.lengths is not None else None
            saved_cache.append((cache_copy, lengths_copy))
        else:
            saved_cache.append(None)

    # Verify with 3 draft tokens
    draft_tokens = mx.array([[2531, 2531, 6715]])  # "job job career"
    output2 = wrapped_model(draft_tokens, cache=cache)
    mx.eval(output2.logits)

    print(f"\n--- After verification (3 new tokens) ---")
    for i, c in enumerate(cache[:3]):
        print(f"Cache {i}: offset={c.offset}, shape={c.cache[0].shape if c.cache[0] is not None else None}")

    # Restore cache
    for i, c in enumerate(cache):
        if saved_cache[i] is not None:
            c.cache = saved_cache[i][0]
            c.lengths = saved_cache[i][1]

    print(f"\n--- After restore ---")
    for i, c in enumerate(cache[:3]):
        print(f"Cache {i}: offset={c.offset}, shape={c.cache[0].shape if c.cache[0] is not None else None}")

    # Rebuild with 1 accepted token
    accepted_tokens = mx.array([[2531]])  # "job"
    output3 = wrapped_model(accepted_tokens, cache=cache)
    mx.eval(output3.logits)

    print(f"\n--- After rebuild (1 accepted token) ---")
    for i, c in enumerate(cache[:3]):
        print(f"Cache {i}: offset={c.offset}, shape={c.cache[0].shape if c.cache[0] is not None else None}")

    # Check if cache[0] size increased
    expected_size = 4  # n_keep = conv_kernel_size - 1 = 4
    actual_size = cache[0].cache[0].shape[1]
    print(f"\nExpected ArraysCache size: {expected_size}, Actual: {actual_size}")
    print(f"ArraysCache size {'CORRECT' if actual_size == expected_size else 'WRONG'}")


if __name__ == "__main__":
    test_arrays_cache_with_restore()

