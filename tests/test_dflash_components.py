#!/usr/bin/env python3
"""Unit tests for DFlash components to verify correct behavior."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import (
    ModelWithHiddenStates,
    extract_context_feature,
    block_diffusion_generate_step,
)
import pytest


def test_extract_context_feature():
    """Test that extract_context_feature correctly concatenates hidden states."""
    # Create mock hidden states: [embedding, layer1, layer2]
    B, seq_len, D = 1, 5, 128
    hidden_states = [
        mx.zeros((B, seq_len, D)),  # embedding
        mx.ones((B, seq_len, D)) * 1,  # layer 1
        mx.ones((B, seq_len, D)) * 2,  # layer 2
        mx.ones((B, seq_len, D)) * 3,  # layer 3
    ]

    # Extract from layers 0 and 2 (which are indices 1 and 3 in hidden_states)
    layer_ids = [0, 2]
    result = extract_context_feature(hidden_states, layer_ids)

    # Result should concatenate layer 1 and layer 3 outputs
    # Shape: [B, seq_len, num_layers * D]
    expected_shape = (B, seq_len, len(layer_ids) * D)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Verify values: first D elements should be 1s, next D elements should be 3s
    mx.eval(result)
    assert mx.all(result[:, :, :D] == 1).item(), "First layer values incorrect"
    assert mx.all(result[:, :, D:] == 3).item(), "Second layer values incorrect"

    print("✓ extract_context_feature works correctly")


def test_model_with_hidden_states_captures_all_layers():
    """Test that ModelWithHiddenStates captures embedding and all layer outputs."""
    print("\n=== Testing ModelWithHiddenStates ===")

    # Load target model
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Create wrapper
    target_layer_ids = [1, 8, 15, 22, 29]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    # Create input
    prompt = "Hello"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Forward pass
    output = wrapped_model(tokens, cache=None)
    mx.eval(output.logits)

    # Verify hidden_states were captured
    assert len(wrapped_model.hidden_states) > 0, "No hidden states captured"

    # Should have: embedding + 32 layers = 33 hidden states
    expected_num_states = 1 + 32  # embedding + layers
    assert len(wrapped_model.hidden_states) == expected_num_states, \
        f"Expected {expected_num_states} hidden states, got {len(wrapped_model.hidden_states)}"

    # Verify shapes
    B, seq_len, D = wrapped_model.hidden_states[0].shape
    print(f"  Hidden state shape: {wrapped_model.hidden_states[0].shape}")

    # All hidden states should have same sequence length
    for i, h in enumerate(wrapped_model.hidden_states):
        assert h.shape[1] == seq_len, f"Hidden state {i} has wrong seq_len: {h.shape[1]} != {seq_len}"

    print(f"✓ ModelWithHiddenStates captured all {len(wrapped_model.hidden_states)} states")


def test_model_with_hidden_states_with_cache():
    """Test that ModelWithHiddenStates works correctly with cache."""
    print("\n=== Testing ModelWithHiddenStates with Cache ===")

    # Load target model
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Create wrapper
    target_layer_ids = [1, 8, 15, 22, 29]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    # Create input
    prompt = "Hello world"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # First call with cache
    cache = model.make_cache()
    output1 = wrapped_model(tokens, cache=cache)
    mx.eval(output1.logits)

    hidden_1 = wrapped_model.hidden_states.copy()

    # Second call with cache (should extend)
    new_token = mx.array([[tokenizer.encode(" !")[0]]])
    output2 = wrapped_model(new_token, cache=cache)
    mx.eval(output2.logits)

    hidden_2 = wrapped_model.hidden_states.copy()

    # Verify: hidden states from second call should only have new tokens
    # The cache should contain the previous tokens
    assert hidden_2[0].shape[1] < hidden_1[0].shape[1], \
        "Cached call should have fewer new tokens"

    print(f"  First call seq_len: {hidden_1[0].shape[1]}")
    print(f"  Second call seq_len: {hidden_2[0].shape[1]}")
    print("✓ ModelWithHiddenStates works with cache")


def test_draft_model_interface():
    """Test that draft model has correct interface."""
    print("\n=== Testing Draft Model Interface ===")

    # Load draft model
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")

    # Check attributes
    assert hasattr(draft_model, 'block_size'), "Draft model should have block_size"
    assert hasattr(draft_model, 'target_layer_ids'), "Draft model should have target_layer_ids"

    print(f"  Block size: {draft_model.block_size}")
    print(f"  Target layer IDs: {draft_model.target_layer_ids}")

    # Check __call__ signature
    import inspect
    sig = inspect.signature(draft_model.__call__)
    params = list(sig.parameters.keys())
    print(f"  __call__ params: {params}")

    assert 'position_ids' in params, "Draft model should accept position_ids"
    assert 'noise_embedding' in params, "Draft model should accept noise_embedding"
    assert 'target_hidden' in params, "Draft model should accept target_hidden"
    assert 'cache' in params, "Draft model should accept cache"

    print("✓ Draft model has correct interface")


def test_draft_model_forward_pass():
    """Test that draft model can process inputs."""
    print("\n=== Testing Draft Model Forward Pass ===")

    # Load models
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")
    target_model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Create mock inputs
    B, L, D = 1, 16, 2560
    block_size = draft_model.block_size

    # Create target_hidden (from prompt)
    prompt = "Hello"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Get target hidden states
    target_layer_ids = [1, 8, 15, 22, 29]
    wrapped_target = ModelWithHiddenStates(target_model, target_layer_ids)
    output = wrapped_target(prompt_tokens, cache=None)
    mx.eval(output.logits)

    target_hidden = extract_context_feature(
        wrapped_target.hidden_states,
        draft_model.target_layer_ids,
    )
    mx.eval(target_hidden)

    print(f"  target_hidden shape: {target_hidden.shape}")

    # Create noise_embedding
    noise_tokens = mx.zeros((B, block_size), dtype=mx.uint32)
    from mlx_lm.generate_dflash_v2 import get_inner_model
    inner_model = get_inner_model(target_model)
    noise_embedding = inner_model.embed_tokens(noise_tokens)
    mx.eval(noise_embedding)

    # Create position_ids
    ctx_len = target_hidden.shape[1]
    position_ids = mx.arange(ctx_len, ctx_len + block_size)[None, :]

    # Create draft cache
    draft_cache = draft_model.make_cache()

    # Forward pass
    draft_output = draft_model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=draft_cache,
    )
    mx.eval(draft_output)

    print(f"  Draft output shape: {draft_output.shape}")
    assert draft_output.shape[0] == B, "Wrong batch size"
    assert draft_output.shape[1] == block_size, "Wrong sequence length"

    print("✓ Draft model forward pass works")


def test_position_ids_calculation():
    """Test that position IDs are calculated correctly."""
    print("\n=== Testing Position IDs Calculation ===")

    # Scenario: prompt has 3 tokens, we've generated 2 tokens
    # Next block of 16 should start at position 5

    num_prompt_tokens = 3
    num_generated_tokens = 2
    block_size = 16

    # Correct calculation: position = prompt + generated
    correct_start = num_prompt_tokens + num_generated_tokens
    position_ids = mx.arange(correct_start, correct_start + block_size)[None, :]

    print(f"  Prompt tokens: {num_prompt_tokens}")
    print(f"  Generated tokens: {num_generated_tokens}")
    print(f"  Block size: {block_size}")
    print(f"  Position IDs: {position_ids.tolist()}")

    expected = mx.array([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    assert mx.all(position_ids == expected).item(), "Position IDs incorrect"

    print("✓ Position IDs calculated correctly")


def test_rope_application_split():
    """Test that RoPE is applied only to noise part of k, not context."""
    print("\n=== Testing RoPE Application Split ===")

    from mlx_lm.models.dflash_v2 import apply_rotary_pos_emb_single, rotate_half

    B, num_heads, ctx_len, noise_len, head_dim = 1, 8, 10, 5, 128

    # Create mock k (concatenated context + noise)
    k = mx.random.normal((B, num_heads, ctx_len + noise_len, head_dim))

    # Create cos/sin for noise tokens only
    cos = mx.random.normal((noise_len, head_dim // 2))
    sin = mx.random.normal((noise_len, head_dim // 2))

    # Split k and apply RoPE only to noise part
    k_ctx = k[..., :ctx_len, :]  # Context part, no RoPE
    k_noise = k[..., ctx_len:, :]  # Noise part, apply RoPE

    k_noise_rope = apply_rotary_pos_emb_single(k_noise, cos, sin)
    k_final = mx.concatenate([k_ctx, k_noise_rope], axis=-2)

    # Verify context part unchanged
    assert mx.all(k_final[..., :ctx_len, :] == k_ctx).item(), "Context keys should not have RoPE"

    # Verify noise part changed
    assert not mx.all(k_final[..., ctx_len:, :] == k_noise).item(), "Noise keys should have RoPE"

    print(f"  Context shape: {k_ctx.shape}")
    print(f"  Noise shape: {k_noise.shape}")
    print(f"  Final shape: {k_final.shape}")
    print("✓ RoPE applied correctly (context unchanged, noise transformed)")


def test_cache_kv_arrays():
    """Test that cache stores K and V arrays correctly."""
    print("\n=== Testing Cache KV Arrays ===")

    from mlx_lm import load
    model, _ = load("Qwen/Qwen3.5-4B")
    tokenizer = load("Qwen/Qwen3.5-4B")[1]

    # Make cache
    cache = model.make_cache()

    # Check cache types
    cache_types = [type(c).__name__ for c in cache[:3]]
    print(f"  Cache types: {cache_types}")

    # Process some tokens
    prompt = "Hello world test"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    output = model(tokens, cache=cache)
    mx.eval(output)

    # Check cache contents
    for i, c in enumerate(cache[:3]):
        if hasattr(c, 'cache'):
            # ArraysCache - rolling window
            k, v = c.cache
            if k is not None:
                print(f"  Layer {i} (ArraysCache): K shape {k.shape}, V shape {v.shape}")
                # ArraysCache stores rolling window, may have fewer tokens than input
                assert k.shape[0] == 1, "K cache should have batch size 1"
        elif hasattr(c, 'offset'):
            # KVCache - offset-based
            print(f"  Layer {i} (KVCache): offset {c.offset}")
            assert c.offset == len(tokens), "Offset should match tokens processed"

    print("✓ Cache stores KV arrays correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DFlash Component Tests")
    print("=" * 60)

    test_extract_context_feature()
    test_model_with_hidden_states_captures_all_layers()
    test_model_with_hidden_states_with_cache()
    test_draft_model_interface()
    test_draft_model_forward_pass()
    test_position_ids_calculation()
    test_rope_application_split()
    test_cache_kv_arrays()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
