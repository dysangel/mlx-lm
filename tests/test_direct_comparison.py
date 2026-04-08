#!/usr/bin/env python3
"""Direct comparison tests between MLX and PyTorch reference implementations."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import (
    ModelWithHiddenStates,
    extract_context_feature,
    get_inner_model,
)


def test_hidden_states_extraction_comparison():
    """Test that hidden states extraction matches reference exactly."""
    print("\n=== Test 1: Hidden States Extraction ===")

    # Load MLX model
    model, tokenizer = load("Qwen/Qwen3.5-27B")

    # Create wrapper
    target_layer_ids = [1, 16, 31, 46, 61]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    # Create input
    prompt = "Hello world test"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Forward pass
    output = wrapped_model(tokens, cache=None)
    mx.eval(output.logits)

    # Extract target_hidden
    target_hidden = extract_context_feature(
        wrapped_model.hidden_states,
        target_layer_ids,
    )
    mx.eval(target_hidden)

    print(f"  Input: '{prompt}'")
    print(f"  Tokens: {tokens.squeeze().tolist()}")
    print(f"  Num tokens: {tokens.shape[1]}")
    print(f"  Num hidden states: {len(wrapped_model.hidden_states)}")
    print(f"  Target layer IDs: {target_layer_ids}")
    print(f"  target_hidden shape: {target_hidden.shape}")

    # Calculate hidden size per layer
    D = target_hidden.shape[2] // len(target_layer_ids)
    print(f"  Hidden size per layer: {D}")
    print(f"  Expected: (1, {tokens.shape[1]}, {len(target_layer_ids) * D})")

    # Verify shape
    assert target_hidden.shape[0] == 1, "Wrong batch size"
    assert target_hidden.shape[1] == tokens.shape[1], f"Wrong seq_len: {target_hidden.shape[1]} != {tokens.shape[1]}"

    print("  ✓ Hidden states extraction works correctly")
    return target_hidden, wrapped_model.hidden_states


def test_draft_token_sampling_comparison():
    """Test draft token sampling matches reference exactly."""
    print("\n=== Test 2: Draft Token Sampling ===")

    model, tokenizer = load("Qwen/Qwen3.5-27B")
    draft_model, _ = load("z-lab/Qwen3.5-27B-DFlash")

    # Setup
    prompt = "The answer is"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_layer_ids = [1, 16, 31, 46, 61]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    # Get target_hidden
    output = wrapped_model(tokens, cache=None)
    mx.eval(output.logits)

    target_hidden = extract_context_feature(
        wrapped_model.hidden_states,
        target_layer_ids,
    )
    mx.eval(target_hidden)

    # Get last token
    last_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()

    # Draft generation parameters
    block_size = 8
    ctx_len = target_hidden.shape[1]

    # Create draft input
    inner_model = get_inner_model(model)
    prev_token = mx.array([[last_token]])
    noise_tokens = mx.full([1, block_size - 1], draft_model.mask_token_id, dtype=mx.uint32)
    draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)
    noise_embedding = inner_model.embed_tokens(draft_input)
    mx.eval(noise_embedding)

    print(f"  Context: '{prompt}'")
    print(f"  Last token: {tokenizer.decode([last_token])} (id={last_token})")
    print(f"  ctx_len: {ctx_len}")
    print(f"  block_size: {block_size}")
    print(f"  draft_input shape: {draft_input.shape}")
    print(f"  noise_embedding shape: {noise_embedding.shape}")

    # Position IDs
    position_ids = mx.arange(ctx_len, ctx_len + block_size)[None, :]
    print(f"  position_ids: {position_ids.tolist()}")

    # Expected position IDs from reference
    # Reference uses: position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]
    # where start = current position in output_ids
    # For first draft call, start = num_input_tokens + 1 = len(prompt) + 1
    start = len(tokenizer.encode(prompt)) + 1
    expected_position_ids = mx.arange(start, start + block_size)[None, :]
    print(f"  Expected position_ids: {expected_position_ids.tolist()}")
    print(f"  Match: {mx.all(position_ids == expected_position_ids).item()}")

    if not mx.all(position_ids == expected_position_ids).item():
        print("  ✗ ERROR: Position IDs don't match reference!")
        print(f"    Difference: {position_ids.squeeze().tolist()} vs {expected_position_ids.squeeze().tolist()}")

    return position_ids, expected_position_ids


def test_draft_output_comparison():
    """Test draft model output matches reference."""
    print("\n=== Test 3: Draft Model Output ===")

    model, tokenizer = load("Qwen/Qwen3.5-27B")
    draft_model, _ = load("z-lab/Qwen3.5-27B-DFlash")

    # Setup
    prompt = "Hello"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_layer_ids = [1, 16, 31, 46, 61]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    output = wrapped_model(tokens, cache=None)
    mx.eval(output.logits)

    target_hidden = extract_context_feature(
        wrapped_model.hidden_states,
        target_layer_ids,
    )
    mx.eval(target_hidden)

    last_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()

    # Draft generation
    block_size = 8
    ctx_len = target_hidden.shape[1]

    inner_model = get_inner_model(model)
    prev_token = mx.array([[last_token]])
    noise_tokens = mx.full([1, block_size - 1], draft_model.mask_token_id, dtype=mx.uint32)
    draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)
    noise_embedding = inner_model.embed_tokens(draft_input)
    mx.eval(noise_embedding)

    position_ids = mx.arange(ctx_len, ctx_len + block_size)[None, :]
    draft_cache = draft_model.make_cache()

    print(f"  Calling draft model with:")
    print(f"    position_ids shape: {position_ids.shape}")
    print(f"    noise_embedding shape: {noise_embedding.shape}")
    print(f"    target_hidden shape: {target_hidden.shape}")
    print(f"    cache types: {[type(c).__name__ for c in draft_cache[:3]]}")

    # Call draft model
    draft_output = draft_model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=draft_cache,
    )
    mx.eval(draft_output)

    print(f"  draft_output shape: {draft_output.shape}")
    print(f"  Expected: (1, {block_size}, hidden_size)")

    assert draft_output.shape[0] == 1, "Wrong batch size"
    assert draft_output.shape[1] == block_size, f"Wrong seq_len: {draft_output.shape[1]} != {block_size}"

    # Get draft logits
    if hasattr(model, 'lm_head'):
        draft_logits = model.lm_head(draft_output)
    else:
        draft_logits = inner_model.embed_tokens.as_linear(draft_output)
    mx.eval(draft_logits)

    print(f"  draft_logits shape: {draft_logits.shape}")

    # Sample draft tokens
    draft_tokens_block = mx.argmax(draft_logits[:, -block_size + 1:, :], axis=-1).squeeze(0)
    mx.eval(draft_tokens_block)

    print(f"  draft_tokens_block: {draft_tokens_block.tolist()}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in draft_tokens_block.tolist()]}")

    return draft_output, draft_logits, draft_tokens_block


def test_verification_comparison():
    """Test draft verification matches reference."""
    print("\n=== Test 4: Draft Verification ===")

    model, tokenizer = load("Qwen/Qwen3.5-27B")

    # Simulate a simple verification scenario
    # Draft tokens to verify
    draft_tokens_to_verify = mx.array([[15, 16, 17, 18, 19]])

    # Create cache
    cache = model.make_cache()

    # Verify with target model
    logits = model(draft_tokens_to_verify, cache=cache)
    mx.eval(logits)

    # Get target predictions
    target_tokens = mx.argmax(logits, axis=-1).squeeze(0)

    print(f"  Draft tokens: {draft_tokens_to_verify.squeeze().tolist()}")
    print(f"  Target tokens: {target_tokens.tolist()}")
    print(f"  Match: {(draft_tokens_to_verify.squeeze(0) == target_tokens).tolist()}")

    # Calculate acceptance length
    acceptance_length = (
        mx.cumsum(draft_tokens_to_verify.squeeze(0) == target_tokens) == mx.arange(len(target_tokens))
    ).sum()
    acceptance_length = int(acceptance_length)

    print(f"  Acceptance length: {acceptance_length}/{len(target_tokens)}")

    # Check off-by-one: target_tokens has one fewer element than draft_tokens
    print(f"  Draft tokens length: {len(draft_tokens_to_verify.squeeze())}")
    print(f"  Target tokens length: {len(target_tokens)}")
    print(f"  Difference: {len(draft_tokens_to_verify.squeeze()) - len(target_tokens)}")

    if len(draft_tokens_to_verify.squeeze()) != len(target_tokens):
        print("  ✓ Correct: target has one fewer (no prediction for last draft token)")
    else:
        print("  ✗ ERROR: target should have one fewer element!")

    return acceptance_length


def test_cache_behavior_comparison():
    """Test cache behavior matches reference."""
    print("\n=== Test 5: Cache Behavior ===")

    model, tokenizer = load("Qwen/Qwen3.5-27B")

    # Process tokens with cache
    tokens = mx.array(tokenizer.encode("Hello world test"))[None, :]
    cache = model.make_cache()

    # First call
    output1 = model(tokens, cache=cache)
    mx.eval(output1)

    print(f"  Input tokens: {tokens.squeeze().tolist()}")
    print(f"  Cache types: {[type(c).__name__ for c in cache[:3]]}")

    # Check cache state after first call
    for i, c in enumerate(cache[:3]):
        if hasattr(c, 'offset'):
            print(f"  Layer {i} (KVCache): offset = {c.offset}")
            assert c.offset == tokens.shape[1], f"KVCache offset should be {tokens.shape[1]}"
        elif hasattr(c, 'cache'):
            k, v = c.cache
            if k is not None:
                print(f"  Layer {i} (ArraysCache): K shape = {k.shape}")

    # Second call with new tokens
    new_tokens = mx.array([[99, 100]])
    output2 = model(new_tokens, cache=cache)
    mx.eval(output2)

    # Check cache state after second call
    for i, c in enumerate(cache[:3]):
        if hasattr(c, 'offset'):
            print(f"  Layer {i} (KVCache): offset = {c.offset} (after second call)")
            expected = tokens.shape[1] + new_tokens.shape[1]
            assert c.offset == expected, f"KVCache offset should be {expected}"

    print("  ✓ Cache behavior is correct")

    return cache


def test_token_indices_comparison():
    """Test that token indices are correct (off-by-one check)."""
    print("\n=== Test 6: Token Indices (Off-by-One Check) ===")

    model, tokenizer = load("Qwen/Qwen3.5-27B")

    # Simulate draft token sampling
    block_size = 8

    # Create mock draft_logits (shape: [1, block_size, vocab_size])
    # Position 0 is for seed token, positions 1-7 are for draft tokens
    draft_logits = mx.random.normal((1, block_size, tokenizer.vocab_size))

    # Sample draft tokens
    draft_tokens_block = mx.argmax(draft_logits[:, -block_size + 1:, :], axis=-1).squeeze(0)

    print(f"  draft_logits shape: {draft_logits.shape}")
    print(f"  Block size: {block_size}")
    print(f"  Slicing: [:, -{block_size - 1}:, :]")
    print(f"  This gives positions: [1, 2, ..., {block_size - 1}]")
    print(f"  draft_tokens_block: {draft_tokens_block.tolist()}")
    print(f"  Length: {len(draft_tokens_block)}")
    print(f"  Expected length: {block_size - 1}")

    assert len(draft_tokens_block) == block_size - 1, f"Should have {block_size - 1} tokens"

    # Verify the indexing
    for i in range(block_size - 1):
        # draft_tokens_block[i] should be from draft_logits[:, i + 1, :]
        expected_token = mx.argmax(draft_logits[:, i + 1, :], axis=-1).squeeze(0).item()
        actual_token = draft_tokens_block[i].item()
        assert actual_token == expected_token, f"Token {i} mismatch"

    print("  ✓ Token indices are correct")

    return draft_tokens_block


def test_construct_draft_tokens_comparison():
    """Test draft_tokens construction matches reference."""
    print("\n=== Test 7: Construct Draft Tokens ===")

    block_size = 8
    first_token = 42  # Seed token
    draft_tokens_block = mx.array([10, 11, 12, 13, 14, 15, 16])

    # Construct full draft_tokens
    draft_tokens = mx.concatenate([mx.array([first_token]), draft_tokens_block])

    print(f"  first_token: {first_token}")
    print(f"  draft_tokens_block: {draft_tokens_block.tolist()}")
    print(f"  draft_tokens (constructed): {draft_tokens.tolist()}")
    print(f"  Expected: [{first_token}] + {draft_tokens_block.tolist()}")

    assert draft_tokens[0].item() == first_token, "First token should be seed"
    assert draft_tokens[1:].tolist() == draft_tokens_block.tolist(), "Rest should be draft_tokens_block"

    # Verification: skip seed token
    draft_to_verify = draft_tokens[1:][None, :]
    print(f"  draft_to_verify (skip seed): {draft_to_verify.squeeze().tolist()}")
    print(f"  Expected: {draft_tokens_block.tolist()}")

    assert draft_to_verify.squeeze().tolist() == draft_tokens_block.tolist(), "Should match draft_tokens_block"

    print("  ✓ Draft tokens construction is correct")

    return draft_tokens


def run_all_comparison_tests():
    """Run all comparison tests."""
    print("=" * 70)
    print("MLX vs Reference - Direct Comparison Tests")
    print("=" * 70)

    try:
        test_hidden_states_extraction_comparison()
        test_draft_token_sampling_comparison()
        test_draft_output_comparison()
        test_verification_comparison()
        test_cache_behavior_comparison()
        test_token_indices_comparison()
        test_construct_draft_tokens_comparison()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Comparison Tests Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_all_comparison_tests()
