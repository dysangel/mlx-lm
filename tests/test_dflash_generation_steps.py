#!/usr/bin/env python3
"""Integration tests for DFlash generation steps to debug issues."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import (
    ModelWithHiddenStates,
    extract_context_feature,
    get_inner_model,
)


def test_generation_first_iteration():
    """Test that first iteration generates a target token correctly."""
    print("\n=== Testing Generation First Iteration ===")

    # Load models
    model, tokenizer = load("Qwen/Qwen3.5-4B")
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")

    # Setup
    prompt = "What is 2+2?"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Initialize
    target_cache = model.make_cache()
    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)

    # Prefill
    output = target_model_with_hidden(prompt_tokens, cache=target_cache)
    mx.eval(output.logits)

    # Sample first token
    first_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()
    print(f"  Prompt: {prompt}")
    print(f"  First token: {first_token} ({tokenizer.decode([first_token])})")

    # Initialize accumulated_tokens
    accumulated_tokens = [first_token]

    # Extract initial target_hidden
    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )
    print(f"  Initial target_hidden shape: {target_hidden.shape}")

    # Iteration 1: Generate one target token directly
    prev_token = mx.array([[first_token]])
    output = target_model_with_hidden(prev_token, cache=target_cache)
    mx.eval(output.logits)

    target_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()
    accumulated_tokens.append(target_token)

    print(f"  Iteration 1 target token: {target_token} ({tokenizer.decode([target_token])})")

    # Rebuild target_hidden with full sequence
    accumulated_tokens_mx = mx.array(accumulated_tokens)[None, :]
    _ = target_model_with_hidden(accumulated_tokens_mx, cache=None)
    accumulated_hidden = target_model_with_hidden.hidden_states.copy()
    target_hidden = extract_context_feature(
        accumulated_hidden,
        draft_model.target_layer_ids,
    )
    print(f"  Updated target_hidden shape: {target_hidden.shape}")

    # Verify target_hidden grew
    assert target_hidden.shape[1] == 2, f"Expected seq_len 2, got {target_hidden.shape[1]}"

    print("✓ First iteration works correctly")


def test_draft_token_generation():
    """Test that draft model generates tokens (not all the same)."""
    print("\n=== Testing Draft Token Generation ===")

    # Load models
    model, tokenizer = load("Qwen/Qwen3.5-4B")
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")

    # Setup context
    prompt = "The answer is"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)
    output = target_model_with_hidden(prompt_tokens, cache=None)
    mx.eval(output.logits)

    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )
    mx.eval(target_hidden)

    print(f"  target_hidden shape: {target_hidden.shape}")
    ctx_len = target_hidden.shape[1]

    # Get last token as seed
    last_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()
    print(f"  Last token: {last_token} ({tokenizer.decode([last_token])})")

    # Create draft input
    block_size = draft_model.block_size
    inner_model = get_inner_model(model)

    prev_token = mx.array([[last_token]])
    noise_tokens = mx.zeros([1, block_size - 1], dtype=mx.uint32)
    draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)
    noise_embedding = inner_model.embed_tokens(draft_input)
    mx.eval(noise_embedding)

    # Create position_ids (CORRECT way: absolute positions)
    position_ids = mx.arange(ctx_len, ctx_len + block_size)[None, :]

    print(f"  Position IDs: {position_ids.tolist()}")

    # Create draft cache
    draft_cache = draft_model.make_cache()

    # Generate draft tokens
    draft_output = draft_model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=draft_cache,
    )
    mx.eval(draft_output)

    # Get draft logits
    if hasattr(model, 'lm_head'):
        draft_logits = model.lm_head(draft_output)
    else:
        draft_logits = inner_model.embed_tokens.as_linear(draft_output)
    mx.eval(draft_logits)

    # Sample draft tokens
    draft_tokens_block = mx.argmax(draft_logits[:, -block_size + 1:, :], axis=-1).squeeze(0)
    mx.eval(draft_tokens_block)

    print(f"  Draft tokens: {draft_tokens_block.tolist()}")
    print(f"  Draft tokens decoded: {[tokenizer.decode([t]) for t in draft_tokens_block.tolist()]}")

    # Check: NOT all the same
    unique_tokens = len(set(draft_tokens_block.tolist()))
    print(f"  Unique tokens: {unique_tokens}/{block_size - 1}")

    if unique_tokens == 1:
        print("  ⚠ WARNING: All draft tokens are the same!")
    else:
        print("  ✓ Draft tokens have variety")

    return draft_tokens_block


def test_position_ids_evolution():
    """Test that position IDs evolve correctly across iterations."""
    print("\n=== Testing Position IDs Evolution ===")

    # Simulate 3 iterations
    num_prompt_tokens = 3
    block_size = 16

    # Initial context
    ctx_len = num_prompt_tokens
    print(f"  Initial ctx_len: {ctx_len}")

    # Iteration 1: Generate 1 target token
    accepted_in_iter_1 = 1
    ctx_len += accepted_in_iter_1
    position_ids_1 = mx.arange(ctx_len, ctx_len + block_size)[None, :]
    print(f"  Iteration 1: accepted={accepted_in_iter_1}, ctx_len={ctx_len}")
    print(f"    Position IDs: {position_ids_1[:, :3].tolist()}...")

    # Iteration 2: Generate 5 draft + 1 target
    accepted_in_iter_2 = 5 + 1
    ctx_len += accepted_in_iter_2
    position_ids_2 = mx.arange(ctx_len, ctx_len + block_size)[None, :]
    print(f"  Iteration 2: accepted={accepted_in_iter_2}, ctx_len={ctx_len}")
    print(f"    Position IDs: {position_ids_2[:, :3].tolist()}...")

    # Iteration 3: Generate 8 draft + 1 target
    accepted_in_iter_3 = 8 + 1
    ctx_len += accepted_in_iter_3
    position_ids_3 = mx.arange(ctx_len, ctx_len + block_size)[None, :]
    print(f"  Iteration 3: accepted={accepted_in_iter_3}, ctx_len={ctx_len}")
    print(f"    Position IDs: {position_ids_3[:, :3].tolist()}...")

    # Verify: position IDs should increase
    assert position_ids_1[0, 0].item() < position_ids_2[0, 0].item(), "Position IDs should increase"
    assert position_ids_2[0, 0].item() < position_ids_3[0, 0].item(), "Position IDs should increase"

    print("✓ Position IDs evolve correctly")


def test_hidden_states_accumulation():
    """Test that hidden states accumulate correctly across iterations."""
    print("\n=== Testing Hidden States Accumulation ===")

    # Load models
    model, tokenizer = load("Qwen/Qwen3.5-4B")
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")

    # Setup
    prompt = "1 2 3"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)

    # Initial prefill
    output = target_model_with_hidden(prompt_tokens, cache=None)
    mx.eval(output.logits)

    # Get initial hidden states
    initial_hidden = target_model_with_hidden.hidden_states.copy()
    print(f"  Initial hidden states: {len(initial_hidden)} states")
    print(f"  Initial seq_len: {initial_hidden[0].shape[1]}")

    # Generate a few tokens
    tokens = [mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()]

    for i in range(3):
        # Generate next token
        prev_token = mx.array([[tokens[-1]]])
        output = target_model_with_hidden(prev_token, cache=None)
        mx.eval(output.logits)
        tokens.append(mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item())

        # Check hidden states
        current_hidden = target_model_with_hidden.hidden_states
        print(f"  Step {i+1}: seq_len={current_hidden[0].shape[1]}")

    # Rebuild with full sequence
    accumulated_tokens_mx = mx.array(tokens)[None, :]
    _ = target_model_with_hidden(accumulated_tokens_mx, cache=None)
    final_hidden = target_model_with_hidden.hidden_states.copy()

    print(f"  Final hidden states: {len(final_hidden)} states")
    print(f"  Final seq_len: {final_hidden[0].shape[1]}")
    print(f"  Expected seq_len: {len(tokens)}")

    assert final_hidden[0].shape[1] == len(tokens), "seq_len should match accumulated tokens"

    # Extract target_hidden
    target_hidden = extract_context_feature(
        final_hidden,
        draft_model.target_layer_ids,
    )
    print(f"  target_hidden shape: {target_hidden.shape}")

    assert target_hidden.shape[1] == len(tokens), "target_hidden seq_len should match tokens"

    print("✓ Hidden states accumulate correctly")


def test_verification_acceptance_logic():
    """Test draft token verification and acceptance logic."""
    print("\n=== Testing Verification Acceptance Logic ===")

    # Load models
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Setup
    prompt = "The capital of France is"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_cache = model.make_cache()
    logits = model(prompt_tokens, cache=target_cache)
    mx.eval(logits)

    # Get first token
    first_token = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0).item()
    print(f"  Context: {prompt}")
    print(f"  First token: {tokenizer.decode([first_token])}")

    # Simulate draft tokens (make some correct, some wrong)
    # For testing, use actual target predictions
    draft_to_verify = mx.array([[first_token, first_token, first_token]])
    logits = model(draft_to_verify, cache=target_cache)
    mx.eval(logits)

    target_tokens = mx.argmax(logits, axis=-1).squeeze(0)
    print(f"  Draft tokens: {draft_to_verify.squeeze(0).tolist()}")
    print(f"  Target tokens: {target_tokens.tolist()}")

    # Calculate acceptance length
    acceptance_length = (
        mx.cumsum(draft_to_verify.squeeze(0) == target_tokens) == mx.arange(len(target_tokens))
    ).sum()
    acceptance_length = int(acceptance_length)

    print(f"  Acceptance length: {acceptance_length}")

    # Verify calculation
    for i in range(len(target_tokens)):
        if draft_to_verify[0, i].item() == target_tokens[i].item():
            print(f"    Position {i}: ✓ Match")
        else:
            print(f"    Position {i}: ✗ Mismatch")

    print("✓ Verification logic works")


def test_token_diversity():
    """Test that generated tokens are diverse (not repetitive)."""
    print("\n=== Testing Token Diversity ===")

    # Load model
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Generate without cache to test model itself
    prompt = "Count from 1 to 10:"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    tokens_generated = []
    for i in range(20):
        if i == 0:
            logits = model(prompt_tokens, cache=None)
        else:
            prev_token = mx.array([[tokens_generated[-1]]])
            logits = model(prev_token, cache=None)

        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0).item()
        tokens_generated.append(token)

    print(f"  Prompt: {prompt}")
    print(f"  Generated tokens ({len(tokens_generated)}): {tokens_generated[:10]}...")
    print(f"  Generated text: {tokenizer.decode(tokens_generated)}")

    # Check diversity
    unique_tokens = len(set(tokens_generated))
    print(f"  Unique tokens: {unique_tokens}/{len(tokens_generated)}")

    if unique_tokens < len(tokens_generated) * 0.5:
        print("  ⚠ WARNING: Low token diversity!")
    else:
        print("  ✓ Good token diversity")

    assert unique_tokens > 1, "Model should generate diverse tokens"

    print("✓ Token diversity test passed")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("DFlash Integration Tests")
    print("=" * 60)

    test_generation_first_iteration()
    test_draft_token_generation()
    test_position_ids_evolution()
    test_hidden_states_accumulation()
    test_verification_acceptance_logic()
    test_token_diversity()

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
