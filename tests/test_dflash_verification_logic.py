#!/usr/bin/env python3
"""Test DFlash draft verification logic to ensure correctness."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx


def test_verification_alignment():
    """Test that draft tokens and target predictions are aligned correctly."""

    print("\n=== Testing Draft Verification Alignment ===\n")

    # Simulate the verification logic
    prev_token = mx.array([[100]])  # Seed token at position 5
    draft_tokens_block = mx.array([201, 202, 203, 204, 205])  # Draft tokens for positions 6-10

    print(f"Setup:")
    print(f"  prev_token (position 5): {prev_token.item()}")
    print(f"  draft_tokens_block (positions 6-10): {draft_tokens_block.tolist()}")

    # Construct verification input: seed + draft tokens
    verification_input = mx.concatenate([prev_token, draft_tokens_block[None, :]], axis=-1)

    print(f"\nVerification input shape: {verification_input.shape}")
    print(f"Verification input: {verification_input.tolist()}")

    # Simulate target model output (logits)
    # In real code, this comes from the target model forward pass
    # Here we simulate predictions that partially match draft
    batch_size, seq_len = verification_input.shape
    vocab_size = 1000
    logits = mx.random.normal((batch_size, seq_len, vocab_size))

    # Set specific predictions for testing
    # Position 0 (seed token): predict token 201 (matches draft[0])
    logits[0, 0, 201] = 10.0
    # Position 1: predict token 201 (doesn't match draft[1]=202)
    logits[0, 1, 201] = 10.0
    # Position 2: predict token 202 (doesn't match draft[2]=203)
    logits[0, 2, 202] = 10.0
    # Position 3: predict token 999 (doesn't match draft[3]=204)
    logits[0, 3, 999] = 10.0
    # Position 4: predict token 204 (doesn't match draft[4]=205)
    logits[0, 4, 204] = 10.0

    # Get target predictions (exclude last position)
    target_tokens = mx.argmax(logits[:, :-1, :], axis=-1).squeeze(0)

    print(f"\nTarget predictions (exclude last):")
    for i, token in enumerate(target_tokens.tolist()):
        draft_token = draft_tokens_block[i].item() if i < len(draft_tokens_block) else None
        match = "✓" if token == draft_token else "✗"
        print(f"  Position {i}: draft={draft_token}, target={token} {match}")

    # Find acceptance length (longest prefix where tokens match)
    acceptance_length = (
        mx.cumsum(draft_tokens_block == target_tokens) == mx.arange(len(target_tokens))
    ).sum()
    acceptance_length = int(acceptance_length)

    print(f"\nAcceptance length: {acceptance_length}/{len(draft_tokens_block)}")

    # Verify the result (only first token matches)
    expected_acceptance = 1  # Only first token matches
    assert acceptance_length == expected_acceptance, f"Expected acceptance {expected_acceptance}, got {acceptance_length}"
    print(f"  ✓ Acceptance calculation CORRECT")

    print(f"\n=== Draft Verification Alignment Test Passed ===")


def test_position_ids_calculation():
    """Test that position_ids are calculated correctly for draft model."""

    print("\n=== Testing Position IDs Calculation ===\n")

    # Simulate position_ids calculation
    start = 10  # Current position in output_ids
    current_block_size = 16

    noise_position_ids = mx.arange(start, start + current_block_size)[None, :]

    print(f"start: {start}")
    print(f"current_block_size: {current_block_size}")
    print(f"noise_position_ids: {noise_position_ids.tolist()}")

    # Verify
    expected_ids = list(range(start, start + current_block_size))
    actual_ids = noise_position_ids[0, :].tolist()
    assert actual_ids == expected_ids, f"Expected {expected_ids}, got {actual_ids}"
    print(f"  ✓ Position IDs calculation CORRECT")

    # Verify that RoPE will be applied to correct positions
    print(f"\nRoPE will be applied to positions:")
    print(f"  Position 0 (seed): {actual_ids[0]}")
    print(f"  Position 1 (first noise): {actual_ids[1]}")
    print(f"  Position {current_block_size-1} (last noise): {actual_ids[-1]}")

    print(f"\n=== Position IDs Calculation Test Passed ===")


if __name__ == "__main__":
    test_verification_alignment()
    test_position_ids_calculation()
    print("\n✅ All verification logic tests passed!")
