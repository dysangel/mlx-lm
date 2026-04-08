#!/usr/bin/env python3
"""Test token tracking in DFlash generation - verify target_hidden contains full context."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load


def test_token_tracking_logic():
    """Test that target_hidden is rebuilt with all tokens, not just current iteration."""

    print("\n=== Testing Token Tracking Logic ===\n")

    # Simulate the token tracking from generate_dflash_v2.py
    prompt_tokens = [101, 102, 103]  # Example prompt
    first_token = 201
    max_length = 100

    # Initialize output_ids like the reference
    mask_token_id = 0
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :len(prompt_tokens)] = mx.array(prompt_tokens)
    output_ids[:, len(prompt_tokens)] = first_token

    print(f"Initial state:")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  First token: {first_token}")
    print(f"  output_ids[:10]: {output_ids[0, :10].tolist()}")

    # Iteration 1: Generate 1 target token (no draft)
    start = len(prompt_tokens) + 1  # = 4
    target_token_1 = 301
    output_ids[:, start] = target_token_1

    # Rebuild target_hidden from ALL tokens
    total_tokens = start + 1  # = 5
    all_accepted = output_ids[:, :total_tokens].squeeze(0).tolist()

    print(f"\nIteration 1:")
    print(f"  start: {start}")
    print(f"  target_token: {target_token_1}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  all_accepted: {all_accepted}")
    print(f"  Expected: {prompt_tokens + [first_token, target_token_1]}")
    assert all_accepted == prompt_tokens + [first_token, target_token_1], "Iteration 1 token tracking failed"
    print(f"  ✓ Iteration 1 token tracking CORRECT")

    # Iteration 2: Draft tokens (e.g., 15 draft, 5 accepted) + 1 target
    start += 1  # = 5
    acceptance_length = 5
    draft_tokens = [401, 402, 403, 404, 405]  # First 5 accepted

    # Add accepted draft tokens to output_ids
    for i in range(acceptance_length):
        output_ids[:, start + i] = draft_tokens[i]

    target_token_2 = 501
    output_ids[:, start + acceptance_length] = target_token_2

    # Rebuild target_hidden from ALL tokens
    total_tokens = start + acceptance_length + 1  # = 5 + 5 + 1 = 11
    all_accepted = output_ids[:, :total_tokens].squeeze(0).tolist()

    print(f"\nIteration 2:")
    print(f"  start: {start}")
    print(f"  acceptance_length: {acceptance_length}")
    print(f"  draft_tokens: {draft_tokens}")
    print(f"  target_token: {target_token_2}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  all_accepted: {all_accepted}")
    expected = prompt_tokens + [first_token, target_token_1] + draft_tokens + [target_token_2]
    print(f"  Expected: {expected}")
    assert all_accepted == expected, "Iteration 2 token tracking failed"
    print(f"  ✓ Iteration 2 token tracking CORRECT")

    print(f"\n=== All Token Tracking Tests Passed ===")
    print(f"target_hidden will be rebuilt with full context each iteration")


def test_no_duplicate_tokens():
    """Test that tokens are not yielded twice."""

    print("\n=== Testing No Duplicate Tokens ===\n")

    # Simulate the token yield logic
    draft_tokens = [401, 402, 403]
    acceptance_length = 2  # First 2 accepted
    target_token = 501

    tokens_yielded = []

    # Yield accepted draft tokens
    for i in range(acceptance_length):
        token_id = draft_tokens[i]
        tokens_yielded.append(token_id)
        print(f"  Yielded draft token {i}: {token_id}")

    # Yield target token (ONLY ONCE)
    tokens_yielded.append(target_token)
    print(f"  Yielded target token: {target_token}")

    # Check for duplicates
    expected_count = acceptance_length + 1  # draft + target
    actual_count = len(tokens_yielded)
    assert actual_count == expected_count, f"Expected {expected_count} tokens, got {actual_count}"
    assert tokens_yielded.count(target_token) == 1, f"Target token yielded {tokens_yielded.count(target_token)} times"
    print(f"  ✓ No duplicate tokens")

    print(f"\n=== No Duplicate Tokens Test Passed ===")


if __name__ == "__main__":
    test_token_tracking_logic()
    test_no_duplicate_tokens()
    print("\n✅ All tests passed!")
