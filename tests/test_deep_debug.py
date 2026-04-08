#!/usr/bin/env python3
"""Deep debugging: Compare draft model behavior step by step."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import (
    ModelWithHiddenStates,
    extract_context_feature,
    get_inner_model,
)


def test_draft_model_call_detailed():
    """Test draft model call with detailed inspection."""
    print("\n=== Deep Debug: Draft Model Call ===")

    # Load models
    model, tokenizer = load("Qwen/Qwen3.5-27B")
    draft_model, _ = load("z-lab/Qwen3.5-27B-DFlash")

    # Setup: simple prompt
    prompt = "The capital of"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Get target hidden states
    target_layer_ids = [1, 16, 31, 46, 61]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)

    # Prefill
    output = wrapped_model(tokens, cache=None)
    mx.eval(output.logits)

    # Get first token
    first_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()
    print(f"Prompt: '{prompt}'")
    print(f"First token: {tokenizer.decode([first_token])} (id={first_token})")

    # Extract target_hidden from PROMPT ONLY (like reference prefill)
    target_hidden = extract_context_feature(
        wrapped_model.hidden_states,
        target_layer_ids,
    )
    mx.eval(target_hidden)

    print(f"\ntarget_hidden from prefill:")
    print(f"  Shape: {target_hidden.shape}")
    print(f"  Mean: {target_hidden.mean().item():.4f}")
    print(f"  Std: {target_hidden.std().item():.4f}")
    print(f"  Min: {target_hidden.min().item():.4f}")
    print(f"  Max: {target_hidden.max().item():.4f}")

    # Draft parameters
    block_size = 4  # Use small block for easier debugging
    ctx_len = target_hidden.shape[1]

    print(f"\nDraft parameters:")
    print(f"  block_size: {block_size}")
    print(f"  ctx_len: {ctx_len}")

    # Calculate position_ids
    # Reference: position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]
    # For first call after prefill, start should be num_input_tokens + 1
    num_input_tokens = tokens.shape[1]
    start = num_input_tokens + 1  # We've generated 1 token (first_token)
    position_ids = mx.arange(start, start + block_size)[None, :]

    print(f"\nPosition IDs:")
    print(f"  num_input_tokens: {num_input_tokens}")
    print(f"  ntoks after first_token: 1")
    print(f"  start: {start}")
    print(f"  position_ids: {position_ids.tolist()}")
    print(f"  Expected (reference): [{start}, {start+1}, {start+2}, {start+3}]")

    # Create noise_embedding
    inner_model = get_inner_model(model)
    prev_token = mx.array([[first_token]])
    noise_tokens = mx.full([1, block_size - 1], draft_model.mask_token_id, dtype=mx.uint32)
    draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)
    noise_embedding = inner_model.embed_tokens(draft_input)
    mx.eval(noise_embedding)

    print(f"\nNoise input:")
    print(f"  prev_token: {prev_token.item()} ({tokenizer.decode([prev_token.item()])})")
    print(f"  noise_tokens: {noise_tokens.squeeze().tolist()}")
    print(f"  draft_input shape: {draft_input.shape}")
    print(f"  noise_embedding shape: {noise_embedding.shape}")

    # Check: what does mask_token_id decode to?
    print(f"\nMask token:")
    print(f"  mask_token_id: {draft_model.mask_token_id}")
    try:
        decoded_mask = tokenizer.decode([draft_model.mask_token_id])
        print(f"  Decoded: {repr(decoded_mask)}")
    except:
        print(f"  Decoded: <error>")

    # Call draft model
    draft_cache = draft_model.make_cache()

    print(f"\nCalling draft_model:")
    print(f"  position_ids: {position_ids.tolist()}")
    print(f"  noise_embedding shape: {noise_embedding.shape}")
    print(f"  target_hidden shape: {target_hidden.shape}")
    print(f"  cache types: {[type(c).__name__ for c in draft_cache[:3]]}")

    draft_output = draft_model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=draft_cache,
    )
    mx.eval(draft_output)

    print(f"\nDraft output:")
    print(f"  Shape: {draft_output.shape}")
    print(f"  Mean: {draft_output.mean().item():.4f}")
    print(f"  Std: {draft_output.std().item():.4f}")
    print(f"  Min: {draft_output.min().item():.4f}")
    print(f"  Max: {draft_output.max().item():.4f}")

    # Get draft logits
    if hasattr(model, 'lm_head'):
        draft_logits = model.lm_head(draft_output)
    else:
        draft_logits = inner_model.embed_tokens.as_linear(draft_output)
    mx.eval(draft_logits)

    print(f"\nDraft logits:")
    print(f"  Shape: {draft_logits.shape}")

    # Sample draft tokens
    draft_tokens_block = mx.argmax(draft_logits[:, -block_size + 1:, :], axis=-1).squeeze(0)

    print(f"\nDraft tokens (sampled):")
    for i, token in enumerate(draft_tokens_block.tolist()):
        decoded = tokenizer.decode([token])
        print(f"  Position {i}: token={token}, decoded={repr(decoded)}")

    # Get TOP 5 tokens for each position (to see what draft model prefers)
    print(f"\nTop 5 tokens at each draft position:")
    top_k = 5
    for pos in range(draft_logits.shape[1] - block_size + 1, draft_logits.shape[1]):
        pos_in_logits = pos + 1  # Because we sliced from position 1
        logits_pos = draft_logits[:, pos_in_logits, :]
        top_tokens = mx.argsort(logits_pos, axis=-1, descending=True)[0, :top_k]
        top_probs = mx.sort(logits_pos, axis=-1, descending=True)[0, :top_k]
        top_probs = mx.softmax(top_probs, axis=-1)

        print(f"  Position {pos}:")
        for j in range(top_k):
            token = top_tokens[0, j].item()
            prob = top_probs[0, j].item()
            decoded = tokenizer.decode([token])
            print(f"    {j}. {token:6d} ({prob*100:5.1f}%): {repr(decoded)}")

    # Now verify with target model
    print(f"\n--- Verification with Target Model ---")

    # Construct draft_tokens for verification
    draft_tokens = mx.concatenate([mx.array([first_token]), draft_tokens_block])
    draft_to_verify = draft_tokens[1:][None, :]

    print(f"draft_to_verify: {draft_to_verify.squeeze().tolist()}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in draft_to_verify.squeeze().tolist()]}")

    # Get target predictions
    target_logits = model(draft_to_verify, cache=None)
    mx.eval(target_logits)
    target_tokens = mx.argmax(target_logits, axis=-1).squeeze(0)

    print(f"target_tokens: {target_tokens.tolist()}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in target_tokens.tolist()]}")

    # Compare
    matches = (draft_to_verify.squeeze(0) == target_tokens).tolist()
    print(f"\nMatches: {matches}")

    # Get target top tokens
    print(f"\nTop 5 target tokens at each position:")
    for pos in range(draft_to_verify.shape[1]):
        logits_pos = target_logits[:, pos, :]
        top_tokens = mx.argsort(logits_pos, axis=-1, descending=True)[0, :5]
        top_probs = mx.sort(logits_pos, axis=-1, descending=True)[0, :5]
        top_probs = mx.softmax(top_probs, axis=-1)

        print(f"  Position {pos}:")
        for j in range(5):
            token = top_tokens[0, j].item()
            prob = top_probs[0, j].item()
            decoded = tokenizer.decode([token])
            print(f"    {j}. {token:6d} ({prob*100:5.1f}%): {repr(decoded)}")

    return draft_tokens_block, target_tokens


if __name__ == "__main__":
    test_draft_model_call_detailed()
