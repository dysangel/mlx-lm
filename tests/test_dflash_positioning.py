#!/usr/bin/env python3
"""Test DFlash positioning and cache behavior to identify RoPE issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import block_diffusion_generate_step, ModelWithHiddenStates, extract_context_feature

def test_position_ids_evolution():
    """Track how position_ids evolve across iterations."""

    print("Loading models...")
    target_model, tokenizer = load('Qwen/Qwen3.5-4B')
    draft_model, _ = load('z-lab/Qwen3.5-4B-DFlash')

    prompt = "Hello"
    print(f"Prompt: {prompt}\n")

    # Monkey patch to inspect internals
    import mlx_lm.generate_dflash_v2 as gen_mod
    original_step = gen_mod.block_diffusion_generate_step

    iteration_data = []

    def tracked_step(*args, **kwargs):
        """Wrapped generator that tracks internal state."""
        # This is a simplified version - in practice we'd need to instrument the actual function
        for token_id, logprobs, from_draft in original_step(*args, **kwargs):
            iteration_data.append({
                'token': token_id,
                'from_draft': from_draft,
            })
            yield token_id, logprobs, from_draft

    # Run generation
    tokens_list = []
    for i, (token_id, logprobs, from_draft) in enumerate(block_diffusion_generate_step(
        prompt=prompt,
        model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        max_tokens=20,
    )):
        tokens_list.append(token_id)
        print(f"{i+1:2d}. [{('D' if from_draft else 'T')}] {tokenizer.decode([token_id])!r}")

    print(f"\nOutput: {tokenizer.decode(tokens_list)}")

    # Now let's manually trace through the algorithm
    print("\n=== Manual Trace ===")

    # Setup
    prompt_tokens = mx.array([tokenizer.encode(prompt)])
    model_cache = target_model.make_cache()
    draft_cache = draft_model.make_cache()

    # Prefill
    target_with_hidden = ModelWithHiddenStates(target_model, draft_model.target_layer_ids)
    output = target_with_hidden(prompt_tokens, cache=model_cache)

    first_token = mx.argmax(output.logits[0, -1, :])
    print(f"First token: {tokenizer.decode([first_token.item()])}")

    # Get target_hidden
    target_hidden = extract_context_feature(
        target_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )
    print(f"Initial target_hidden shape: {target_hidden.shape}")

    accumulated_tokens = list(prompt_tokens.squeeze(0).tolist()) + [first_token.item()]

    # First iteration
    print("\n--- Iteration 1 ---")
    block_size = draft_model.block_size
    current_block_size = min(block_size, 19)

    # Compute position_ids
    global_pos = len(accumulated_tokens)
    total_positions = global_pos + current_block_size
    position_ids = mx.arange(0, total_positions)[None, :]

    print(f"  accumulated_tokens length: {len(accumulated_tokens)}")
    print(f"  global_pos: {global_pos}")
    print(f"  total_positions: {total_positions}")
    print(f"  position_ids: {position_ids.tolist()}")

    # Check cache sizes
    print(f"  model_cache size: {[c.size() for c in model_cache]}")
    print(f"  draft_cache size: {[c.size() for c in draft_cache]}")

    # What does the reference do?
    print("\n  Reference position_ids would be:")
    print(f"    = position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]")
    print(f"    = position_ids[:, 0: {global_pos + current_block_size}]")
    print(f"    = [0, 1, ..., {global_pos + current_block_size - 1}]")

    # After processing, the cache should grow
    print(f"\n  After draft model, cache should contain:")
    print(f"    - Keys for positions [0, {global_pos + current_block_size})")
    print(f"    - But we only need positions [{global_pos}, {global_pos + current_block_size}) for new tokens")

def test_rope_positions():
    """Test that RoPE is applied to correct positions."""

    print("\n=== Testing RoPE Position Application ===")

    from mlx_lm.models.dflash_v2 import DFlashDraftModel, Attention

    # Load draft model
    draft_model, _ = load('z-lab/Qwen3.5-4B-DFlash')

    # Get an attention layer
    attention_layer = draft_model.layers[0]

    # Create test inputs
    batch_size = 1
    ctx_len = 10  # target_hidden length
    noise_len = 5  # block_size

    # Create dummy hidden states
    target_hidden = mx.random.normal((batch_size, ctx_len, draft_model.config.hidden_size))
    noise_embedding = mx.random.normal((batch_size, noise_len, draft_model.config.hidden_size))

    # Create position_ids - this is what we currently pass
    total_positions = ctx_len + noise_len
    position_ids = mx.arange(0, total_positions)[None, :]

    print(f"ctx_len: {ctx_len}, noise_len: {noise_len}")
    print(f"position_ids shape: {position_ids.shape}")
    print(f"position_ids: {position_ids.tolist()}")

    # What RoPE positions should be applied:
    # - Q (from noise): positions [ctx_len, ctx_len + noise_len) = [10, 15)
    # - K_context (from target_hidden): positions [0, ctx_len) = [0, 10)
    # - K_noise (from noise_embedding): positions [ctx_len, ctx_len + noise_len) = [10, 15)

    print("\nExpected RoPE positions:")
    print(f"  Q: [{ctx_len}, {ctx_len + noise_len})")
    print(f"  K_context: [0, {ctx_len})")
    print(f"  K_noise: [{ctx_len}, {ctx_len + noise_len})")

    # Check what our implementation actually does
    print("\nOur implementation:")
    print(f"  cos shape: ({ctx_len + noise_len}, rotary_dim/2)")
    print(f"  Q gets cos[{ctx_len}:, ...] = positions [{ctx_len}, {ctx_len + noise_len})")
    print(f"  K_context gets cos[:{ctx_len}, ...] = positions [0, {ctx_len})")
    print(f"  K_noise gets cos[{ctx_len}:{ctx_len + noise_len}, ...] = positions [{ctx_len}, {ctx_len + noise_len})")

    # This looks correct! The issue might be elsewhere

if __name__ == "__main__":
    test_position_ids_evolution()
    test_rope_positions()
