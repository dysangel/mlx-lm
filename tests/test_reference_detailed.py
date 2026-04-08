#!/usr/bin/env python3
"""Detailed comparison of MLX vs PyTorch reference implementations."""

import sys
sys.path.insert(0, '.')

print("=" * 70)
print("MLX vs PyTorch Reference - Key Implementation Differences")
print("=" * 70)

print("""
Based on analysis of reference implementation (dflash/benchmark.py):

KEY DIFFERENCES FOUND:

1. POSITION IDS CALCULATION
   Reference:
     position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]
     - Uses draft cache sequence length
     - Crops to [start:start+block_size]

   MLX (current):
     mx.arange(ctx_len, ctx_len + current_block_size)
     - Uses target_hidden length (ctx_len)
     - May be misaligned with draft cache

   FIX: Track draft cache position separately


2. DRAFT CACHE MANAGEMENT
   Reference:
     past_key_values_draft.crop(start)
     - Crops cache to current position
     - Reuses cache across iterations

   MLX (current):
     draft_cache = draft_model.make_cache()  # Fresh each time
     - Creates new cache each iteration
     - Loses position tracking

   FIX: Reuse draft cache and implement crop()


3. TARGET CACHE MANAGEMENT
   Reference:
     past_key_values_target.crop(start)
     - Crops to accepted position

   MLX (current):
     target_cache.trim(num_rejected)  # For KVCache
     - Only trims rejected tokens
     - ArraysCache save/restore workaround

   FIX: Use crop-style positioning


4. TARGET_HIDDEN UPDATE
   Reference:
     target_hidden = extract_context_feature(
         output.hidden_states,
         model.target_layer_ids
     )[:, :acceptance_length + 1, :]

   MLX (current):
     accumulated_tokens_mx = mx.array(accumulated_tokens)[None, :]
     _ = target_model_with_hidden(accumulated_tokens_mx, cache=None)
     accumulated_hidden = target_model_with_hidden.hidden_states.copy()

   FIX: Use cached output hidden_states directly


5. NOISE EMBEDDING
   Reference:
     noise_embedding = target.model.embed_tokens(block_output_ids)
     - block_output_ids starts with previous token
     - Then zeros for noise positions

   MLX (current):
     prev_token = output_ids[:, start - 1][:, None]
     noise_tokens = mx.zeros([1, max(0, current_block_size - 1)], dtype=mx.uint32)
     draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)

   SAME: Both use same approach


RECOMMENDED FIXES:

1. Track draft position separately from target context:
   draft_position = 0  # Track position in draft cache
   noise_position_ids = mx.arange(draft_position, draft_position + block_size)
   draft_position += acceptance_length + 1  # Update after acceptance

2. Reuse draft cache with crop-style trimming:
   # Instead of making new cache each iteration
   # Crop to current position
   for c in draft_cache:
       if hasattr(c, 'crop'):
           c.crop(current_position)

3. Use output.hidden_states from verification step:
   # Already captured by ModelWithHiddenStates during verification
   # Just extract from there, don't rebuild from scratch
   target_hidden = extract_context_feature(
       target_model_with_hidden.hidden_states,
       draft_model.target_layer_ids,
   )[:, :acceptance_length + 1, :]


TESTING APPROACH:

1. Create identical test cases:
   - Same prompt
   - Same random seed
   - Same model checkpoints

2. Compare intermediate outputs:
   - target_hidden after prefill
   - draft_logits for same input
   - acceptance patterns
   - final output_ids

3. Debug position alignment:
   - Print position_ids at each step
   - Verify RoPE cos/sin match
   - Check cache offsets

""")

print("=" * 70)
print("Next: Create test that runs both implementations side-by-side")
print("=" * 70)
