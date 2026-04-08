# DFlash Implementation Learnings

## Overview

Implementing DFlash (block diffusion speculative decoding) for MLX-LM revealed several key insights about MLX's cache system and the challenges of integrating draft models with target models.

## Cache Types in MLX

### KVCache
- Uses **offset-based tracking**: `cache.offset` tracks valid sequence length
- Pre-allocates in blocks of 256 tokens
- `trim(n)` decrements offset (doesn't modify arrays)
- `update_and_fetch()` writes at current offset position
- Fast rollback: just decrement offset, no copying needed

### ArraysCache
- Used for linear attention layers in Qwen models
- Stores **conv_state**: fixed rolling window (conv_kernel_size - 1 tokens)
- No offset tracking originally - arrays grow with `extend()`
- Can't use simple trim() - actual tensor data gets overwritten
- Rolling window slides when new tokens processed, old data lost

## Key Discovery: Cache Corruption

When running draft tokens through target model for verification:
1. **KVCache layers**: draft tokens appended, offset advances
2. **ArraysCache layers**: conv_state updated with draft tokens, rolling window slides
3. On rejection: need to rollback BOTH cache types

### The ArraysCache Problem

```python
# Before draft tokens:
conv_state = [token_5, token_6, token_7]  # Last 3 tokens

# After processing 15 draft tokens:
conv_state = [token_20, token_21, token_22]  # Old data GONE

# Can't just mask with lengths - actual tensor data changed
```

The rolling window nature means we MUST save/restore the actual arrays.

## Hybrid Solution

### KVCache Rollback (Fast)
```python
for c in target_cache:
    if hasattr(c, 'trim'):
        c.trim(num_rejected)  # Just decrements offset
```

### ArraysCache Rollback (Necessary)
```python
# Save before draft verification
arrays_cache_state = []
for i, c in enumerate(target_cache):
    if hasattr(c, 'cache') and c.cache[0] is not None:
        arrays_cache_state.append((i, c.cache[0], c.cache[1]))

# Restore after rejection
for idx, saved_k, saved_v in arrays_cache_state:
    target_cache[idx].cache[0] = saved_k
    target_cache[idx].cache[1] = saved_v
```

**Key insight**: The model creates new arrays with `mx.contiguous()`, so saved references remain valid. No need for `mx.array()` copying.

## Draft Token Sampling Bug

Original code had off-by-one error:
```python
# WRONG - skips first position
draft_tokens_block = mx.argmax(draft_logits[:, 1:, :], axis=-1)

# CORRECT - include all positions
draft_tokens_block = mx.argmax(draft_logits[:, :, :], axis=-1)
```

The draft_input is `[prev_token, noise_0, noise_1, ...]`:
- Position 0: prediction after prev_token (FIRST draft token)
- Position 1: prediction after [prev_token, noise_0]
- etc.

## Performance

### Current State
- **Target-only**: 11.2 tok/s
- **With DFlash (0% acceptance)**: 3.7 tok/s

The overhead comes from:
1. Running draft tokens through target model (wasted work with 0% acceptance)
2. ArraysCache save/restore (minimal overhead with reference approach)
3. Draft cache re-materialization each iteration

### Optimization Opportunities
1. **Fix acceptance rate** - biggest impact (currently 0%)
2. Skip ArraysCache save/restore if no rejection occurred
3. Reuse draft cache instead of regenerating each iteration

## Model Compatibility

The 4B models show 0% acceptance rate, suggesting incompatibility between:
- `Qwen/Qwen3.5-4B` (target)
- `z-lab/Qwen3.5-4B-DFlash` (draft)

Possible causes:
1. Draft model uses noise-based generation (fundamentally different predictions)
2. Training mismatch between draft and target
3. Architectural differences in how context is processed

The 27B models should be tested to see if acceptance improves.

## Hidden States Accumulation

The draft cache needs to be re-materialized each iteration because:
1. Target cache grows with accepted tokens
2. Draft cache must stay in sync
3. We accumulate hidden states from `ModelWithHiddenStates` wrapper

```python
# Accumulate hidden states (only for accepted tokens)
for i, h in enumerate(target_model_with_hidden.hidden_states):
    accumulated_hidden[i] = mx.concatenate([accumulated_hidden[i], h], axis=1)

# Re-materialize draft cache with full context
current_target_hidden = extract_context_feature(accumulated_hidden, target_layer_ids)
draft_cache = draft_model.make_cache()
draft_model.materialize_target_hidden(current_target_hidden, draft_cache, ctx_position_ids)
```

## Files Modified

1. **mlx_lm/models/cache.py**:
   - Added `offset` property to ArraysCache
   - Added `trim()` method to ArraysCache
   - Added `size()` method to ArraysCache
   - Added `is_trimmable()` method to ArraysCache

2. **mlx_lm/generate_dflash_v2.py**:
   - Fixed draft token sampling (removed `[:, 1:, :]` slicing)
   - Implemented hybrid cache rollback approach
   - Added hidden states accumulation
   - Added draft cache re-materialization each iteration

## Next Steps

1. Test with 27B models to check if acceptance rate improves
2. Investigate why 4B models have 0% acceptance
3. Consider alternative draft models or training approaches
4. Profile performance to identify other bottlenecks
