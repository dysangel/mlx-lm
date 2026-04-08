# DFlash Acceptance Rate Investigation Plan

## Current State

**Problem:** Acceptance rate dropped from ~92% to 0% when we were fixing cache corruption.

**Root Cause Found:** In commit fixing cache corruption, draft token sampling was accidentally changed:
- **Working:** `draft_logits[:, -current_block_size + 1:, :]` (take last block_size-1 positions)
- **Broken:** `draft_logits[:, :, :]` (take ALL positions)

This broke the alignment between draft tokens and target model predictions.

## What Was Fixed

1. ✅ Restored correct draft token sampling logic
2. ✅ Restored draft_tokens construction: `[first_token] + draft_tokens_block`
3. ✅ Restored verification logic: compare `draft_tokens[1:]` with target predictions
4. ✅ Restored token yielding: yield accepted drafts + one target token per iteration
5. ✅ ArraysCache rollback implemented (reference-based, minimal overhead)
6. ✅ KVCache rollback using trim() (fast, no copying)

## What Needs Testing

1. **Test acceptance rate** - Does fixing the sampling bug restore 92% acceptance?
2. **Test output quality** - Is the output coherent (not repetitive)?
3. **Test speed** - What's the tok/s with good acceptance?

## Next Steps

### Immediate (Test the fix)
```bash
python test_dflash_gen.py
```

Check:
- Acceptance rate debug output
- Output quality  
- Speed

### If acceptance is still 0%:

**Hypothesis 1: Draft model interface mismatch**
- Check if draft model call is correct
- The draft model takes: `position_ids, noise_embedding, target_hidden, cache`
- Our code might be passing wrong parameters

**Hypothesis 2: target_hidden not being passed correctly**
- target_hidden should be [B, ctx_len, num_layers * D]
- Check if extract_context_feature is producing correct shape
- Check if target_hidden is being updated correctly with each iteration

**Hypothesis 3: Position IDs mismatch**
- Draft model expects positions from cache_size to cache_size + block_size
- Target model expects positions from accumulated tokens
- These might be misaligned

### Debug Commands

```bash
# Check draft model call
python -c "
from mlx_lm import load
draft, _ = load('z-lab/Qwen3.5-4B-DFlash')
print(draft.__class__.__name__)
# Check what parameters it expects
"

# Test with 27B models (might have better compatibility)
# Edit test_dflash_gen.py to use 27B models
```

## Files to Review

1. **mlx_lm/generate_dflash_v2.py** - Main generation loop
   - Lines ~290-320: Draft token generation and sampling
   - Lines ~320-340: Draft token verification
   - Lines ~340-370: Token yielding and accumulation

2. **mlx_lm/models/dflash_v2.py** - Draft model interface
   - Line 303: `__call__` method signature
   - Check what parameters are required

## Key Code Sections

### Draft Token Sampling (FIXED)
```python
# CORRECT - from working version
draft_tokens_block = sampler(draft_logits[:, -current_block_size + 1:, :]).squeeze(0)
draft_tokens = mx.concatenate([mx.array([first_token]), draft_tokens_block])
```

### Verification Logic (FIXED)
```python
draft_to_verify = draft_tokens[1:][None, :]  # Skip first token (seed)
logits = model(draft_to_verify, cache=target_cache)
target_tokens = mx.argmax(logits, axis=-1).squeeze(0)
acceptance_length = (
    mx.cumsum(draft_to_verify.squeeze(0) == target_tokens) == mx.arange(len(target_tokens))
).sum()
```

### ArraysCache Rollback (NECESSARY)
```python
# Save before draft verification
for i, c in enumerate(target_cache):
    if hasattr(c, 'cache') and c.cache[0] is not None:
        arrays_cache_state.append((i, c.cache[0], c.cache[1]))

# Restore after rejection
for idx, saved_k, saved_v in arrays_cache_state:
    target_cache[idx].cache[0] = saved_k
    target_cache[idx].cache[1] = saved_v
```

## Success Criteria

- [ ] Acceptance rate > 50% (ideally 80%+)
- [ ] Output quality coherent (matches target-only quality)
- [ ] Speed > 8 tok/s (improvement over current 3.7 tok/s)
- [ ] Both 4B and 27B models work

## Reference Commits

- `2d7f5fa` - "fix: DFlash block diffusion generation working" (had good acceptance)
- `9f0470f` - Current commit (restored sampling logic)
