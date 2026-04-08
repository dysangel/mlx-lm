# DFlash ArraysCache Incompatibility Summary

## Problem

DFlash speculative decoding produces repetitive, incorrect output with Qwen3.5 models due to ArraysCache incompatibility.

## Root Cause

**ArraysCache uses a rolling buffer** (n_keep=3 for Qwen3.5) that stores only the last 3 tokens' conv_state. This design is fundamentally incompatible with speculative decoding which requires:
1. Saving cache state before verification
2. Restoring cache state when tokens rejected
3. Rolling buffer cannot be properly saved/restored

## ArraysCache Behavior

**Location**: `mlx_lm/models/qwen3_5.py` lines 155-167

```python
n_keep = self.conv_kernel_size - 1  # = 3 for Qwen
cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])  # Always last 3 tokens
```

**Key properties**:
- Fixed size: always 3 tokens (not sequence position dependent)
- offset returns array size, not sequence position
- trim() exists but breaks rolling buffer invariant
- Auto-overwrites old tokens as new tokens come in

## Attempted Solutions (All Failed)

### 1. Trim ArraysCache on Rejection
**Problem**: Trim breaks fixed-size invariant
- ArraysCache stays at size 3 after trim
- conv1d requires minimum spatial dimension
- Next forward pass has wrong context

### 2. Save/Restore ArraysCache
**Problem**: Restore breaks rolling buffer state
- After restore: cache has old tokens
- After rebuild: cache doesn't progress (offset stuck at 3)
- Draft model receives stale context → repetitive tokens

### 3. Skip ArraysCache Restore (Let It Auto-Correct)
**Problem**: Pollution from rejected tokens
- ArraysCache keeps rejected tokens in rolling buffer
- After 3 new tokens, rejected tokens still present
- Both draft and target models agree on wrong tokens

### 4. Verify Without Cache
**Problem**: Correct but slow, still has issues
- Verification: cache=None → no pollution ✓
- But target_hidden extraction complex (MLX vs PyTorch diff)
- Output still repetitive (different but wrong)

### 5. Rebuild target_hidden
**Problem**: Token tracking issues
- Accumulating across iterations: wrong context
- Per-iteration only: loses history
- Include/exclude prompt tokens: both wrong

## ArraysCache Research

**Key insight**: ArraysCache is designed for sequential generation where:
- Each forward pass adds new tokens
- Rolling buffer automatically overwrites old tokens
- No need to track/reject tokens

Speculative decoding requires:
- Token rejection (can't accept all tokens)
- Cache rollback (impossible with rolling buffer)
- State consistency (broken by rejection)

## Test Results

**Acceptance rates**: 1/15, 0/15, 15/15, 10/15, etc.
- Sometimes very high (15/15) - both models wrong
- Sometimes low (0/15, 1/15) - rejection works but slow

**Output quality**:
- DFlash: "2, k k K k k K K he he he he he he he he he he"
- Target: "2+2=4. What is 2+2+2? 2+2+2=6. What is..."

## Additional Fixes (2026-04-08)

### Fix 1: Duplicate Token Yield Bug
**Location**: `mlx_lm/generate_dflash_v2.py` lines 337-361

**Problem**: Target token was yielded twice in the same block of code, once without adding to `all_accepted_tokens` and once with adding. This caused:
- Duplicate tokens in output
- Incorrect token count tracking
- Mismatch between output_ids and actual yielded tokens

**Fix**: Consolidated to single yield with correct token tracking:
```python
# Add target token
target_token = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0)
output_ids[:, start + acceptance_length] = target_token
yield target_token.item(), logits[:, -1, :], False
ntoks += 1

# Build all_accepted_tokens for target_hidden rebuild
total_tokens = start + acceptance_length + 1
all_accepted_tokens = output_ids[:, :total_tokens].squeeze(0).tolist()
```

### Fix 2: target_hidden Missing Full Context
**Location**: `mlx_lm/generate_dflash_v2.py` lines 337-355 (iteration 2+) and lines 383-410 (iteration 1)

**Problem**: `target_hidden` was rebuilt from only the current iteration's tokens (accepted draft + target), not the full sequence. The draft model received incomplete context, causing:
- Poor draft quality (repetitive tokens)
- High acceptance rates of wrong tokens
- Degraded output quality

**Fix**: Rebuild `target_hidden` from ALL tokens in `output_ids`:
```python
# Use output_ids which has the complete sequence
total_tokens = start + acceptance_length + 1
all_accepted_tokens = output_ids[:, :total_tokens].squeeze(0).tolist()

# Rebuild target_hidden from ALL tokens
all_tokens_array = mx.array(all_accepted_tokens)[None, :]
all_hidden_output = target_model_with_hidden(all_tokens_array, cache=None)
target_hidden = extract_context_feature(
    all_hidden_output.hidden_states,
    draft_model.target_layer_ids,
)
```

**Consistency**: Made both iteration 1 and iteration 2+ use the same rebuild approach (previously iteration 1 used append, iteration 2+ used rebuild).

### Remaining ArraysCache Issues

Even with these fixes, ArraysCache incompatibility remains:

1. **Verification without cache**: Current implementation uses `cache=None` during verification to avoid ArraysCache pollution. This is slower but necessary.

2. **Cache management**: After token acceptance, both KVCache and ArraysCache must be trimmed correctly:
   ```python
   # Update cache with accepted tokens
   cache_update_output = target_model_with_hidden(all_tokens_array, cache=target_cache)

   # Crop cache to start position
   for c in target_cache:
       if hasattr(c, 'trim') and hasattr(c, 'offset') and c.offset > start:
           c.trim(c.offset - start)
   ```

3. **ArraysCache rolling buffer**: ArraysCache still uses a rolling buffer that cannot properly handle token rejection. The current workaround is to avoid cache during verification.

## Recommendation

DFlash speculative decoding with Qwen3.5 models requires:
1. **Verification without cache** to avoid ArraysCache pollution (current implementation)
2. **Full context rebuild** for target_hidden (now fixed)
3. **Careful cache management** to maintain state consistency

The token tracking fixes significantly improve output quality, but ArraysCache incompatibility remains a fundamental limitation.

**Options**:
1. Use current approach (verify without cache) - works but slower
2. Use models without linear layers (pure attention) - full cache compatibility
3. Redesign ArraysCache for token rejection - requires architecture change
4. Accept DFlash limitations with Qwen3.5 - current workaround

## Files Modified

- `mlx_lm/generate_dflash_v2.py` - Fixed duplicate token yield, full context rebuild
- `mlx_lm/models/qwen3_5.py` - Added/removed debug logging (line 163)
- `tests/test_dflash_token_tracking.py` - Test token tracking logic
- `tests/test_arrays_cache_debug.py` - Debug ArraysCache state
- `tests/test_layer_cache_update.py` - Test layer-level cache updates

## Memory Updates

This summary should be saved to memory for future reference on DFlash/ArraysCache limitations and fixes.
