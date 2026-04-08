# DFlash Implementation - Bug Summary and Current State

## Overview

DFlash implementation has been systematically debugged through unit tests and direct comparison with the PyTorch reference. Multiple bugs have been fixed, but the fundamental issue remains: **draft models generate garbage tokens**.

## Bugs Fixed

### 1. Draft Token Sampling Bug (CRITICAL)
**Issue**: Draft token sampling was changed from `[:, -current_block_size + 1:, :]` to `[:, :, :]`, which broke alignment.
**Fix**: Restored correct sampling logic
**Impact**: Acceptance dropped from ~92% to 0%

### 2. Duplicate Code Blocks
**Issue**: First token yielded twice, duplicate yield blocks
**Fix**: Removed duplicates
**Impact**: Caused crashes and double-yielded tokens

### 3. Undefined Variables
**Issue**: `accumulated_tokens`, `last_token_id` undefined
**Fix**: Proper initialization and rename to `first_token`
**Impact**: Crashes

### 4. Wrong Logprobs Index
**Issue**: Used `draft_logits[:, i, :]` instead of `draft_logits[:, i + 1, :]`
**Fix**: Corrected index to match draft_tokens_block
**Impact**: Wrong logprobs for yielded tokens

### 5. output_ids Not Updated
**Issue**: `output_ids` never updated after accepting tokens, causing `prev_token` to always be stale
**Fix**: Update `output_ids` when yielding tokens
**Impact**: Draft model always used same seed token

### 6. Position IDs Off-by-One
**Issue**: Used `ctx_len` instead of `start` for position IDs, also `num_input_tokens` calculation was wrong
**Fix**: Use `start` and calculate `num_input_tokens = prompt_tokens.shape[1]`
**Impact**: Position IDs were `[3, 4, 5, ...]` instead of `[4, 5, 6, ...]`

### 7. RoPE Applied to All Keys
**Issue**: RoPE was applied to entire k (context + noise) instead of just noise part
**Fix**: Split k into context and noise parts, apply RoPE only to noise
**Impact**: Shape mismatch errors, incorrect embeddings

### 8. mask_token_id Not Used
**Issue**: Used `mx.zeros` instead of `mask_token_id` for noise tokens
**Fix**: Use `mx.full(..., mask_token_id, dtype=mx.uint32)`
**Impact**: Wrong embeddings for noise positions

## Current State

### What Works
- ✓ Target-only generation: **PERFECT** ("2+2 equals 4. This is a basic arithmetic operation...")
- ✓ Hidden states extraction
- ✓ Cache operations (KVCache and ArraysCache)
- ✓ Verification logic
- ✓ Acceptance calculation

### What Doesn't Work
- ✗ Draft model generates garbage tokens
- ✗ Low acceptance rates (14-20%)
- ✗ Garbled output ("ম矿泉水YESarterমarterarterarterarterwicharterอย่างarterarterอย่างarterarterอย่าง.getField")

### Root Cause Analysis

**Draft model generates tokens that don't match target model:**
- Draft: `['():', '():', '最低']`
- Target: `['-', '-', 'android']`
- Acceptance: 1/7 (14%)

This suggests:
1. Draft model is fundamentally broken
2. Draft model is incompatible with target model
3. Or we're calling it incorrectly

### Test Results

#### Target-Only Generation (Perfect)
```
What is 2+2?
2+2 equals 4. This is a basic arithmetic operation where you add two numbers together.
Speed: 36.0 tok/s
```

#### With DFlash (Garbled)
```
What is 2+2?
2ম矿泉水YESarterমarterarterarterarterwicharterอย่างarterarterอย่างarterarterอย่าง.getField
Speed: 10.9 tok/s
```

## Comparison Test Results

### Test 1: Hidden States Extraction ✓
- target_hidden shape correct: (1, seq_len, num_layers * hidden_size)
- All components working correctly

### Test 2: Draft Token Sampling ✓ (after fix)
- Position IDs now correct: [4, 5, 6, 7, 8, 9, 10, 11]
- Was off-by-one before fix

### Test 3: Draft Model Output
- draft_output shape correct: (1, block_size, hidden_size)
- But tokens are garbage

### Test 4: Verification ✓
- Acceptance logic working correctly
- Target tokens computed correctly

### Test 5: Cache Behavior
- ArraysCache doesn't track offset like KVCache
- Different cache types for different model architectures

## Key Differences from Reference

1. **Position IDs**: Fixed to use `start` instead of `ctx_len`
2. **output_ids tracking**: Now updates when yielding tokens
3. **Logprobs index**: Fixed to use correct index for draft tokens
4. **Context accumulation**: Using full accumulated context for target_hidden

## Recommendations

1. **Try alternative draft models**: Current 4B and 27B models are broken
2. **Check training compatibility**: Ensure draft/target were trained together
3. **Verify draft model interface**: May be calling it incorrectly
4. **Consider temperature sampling**: Break repetitive patterns
5. **Test with PyTorch reference**: Run side-by-side comparison

## Files Modified

1. `mlx_lm/generate_dflash_v2.py` - Main generation loop
2. `mlx_lm/models/dflash_v2.py` - RoPE application
3. `tests/test_dflash_components.py` - Unit tests
4. `tests/test_dflash_generation_steps.py` - Integration tests
5. `tests/test_direct_comparison.py` - Comparison tests
