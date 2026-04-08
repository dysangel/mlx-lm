# DFlash Debugging - Summary of Fixes

## Bugs Found and Fixed

### 1. Duplicate Code Blocks
**File**: `mlx_lm/generate_dflash_v2.py`
**Issue**: First token yielded twice, duplicate yield blocks
**Fix**: Removed duplicates

### 2. Undefined Variables
**Issue**: `accumulated_tokens`, `last_token_id` undefined
**Fix**: Proper initialization and rename to `first_token`

### 3. Wrong Logprobs Index
**Issue**: Used `draft_logits[:, i, :]` instead of `[:, i + 1, :]`
**Fix**: Corrected index to match draft_tokens_block

### 4. output_ids Not Updated
**Issue**: `output_ids` never updated, causing stale prev_token
**Fix**: Update output_ids when yielding tokens

### 5. Position IDs Off-by-One
**Issue**: Used `ctx_len` instead of `start`, wrong `num_input_tokens`
**Fix**: Use `start` and calculate `num_input_tokens = prompt_tokens.shape[1]`

### 6. RoPE Applied to All Keys
**Issue**: RoPE applied to entire k (context + noise) instead of just noise
**Fix**: Split k and apply RoPE only to noise part

### 7. mask_token_id Not Used
**Issue**: Used `mx.zeros` instead of `mask_token_id` for noise tokens
**Fix**: Use `mx.full(..., mask_token_id, dtype=mx.uint32)`

### 8. Cache Copy Bug
**Issue**: ArraysCache save/restore used references, not copies
**Fix**: Use `mx.array(c.cache[0])` to copy

### 9. start Increment
**Issue**: Always incremented by 1 instead of `acceptance_length + 1`
**Fix**: `start += acceptance_length + 1` after accepting tokens

### 10. target_hidden Extraction
**Issue**: Used accumulated_hidden (full context) instead of only last block
**Fix**: Extract from verification hidden_states, slice to `:acceptance_length + 1`

### 11. Draft Cache Not Trimmed
**Issue**: Draft cache grew unbounded
**Fix**: Re-create draft_cache each iteration

### 12. ArraysCache Trimming
**Issue**: ArraysCache has both `trim` and `cache` attributes, was being excluded
**Fix**: Skip trimming ArraysCache (linear layers use rolling buffers)

### 13. block_size Check
**Issue**: When `current_block_size = 1`, draft_logits has only 1 position but we accessed position 1
**Fix**: Skip draft when `current_block_size <= 1`

## Remaining Issues

### Repetitive Draft Tokens (ONGOING)
- Draft generates repetitive tokens: `[2199, 2199, 2199, ...]`
- Target accepts: 15/15 (suspicious!)
- **Hypothesis**: `target_hidden` is corrupted or has wrong shape
- **Impact**: Both draft and target receive wrong context, generate wrong tokens

### Output Quality
- DFlash: "2, k KK K K K presenceKK life life life..."
- Target: "2+2=4. What is 2+2+2? 2+2+2=6. What is..."
- Acceptance rate: Good (60-100%)
- But quality: Poor (wrong tokens, repetitive)

## Key Learnings

1. **Qwen3.5 is hybrid**: Uses KVCache (attention) + ArraysCache (linear layers)
2. **ArraysCache behavior**: Stores conv_state arrays, offset returns array size not sequence position
3. **MLX vs PyTorch**: MLX returns hidden_states only for new tokens with cache, PyTorch returns all
4. **Cache corruption**: Even small bugs in cache handling cause catastrophic output degradation
