# DFlash Implementation Progress

## Status: Partially Working

Output is coherent but not matching target quality. Acceptance rates are good but draft generates repetitive tokens.

## What's Fixed

1. ✅ Cache save/restore for ArraysCache (skip trimming - rolling buffers auto-overwrite)
2. ✅ Draft token sampling (correct slicing)
3. ✅ Position IDs (use `start` not `ctx_len`)
4. ✅ output_ids tracking (update when yielding)
5. ✅ Logprobs index (correct offset)
6. ✅ RoPE application (only to noise part)
7. ✅ mask_token_id usage (not zeros)
8. ✅ block_size check (skip draft when current_block_size <= 1)
9. ✅ Cache copy (use mx.array() not .copy())

## Current Issues

### Repetitive Draft Tokens
- Draft generates: `[2199, 2199, 2199, 2199, 2199]` ("life life life...")
- Target accepts: 15/15 (suspicious!)
- This means target model ALSO predicts repetitive tokens

### Output Quality
- DFlash: "2, k KK K K K presenceKK life life life life..."
- Target: "2+2=4. What is 2+2+2? 2+2+2=6. What is..."
- Acceptance rate: Good (60-100%)
- But quality: Poor (wrong tokens, repetitive)

## Root Cause Hypothesis

The `target_hidden` passed to draft model is corrupted or has wrong shape. When target_hidden is wrong:
1. Draft model gets wrong context → generates repetitive tokens
2. Target model's verification also uses wrong context → accepts repetitive tokens

## Next Steps

1. Inspect target_hidden shape and values at each iteration
2. Compare with reference implementation's target_hidden
3. Verify extraction from correct hidden_states (MLX vs PyTorch difference)
4. Check if ArraysCache is correctly tracking sequence position

## Memory: Qwen3.5 Hybrid Model

Qwen3.5 uses both KVCache (attention) and ArraysCache (linear layers). See `memory/project_qwen_hybrid.md`.

ArraysCache stores conv_state (arbitrary arrays), not K/V pairs. ArraysCache.offset returns array size, not sequence position.
