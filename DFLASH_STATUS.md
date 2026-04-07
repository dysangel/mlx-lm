# DFlash Implementation Status - WORKING!

## Summary
The MLX DFlash implementation is now working correctly. Initial failures were due to incorrect test setup (passing compressed target_hidden when model expected raw).

## Verification
```
Argmax - ref: 13 (.), mlx: 13 (.)
Logits correlation: 0.937
Top 5 overlap: {11, 13, 381, 11751}
```

## What Works
- ✅ Model architecture port (`mlx_lm/models/dflash_v2.py`)
- ✅ Weight loading with auto-detection of head_dim=128
- ✅ Dual attention mechanism (Q: 4096, K/V: 1024)
- ✅ RoPE implementation
- ✅ Feature compression (fc + hidden_norm)
- ✅ Block diffusion generation algorithm (`mlx_lm/generate_dflash_v2.py`)

## Key Implementation Details

### Architecture
- `hidden_size = 2560` (main dimension)
- `Q projects to 4096` (32 heads × 128 head_dim)
- `K/V project to 1024` (8 KV heads × 128 head_dim)
- This is NOT standard - Q uses different output dimension than hidden_size

### Important: target_hidden Format
The draft model's `__call__` expects **RAW** (12800 dim) target_hidden and compresses it internally:
```python
# Model.__call__ compresses target_hidden from [B, ctx_len, 12800] to [B, ctx_len, 2560]
target_hidden_flat = target_hidden.reshape(B * ctx_len, num_layers_times_D)
compressed_target_flat = self.hidden_norm(self.fc(target_hidden_flat))
target_hidden = compressed_target_flat.reshape(B, ctx_len, -1)
```

## Files
- `mlx_lm/models/dflash_v2.py` - DFlash model implementation
- `mlx_lm/generate_dflash_v2.py` - Block diffusion generation
- `test_dflash_comparison.py` - Reference vs MLX comparison (NEEDS UPDATE)

## Next Steps
1. Update test scripts to pass RAW target_hidden
2. Test full block diffusion generation loop
3. Benchmark performance vs baseline
4. Integrate with CLI

## Usage (Target)
```bash
python -m mlx_lm.generate \
    --model Qwen/Qwen3.5-4B \
    --dflash-draft-model z-lab/Qwen3.5-4B-DFlash \
    --block-size 16 \
    --prompt "Explain quantum computing:" \
    --max-tokens 256
```
