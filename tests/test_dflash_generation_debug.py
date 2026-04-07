#!/usr/bin/env python3
"""Debug DFlash generation step by step comparing with reference."""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# PyTorch imports
sys.path.insert(0, '/Users/ali/Projects/dflash')
import torch
from dflash.model import DFlashDraftModel, extract_context_feature as ref_extract_context_feature

# MLX imports
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import block_diffusion_generate_step, extract_context_feature

# Model paths
DRAFT_MODEL_ID = "z-lab/Qwen3.5-4B-DFlash"
TARGET_MODEL_ID = "Qwen/Qwen3.5-4B"

def compare_single_step():
    """Compare a single generation step between MLX and PyTorch."""

    print("Loading models...")
    draft_torch = DFlashDraftModel.from_pretrained(DRAFT_MODEL_ID)
    draft_mlx, _ = load(DRAFT_MODEL_ID)
    target_model, tokenizer = load(TARGET_MODEL_ID)

    prompt = "Hello"
    print(f"Prompt: {prompt}\n")

    # Tokenize
    input_ids = tokenizer.encode(prompt)
    input_ids_torch = torch.tensor([input_ids], dtype=torch.long)

    # Get first token from PyTorch reference
    print("=== PyTorch Reference ===")
    with torch.no_grad():
        # Prefill
        past_key_values_target = torch.load('test_cache.pt') if Path('test_cache.pt').exists() else None
        if past_key_values_target is None:
            print("Creating new target cache...")
            # Would need to implement this properly

    # Get first token from MLX
    print("\n=== MLX Implementation ===")
    tokens_list = []
    for i, (token_id, logprobs, from_draft) in enumerate(block_diffusion_generate_step(
        prompt=prompt,
        model=target_model,
        draft_model=draft_mlx,
        tokenizer=tokenizer,
        max_tokens=10,
    )):
        tokens_list.append(token_id)
        token_str = tokenizer.decode([token_id])
        print(f"{i+1}. [{('DRAFT' if from_draft else 'TARG')}] {token_id:5d} ({repr(token_str)})")

    print(f"\nMLX Output: {tokenizer.decode(tokens_list)}")

if __name__ == "__main__":
    compare_single_step()
