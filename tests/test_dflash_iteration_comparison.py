#!/usr/bin/env python3
"""
Test DFlash generation iteration by iteration, comparing with PyTorch reference.
This helps identify where the implementations diverge.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/Users/ali/Projects/dflash')

import torch
import numpy as np
from dflash.model import DFlashDraftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import block_diffusion_generate_step

def test_single_iteration_comparison():
    """Compare the first few iterations in detail."""

    print("Loading models...")
    # PyTorch reference
    draft_torch = DFlashDraftModel.from_pretrained('z-lab/Qwen3.5-4B-DFlash')
    target_torch = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-4B',
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    tokenizer_torch = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')

    # MLX implementation
    draft_mlx, tokenizer_mlx = load('z-lab/Qwen3.5-4B-DFlash')
    target_mlx, _ = load('Qwen/Qwen3.5-4B')

    prompt = "Hello"
    print(f"Prompt: {prompt}\n")

    # Tokenize
    input_ids_torch = torch.tensor([tokenizer_torch.encode(prompt)], dtype=torch.long)
    input_ids_mlx = mx.array([tokenizer_mlx.encode(prompt)])

    # === Prefill Stage ===

    # PyTorch prefill
    print("=== PyTorch Prefill ===")
    with torch.no_grad():
        past_key_values_torch = torch.load('/tmp/torch_cache.pt') if Path('/tmp/torch_cache.pt').exists() else None
        if past_key_values_torch is None:
            # Create cache
            from transformers import DynamicCache
            past_key_values_torch = DynamicCache()

            output_torch = target_torch(
                input_ids_torch,
                past_key_values=past_key_values_torch,
                use_cache=True,
                output_hidden_states=True,
            )

            # Get first token
            first_token_torch = torch.argmax(output_torch.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            input_ids_torch = torch.cat([input_ids_torch, first_token_torch], dim=1)

            # Extract target_hidden
            target_hidden_torch = torch.cat([
                output_torch.hidden_states[i+1][:, -1:, :]
                for i in draft_torch.target_layer_ids
            ], dim=-1)

            print(f"First token (torch): {tokenizer_torch.decode(first_token_torch[0])} (id={first_token_torch[0].item()})")
            print(f"target_hidden shape: {target_hidden_torch.shape}")

    # MLX prefill
    print("\n=== MLX Prefill ===")
    from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, extract_context_feature

    target_with_hidden = ModelWithHiddenStates(target_mlx, draft_mlx.target_layer_ids)
    model_cache = target_mlx.make_cache()

    output_mlx = target_with_hidden(input_ids_mlx, cache=model_cache)
    first_token_mlx = mx.argmax(output_mlx.logits[0, -1, :])

    print(f"First token (mlx): {tokenizer_mlx.decode([first_token_mlx.item()])} (id={first_token_mlx.item()})")

    # Check if first tokens match
    if first_token_torch[0].item() == first_token_mlx.item():
        print("✓ First tokens match!")
    else:
        print(f"✗ First tokens differ: torch={first_token_torch[0].item()}, mlx={first_token_mlx.item()}")

    # Extract target_hidden from MLX
    target_hidden_mlx = extract_context_feature(
        target_with_hidden.hidden_states,
        draft_mlx.target_layer_ids,
    )[:, -1:, :]  # Take last position only

    print(f"target_hidden shape: {target_hidden_mlx.shape}")

    # Compare target_hidden
    target_hidden_torch_np = target_hidden_torch.cpu().numpy()
    target_hidden_mlx_np = target_hidden_mlx

    correlation = np.corrcoef(
        target_hidden_torch_np.flatten(),
        target_hidden_mlx_np.flatten()
    )[0, 1]

    print(f"target_hidden correlation: {correlation:.4f}")
    if correlation > 0.95:
        print("✓ target_hidden matches!")
    else:
        print("✗ target_hidden differs!")

if __name__ == "__main__":
    test_single_iteration_comparison()
