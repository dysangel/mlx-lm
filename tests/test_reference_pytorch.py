#!/usr/bin/env python3
"""Reference PyTorch DFlash tests for comparison."""

import sys
sys.path.insert(0, '/Users/ali/Projects/dflash')

import torch
from dflash.model import DFlashModel
from transformers import AutoTokenizer


def test_reference_draft_generation():
    """Test draft model generation with reference."""
    print("\n=== Testing Reference Draft Generation ===")

    # Load models
    model_path = "z-lab/Qwen3.5-4B-DFlash"
    model = DFlashModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

    # Setup
    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=16,
        )

    # Decode
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Reference generated: {generated}")

    return output


def test_reference_hidden_states():
    """Test hidden states extraction with reference."""
    print("\n=== Testing Reference Hidden States ===")

    # Load models
    model_path = "z-lab/Qwen3.5-4B-DFlash"
    model = DFlashModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

    # Setup
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Extract hidden states
    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)

    # Get target hidden states
    target_layer_ids = [1, 8, 15, 22, 29]
    hidden_states = outputs.hidden_states

    # Extract from target layers
    selected_states = [hidden_states[i + 1] for i in target_layer_ids]  # +1 for embedding
    target_hidden = torch.cat(selected_states, dim=-1)

    print(f"  Reference target_hidden shape: {target_hidden.shape}")
    print(f"  Reference target_hidden stats: min={target_hidden.min():.4f}, max={target_hidden.max():.4f}, mean={target_hidden.mean():.4f}")

    return target_hidden


def compare_mlx_reference():
    """Compare MLX and reference outputs."""
    print("\n=== Comparing MLX vs Reference ===")

    # Run reference
    ref_hidden = test_reference_hidden_states()

    # Run MLX (import from our tests)
    import sys
    sys.path.insert(0, '.')
    from tests.test_reference_comparison import compare_hidden_states_extraction

    mlx_hidden = compare_hidden_states_extraction()

    # Convert to same format for comparison
    mlx_torch = torch.from_numpy(mlx_hidden.numpy())

    # Compare
    if torch.allclose(ref_hidden, mlx_torch, atol=1e-3):
        print("  ✓ Hidden states match!")
    else:
        print("  ✗ Hidden states differ")
        print(f"    Max diff: {(ref_hidden - mlx_torch).abs().max():.4f}")


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer

    test_reference_draft_generation()
    test_reference_hidden_states()
    # compare_mlx_reference()  # Uncomment to compare
