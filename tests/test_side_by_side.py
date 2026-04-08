#!/usr/bin/env python3
"""Side-by-side comparison of MLX and PyTorch reference implementations."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import block_diffusion_generate_step


def test_mlx_implementation():
    """Test MLX implementation."""
    print("\n" + "=" * 60)
    print("MLX Implementation")
    print("=" * 60)

    # Load models
    print("Loading models...")
    model, tokenizer = load("Qwen/Qwen3.5-4B")
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")

    # Test prompt
    prompt = "What is the capital of France?"
    max_tokens = 20

    print(f"\nPrompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print("\nGenerating...")

    # Generate
    tokens = []
    debug_info = []

    for token_id, logprobs, from_draft in block_diffusion_generate_step(
        prompt=prompt,
        model=model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    ):
        tokens.append(token_id)
        text = tokenizer.decode(tokens)
        draft_mark = " [D]" if from_draft else " [T]"
        debug_info.append((token_id, tokenizer.decode([token_id]), from_draft))
        print(f"\r{text}{draft_mark}", end="", flush=True)

    print("\n\nGeneration complete!")
    print(f"Total tokens: {len(tokens)}")
    print(f"Final text: {tokenizer.decode(tokens)}")

    # Show breakdown
    print("\nToken breakdown:")
    for i, (tid, ttext, from_draft) in enumerate(debug_info):
        source = "DRAFT" if from_draft else "TARGET"
        print(f"  {i+1}. [{tid:5d}] {ttext:20s} ({source})")

    return tokens, debug_info


def test_pytorch_reference():
    """Test PyTorch reference implementation."""
    print("\n" + "=" * 60)
    print("PyTorch Reference Implementation")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import sys
        sys.path.insert(0, '/Users/ali/Projects/dflash')
        from dflash.model import DFlashDraftModel, extract_context_feature
        from dflash.benchmark import _dflash_generate

        # Load models
        print("Loading models...")
        model_path = "Qwen/Qwen3.5-4B"
        draft_path = "z-lab/Qwen3.5-4B-DFlash"

        target = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        draft = DFlashDraftModel.from_pretrained(
            draft_path,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Test prompt
        prompt = "What is the capital of France?"
        max_tokens = 20

        print(f"\nPrompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        print("\nGenerating...")

        # Prepare input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Generate with DFlash
        result = _dflash_generate(
            model=draft,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft.mask_token_id,
            max_new_tokens=max_tokens,
            block_size=draft.block_size,
            stop_token_ids=None,
            temperature=0.0,
        )

        # Decode output
        output_text = tokenizer.decode(result.output_ids[0], skip_special_tokens=True)
        print(f"\n{output_text}")

        print("\nGeneration complete!")
        print(f"Total tokens: {result.num_output_tokens}")
        print(f"Acceptance lengths: {result.acceptance_lengths[:5]}...")
        print(f"Average acceptance: {sum(result.acceptance_lengths) / len(result.acceptance_lengths):.2f}")

        return result.output_ids[0].tolist(), result

    except Exception as e:
        print(f"\nError running PyTorch reference: {e}")
        print("Skipping reference comparison")
        return None, None


def compare_outputs(mlx_tokens, ref_tokens, mlx_debug, ref_result):
    """Compare outputs from both implementations."""
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    if ref_tokens is None:
        print("Reference implementation not available for comparison")
        return

    print(f"MLX tokens: {len(mlx_tokens)}")
    print(f"Reference tokens: {len(ref_tokens)}")

    # Compare first N tokens
    min_len = min(len(mlx_tokens), len(ref_tokens))
    matches = sum(1 for i in range(min_len) if mlx_tokens[i] == ref_tokens[i])

    print(f"\nToken matches (first {min_len}): {matches}/{min_len} ({100*matches/min_len:.1f}%)")

    # Show differences
    if matches < min_len:
        print("\nToken differences:")
        for i in range(min_len):
            if mlx_tokens[i] != ref_tokens[i]:
                print(f"  Position {i}: MLX={mlx_tokens[i]}, Ref={ref_tokens[i]}")

    # Compare acceptance patterns if available
    if ref_result and hasattr(ref_result, 'acceptance_lengths'):
        mlx_acceptance = [1 for _, _, from_draft in mlx_debug if not from_draft]
        print(f"\nMLX target tokens: {len(mlx_acceptance)}")
        print(f"Reference acceptance lengths: {len(ref_result.acceptance_lengths)}")


def main():
    """Run side-by-side comparison."""
    print("=" * 70)
    print("MLX vs PyTorch Reference - Side-by-Side Comparison")
    print("=" * 70)

    # Run MLX implementation
    mlx_tokens, mlx_debug = test_mlx_implementation()

    # Run PyTorch reference
    ref_tokens, ref_result = test_pytorch_reference()

    # Compare
    compare_outputs(mlx_tokens, ref_tokens, mlx_debug, ref_result)

    print("\n" + "=" * 70)
    print("Comparison complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
