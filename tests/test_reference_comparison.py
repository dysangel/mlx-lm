#!/usr/bin/env python3
"""Compare MLX DFlash implementation with PyTorch reference."""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import (
    ModelWithHiddenStates,
    extract_context_feature,
)


def compare_hidden_states_extraction():
    """Compare hidden states extraction between MLX and reference."""
    print("\n=== Comparing Hidden States Extraction ===")

    # Load MLX model
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Setup
    prompt = "Hello world"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Extract hidden states
    target_layer_ids = [1, 8, 15, 22, 29]
    wrapped_model = ModelWithHiddenStates(model, target_layer_ids)
    output = wrapped_model(prompt_tokens, cache=None)
    mx.eval(output.logits)

    # Extract target_hidden
    target_hidden = extract_context_feature(
        wrapped_model.hidden_states,
        target_layer_ids,
    )
    mx.eval(target_hidden)

    print(f"  MLX target_hidden shape: {target_hidden.shape}")
    print(f"  MLX target_hidden stats: min={target_hidden.min().item():.4f}, max={target_hidden.max().item():.4f}, mean={target_hidden.mean().item():.4f}")

    # TODO: Run reference PyTorch implementation and compare
    print("  ⚠ Reference comparison not yet implemented")
    print("  Need to:")
    print("    1. Import reference dflash model")
    print("    2. Run with same prompt")
    print("    3. Compare hidden states shapes and values")

    return target_hidden


def compare_draft_model_output():
    """Compare draft model output between MLX and reference."""
    print("\n=== Comparing Draft Model Output ===")

    # Load MLX models
    model, tokenizer = load("Qwen/Qwen3.5-4B")
    draft_model, _ = load("z-lab/Qwen3.5-4B-DFlash")

    # Setup
    prompt = "The answer is"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)
    output = target_model_with_hidden(prompt_tokens, cache=None)
    mx.eval(output.logits)

    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )
    mx.eval(target_hidden)

    # Get draft output
    from mlx_lm.generate_dflash_v2 import get_inner_model
    inner_model = get_inner_model(model)

    block_size = 16
    ctx_len = target_hidden.shape[1]
    last_token = mx.argmax(output.logits[:, -1, :], axis=-1).squeeze(0).item()

    prev_token = mx.array([[last_token]])
    noise_tokens = mx.zeros([1, block_size - 1], dtype=mx.uint32)
    draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)
    noise_embedding = inner_model.embed_tokens(draft_input)
    mx.eval(noise_embedding)

    position_ids = mx.arange(ctx_len, ctx_len + block_size)[None, :]
    draft_cache = draft_model.make_cache()

    draft_output = draft_model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=draft_cache,
    )
    mx.eval(draft_output)

    print(f"  MLX draft_output shape: {draft_output.shape}")
    print(f"  MLX draft_output stats: min={draft_output.min().item():.4f}, max={draft_output.max().item():.4f}, mean={draft_output.mean().item():.4f}")

    # Get draft tokens
    if hasattr(model, 'lm_head'):
        draft_logits = model.lm_head(draft_output)
    else:
        draft_logits = inner_model.embed_tokens.as_linear(draft_output)
    mx.eval(draft_logits)

    draft_tokens = mx.argmax(draft_logits[:, -block_size + 1:, :], axis=-1).squeeze(0)
    print(f"  MLX draft tokens: {draft_tokens.tolist()[:10]}...")

    # TODO: Run reference and compare
    print("  ⚠ Reference comparison not yet implemented")

    return draft_output, draft_tokens


def compare_acceptance_patterns():
    """Compare acceptance patterns between MLX and reference."""
    print("\n=== Comparing Acceptance Patterns ===")

    # Load MLX model
    model, tokenizer = load("Qwen/Qwen3.5-4B")

    # Simple test: generate tokens and check if they make sense
    prompt = "What is 2+2?"
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]

    # Generate with MLX
    cache = model.make_cache()
    output = model(prompt_tokens, cache=cache)
    mx.eval(output)

    tokens = []
    for i in range(10):
        prev_token = mx.array([[tokens[-1] if tokens else mx.argmax(output[:, -1, :], axis=-1).squeeze(0).item()]])
        output = model(prev_token, cache=cache)
        mx.eval(output)
        token = mx.argmax(output[:, -1, :], axis=-1).squeeze(0).item()
        tokens.append(token)

    text = tokenizer.decode(tokens)
    print(f"  MLX generated: {text}")

    # TODO: Compare with reference
    print("  ⚠ Reference comparison not yet implemented")


def setup_reference_tests():
    """Guide for setting up reference comparison tests."""
    print("\n=== Setting Up Reference Comparison ===")

    print("""
To compare with reference PyTorch implementation:

1. Install PyTorch dependencies:
   pip install torch transformers

2. Create test file that imports reference:
   import sys
   sys.path.insert(0, '/Users/ali/Projects/dflash')
   from dflash.model import DFlashModel

3. Run same tests on reference:
   - Load same models
   - Use same prompt
   - Compare hidden states
   - Compare draft outputs
   - Compare acceptance rates

4. Add assertions to ensure outputs match:
   assert torch.allclose(ref_hidden, mlx_hidden, atol=1e-3)

Example reference test structure:
```python
def test_reference_hidden_states():
    # Load reference model
    ref_model = DFlashModel.from_pretrained("z-lab/Qwen3.5-4B-DFlash")

    # Run with same prompt
    prompt = "Hello world"
    ref_output = ref_model.generate(prompt, max_tokens=10)

    # Compare with MLX output
    assert ref_output == mlx_output
```
    """)


def create_reference_test_template():
    """Create template for reference comparison tests."""
    print("\n=== Creating Reference Test Template ===")

    template = '''#!/usr/bin/env python3
"""Reference PyTorch DFlash tests for comparison."""

import sys
sys.path.insert(0, '/Users/ali/Projects/dflash')

import torch
from dflash.model import DFlashModel
from transformers import AutoTokenizer


def test_reference_draft_generation():
    """Test draft model generation with reference."""
    print("\\n=== Testing Reference Draft Generation ===")

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
    print("\\n=== Testing Reference Hidden States ===")

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
    print("\\n=== Comparing MLX vs Reference ===")

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
'''

    with open('/Users/ali/Projects/mlx-lm/tests/test_reference_pytorch.py', 'w') as f:
        f.write(template)

    print("  Created template: tests/test_reference_pytorch.py")
    print("  Run with: python tests/test_reference_pytorch.py")


def run_comparison():
    """Run comparison tests."""
    print("=" * 60)
    print("MLX vs Reference Comparison")
    print("=" * 60)

    target_hidden = compare_hidden_states_extraction()
    draft_output, draft_tokens = compare_draft_model_output()
    compare_acceptance_patterns()
    setup_reference_tests()
    create_reference_test_template()

    print("\n" + "=" * 60)
    print("Comparison complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install PyTorch: pip install torch transformers")
    print("2. Run reference tests: python tests/test_reference_pytorch.py")
    print("3. Compare outputs and add assertions")


if __name__ == "__main__":
    run_comparison()
