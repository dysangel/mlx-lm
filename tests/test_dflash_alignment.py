# Copyright © 2025 Apple Inc.

"""Tests comparing MLX DFlash generate loop against reference PyTorch step by step.

Each test verifies one step of the speculative decoding loop produces matching
outputs between the MLX and PyTorch implementations, using the same random weights.
"""

import unittest
import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)


class TestDraftModelAlignment(unittest.TestCase):
    """Compare draft model forward pass between MLX and PyTorch."""

    @classmethod
    def setUpClass(cls):
        """Load both models and create shared test state."""
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not installed")

        from mlx_lm.utils import load
        from mlx_lm.generate_dflash_v2 import get_inner_model, ModelWithHiddenStates, extract_context_feature

        # Load MLX models
        cls.target_model_mlx, cls.tokenizer = load('Qwen/Qwen3.5-4B')
        cls.draft_model_mlx, _ = load('z-lab/Qwen3.5-4B-DFlash')
        cls.target_inner = get_inner_model(cls.target_model_mlx)

        # Load PyTorch reference
        from transformers import AutoModelForCausalLM
        import sys
        sys.path.insert(0, '/Users/ali/.cache/huggingface/hub/z-lab--Qwen3.5-4B-DFlash')
        from dflash import DFlashDraftModel
        cls.draft_model_pt = DFlashDraftModel.from_pretrained(
            '/Users/ali/.cache/huggingface/hub/z-lab--Qwen3.5-4B-DFlash'
        )
        cls.draft_model_pt.eval()
        cls.target_model_pt = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen3.5-4B', torch_dtype=torch.float32
        )
        cls.target_model_pt.eval()

    def test_draft_model_single_forward(self):
        """Draft model forward pass should produce matching outputs."""
        import torch

        np.random.seed(42)

        B, L, D = 1, 4, 2560  # batch, block_size, hidden_size
        ctx_len = 5
        num_target_layers = len(self.draft_model_mlx.target_layer_ids)
        target_hidden_dim = num_target_layers * D

        noise_np = np.random.randn(B, L, D).astype(np.float32)
        target_hidden_np = np.random.randn(B, ctx_len, target_hidden_dim).astype(np.float32)

        # MLX forward
        noise_mlx = mx.array(noise_np)
        target_hidden_mlx = mx.array(target_hidden_np)
        position_ids_mlx = mx.arange(0, ctx_len + L)[None, :]

        draft_cache_mlx = self.draft_model_mlx.make_cache()
        output_mlx = self.draft_model_mlx(
            position_ids=position_ids_mlx,
            noise_embedding=noise_mlx,
            target_hidden=target_hidden_mlx,
            cache=draft_cache_mlx,
        )
        mx.eval(output_mlx)

        # PyTorch forward
        with torch.no_grad():
            dtype = next(self.draft_model_pt.parameters()).dtype
            noise_pt = torch.tensor(noise_np).to(dtype)
            target_hidden_pt = torch.tensor(target_hidden_np).to(dtype)
            position_ids_pt = torch.arange(0, ctx_len + L).unsqueeze(0)

            from transformers import DynamicCache
            cache_pt = DynamicCache()
            output_pt = self.draft_model_pt(
                target_hidden=target_hidden_pt,
                noise_embedding=noise_pt,
                position_ids=position_ids_pt,
                past_key_values=cache_pt,
                use_cache=True,
            )

        # Compare
        np.testing.assert_allclose(
            np.array(output_mlx),
            output_pt.float().numpy(),
            atol=0.3, rtol=0.1,
            err_msg="Draft model forward outputs differ between MLX and PyTorch"
        )

    def test_prefill_alignment(self):
        """Prefill should produce matching first token and target_hidden."""
        import torch

        prompt = "The meaning of life is"
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]
        prompt_pt = torch.tensor([prompt_tokens])

        # MLX prefill
        from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, extract_context_feature

        target_cache_mlx = self.target_model_mlx.make_cache()
        prefill_logits_mlx = self.target_model_mlx(prompt_mlx, cache=target_cache_mlx)
        mx.eval(prefill_logits_mlx)
        first_token_mlx = mx.argmax(prefill_logits_mlx[:, -1, :], axis=-1).squeeze(0)

        # Extract target_hidden via separate pass
        target_model_with_hidden = ModelWithHiddenStates(
            self.target_model_mlx, self.draft_model_mlx.target_layer_ids
        )
        hidden_output = target_model_with_hidden(prompt_mlx, cache=None)
        mx.eval(hidden_output.logits)
        target_hidden_mlx = extract_context_feature(
            hidden_output.hidden_states, self.draft_model_mlx.target_layer_ids
        )
        mx.eval(target_hidden_mlx)

        # PyTorch prefill
        with torch.no_grad():
            from transformers import DynamicCache
            from dflash import extract_context_feature as ref_extract

            cache_pt = DynamicCache()
            position_ids_pt = torch.arange(len(prompt_tokens)).unsqueeze(0)
            output_pt = self.target_model_pt(
                prompt_pt,
                position_ids=position_ids_pt,
                past_key_values=cache_pt,
                use_cache=True,
                output_hidden_states=True,
            )
            first_token_pt = torch.argmax(output_pt.logits[:, -1, :], dim=-1).squeeze(0)
            target_hidden_pt = ref_extract(output_pt.hidden_states, self.draft_model_mlx.target_layer_ids)

        # Compare first token
        self.assertEqual(
            first_token_mlx.item(), first_token_pt.item(),
            "First token mismatch"
        )

        # Compare target_hidden
        np.testing.assert_allclose(
            np.array(target_hidden_mlx),
            target_hidden_pt.numpy(),
            atol=1e-2, rtol=1e-2,
            err_msg="target_hidden mismatch after prefill"
        )

    def test_draft_with_real_prefill(self):
        """Draft model with real prefill data should produce sensible predictions."""
        import torch

        prompt = "The meaning of life is"
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]
        block_size = self.draft_model_mlx.block_size
        mask_token_id = self.draft_model_mlx.mask_token_id

        # MLX: prefill
        from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, extract_context_feature

        target_cache_mlx = self.target_model_mlx.make_cache()
        prefill_logits = self.target_model_mlx(prompt_mlx, cache=target_cache_mlx)
        mx.eval(prefill_logits)
        first_token = mx.argmax(prefill_logits[:, -1, :], axis=-1).squeeze(0)

        # Extract target_hidden
        target_model_with_hidden = ModelWithHiddenStates(
            self.target_model_mlx, self.draft_model_mlx.target_layer_ids
        )
        hidden_output = target_model_with_hidden(prompt_mlx, cache=None)
        mx.eval(hidden_output.logits)
        target_hidden_mlx = extract_context_feature(
            hidden_output.hidden_states, self.draft_model_mlx.target_layer_ids
        )
        mx.eval(target_hidden_mlx)

        # MLX: draft forward
        num_input = len(prompt_tokens)
        start = num_input
        draft_cache_mlx = self.draft_model_mlx.make_cache()

        block_output_ids = mx.full([1, block_size], mask_token_id, dtype=mx.uint32)
        block_output_ids[:, 0] = first_token
        noise_embedding = self.target_inner.embed_tokens(block_output_ids)

        ctx_len = target_hidden_mlx.shape[1]
        draft_position_ids = mx.arange(0, ctx_len + block_size)[None, :]

        draft_output_mlx = self.draft_model_mlx(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden_mlx,
            cache=draft_cache_mlx,
        )
        draft_logits_mlx = self.target_inner.embed_tokens.as_linear(
            draft_output_mlx[:, -block_size + 1:, :]
        )
        mx.eval(draft_logits_mlx)

        # PyTorch: same steps
        with torch.no_grad():
            from transformers import DynamicCache
            from dflash import extract_context_feature as ref_extract, sample

            prompt_pt = torch.tensor([prompt_tokens])
            cache_pt_target = DynamicCache()
            position_ids_pt = torch.arange(num_input).unsqueeze(0)
            output_pt = self.target_model_pt(
                prompt_pt,
                position_ids=position_ids_pt,
                past_key_values=cache_pt_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=True,
            )
            first_token_pt = sample(output_pt.logits, temperature=0.0)
            target_hidden_pt = ref_extract(output_pt.hidden_states, self.draft_model_mlx.target_layer_ids)

            # Draft
            cache_pt_draft = DynamicCache()
            block_pt = torch.full((1, block_size), mask_token_id, dtype=torch.long)
            block_pt[:, 0] = first_token_pt.squeeze()
            noise_emb_pt = self.target_model_pt.model.embed_tokens(block_pt)

            draft_output_pt = self.draft_model_pt(
                target_hidden=target_hidden_pt,
                noise_embedding=noise_emb_pt,
                position_ids=position_ids_pt[:, cache_pt_draft.get_seq_length(): start + block_size],
                past_key_values=cache_pt_draft,
                use_cache=True,
            )
            draft_logits_pt = self.target_model_pt.lm_head(draft_output_pt[:, -block_size + 1:, :])

        # Compare draft logits
        np.testing.assert_allclose(
            np.array(draft_logits_mlx),
            draft_logits_pt.numpy(),
            atol=1e-2, rtol=1e-2,
            err_msg="Draft logits mismatch with real prefill data"
        )

    def test_cache_crop_preserves_invariant(self):
        """After crop, the position ID invariant should hold for next iteration.

        Invariant: position_ids length == ctx_len + noise_len
        where ctx_len = target_hidden.shape[1], noise_len = block_size
        and position_ids span from cache_len to start + block_size
        so: start + block_size - cache_len == ctx_len + block_size
        so: start - cache_len == ctx_len
        """
        block_size = 4
        num_input = 5
        start = num_input  # First iteration start

        # First iteration: cache_len=0, ctx_len=5
        cache_len = 0
        ctx_len = num_input
        pos_len = start + block_size - cache_len
        self.assertEqual(pos_len, ctx_len + block_size, "First iteration invariant broken")

        # After first iter: acceptance=2, new_start=8
        acceptance = 2
        new_start = start + acceptance + 1  # = 8
        new_ctx_len = acceptance + 1  # = 3 (target_hidden replaced)
        new_cache_len = start  # cache cropped to old start = 5

        # Second iteration invariant
        pos_len = new_start + block_size - new_cache_len
        self.assertEqual(pos_len, new_ctx_len + block_size,
            f"Second iteration invariant broken: {pos_len} != {new_ctx_len + block_size}")


class TestAcceptanceComputationAlignment(unittest.TestCase):
    """Test acceptance computation matches reference."""

    def test_acceptance_matches_reference(self):
        """Our acceptance computation should match reference cumprod approach."""
        import torch

        # Test various patterns
        cases = [
            ([10, 20, 30], [10, 20, 30], "all match"),
            ([10, 20, 30], [99, 88, 77], "none match"),
            ([10, 20, 30], [10, 20, 99], "first 2 match"),
            ([10, 20, 30], [10, 99, 99], "first 1 match"),
            ([10], [10], "single match"),
            ([10], [99], "single no match"),
        ]

        for draft, target, desc in cases:
            with self.subTest(desc=desc):
                # Reference (PyTorch)
                draft_pt = torch.tensor([draft])
                target_pt = torch.tensor([target])
                ref_acceptance = (draft_pt == target_pt).cumprod(dim=1).sum(dim=1)[0].item()

                # MLX
                draft_mx = mx.array(draft)
                target_mx = mx.array(target)
                mlx_acceptance = int(
                    (mx.cumsum(draft_mx == target_mx) == mx.arange(1, len(target) + 1)).sum()
                )

                self.assertEqual(ref_acceptance, mlx_acceptance, f"{desc}: acceptance mismatch")


if __name__ == '__main__':
    unittest.main()
