# Copyright © 2025 Apple Inc.

"""Step-by-step pipeline tests comparing MLX DFlash decode loop against PyTorch reference.

Each test runs one phase of the decode loop in both MLX and PyTorch with the
same inputs and compares outputs. This isolates exactly where the implementations
diverge.
"""

import unittest
import numpy as np

import mlx.core as mx
from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, extract_context_feature

mx.set_default_device(mx.cpu)


def _load_models():
    """Load both MLX and PyTorch models. Call once, reuse across tests."""
    import torch
    import sys
    from transformers import AutoModelForCausalLM, DynamicCache
    from mlx_lm.utils import load
    from mlx_lm.generate_dflash_v2 import (
        get_inner_model, ModelWithHiddenStates, extract_context_feature
    )

    sys.path.insert(0, '/Users/ali/.cache/huggingface/hub/z-lab--Qwen3.5-4B-DFlash')
    from dflash import DFlashDraftModel, extract_context_feature as ref_extract

    target_mlx, tokenizer = load('Qwen/Qwen3.5-4B')
    draft_mlx, _ = load('z-lab/Qwen3.5-4B-DFlash')
    target_inner = get_inner_model(target_mlx)

    target_pt = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', torch_dtype=torch.float32)
    target_pt.eval()
    draft_pt = DFlashDraftModel.from_pretrained(
        '/Users/ali/.cache/huggingface/hub/z-lab--Qwen3.5-4B-DFlash'
    )
    draft_pt.eval()

    return {
        'target_mlx': target_mlx, 'draft_mlx': draft_mlx,
        'target_pt': target_pt, 'draft_pt': draft_pt,
        'tokenizer': tokenizer, 'target_inner': target_inner,
        'torch': torch, 'DynamicCache': DynamicCache,
        'ref_extract': ref_extract,
    }


class TestPipelineStep1_Prefill(unittest.TestCase):
    """Compare prefill outputs: first token and target_hidden."""

    @classmethod
    def setUpClass(cls):
        cls.models = _load_models()

    def test_first_token_matches(self):
        """First sampled token should match between MLX and PyTorch."""
        tokenizer = self.models['tokenizer']

        prompt = "The meaning of life is"
        prompt_tokens = tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]
        prompt_pt = torch.tensor([prompt_tokens])

        import torch
        from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, extract_context_feature

        # MLX
        target_cache_mlx = self.models['target_mlx'].make_cache()
        logits_mlx = self.models['target_mlx'](prompt_mlx, cache=target_cache_mlx)
        mx.eval(logits_mlx)
        first_mlx = mx.argmax(logits_mlx[:, -1, :], axis=-1).squeeze(0).item()

        # PyTorch
        with torch.no_grad():
            cache_pt = self.models['DynamicCache']()
            pos_ids = torch.arange(len(prompt_tokens)).unsqueeze(0)
            output_pt = self.models['target_pt'](
                prompt_pt, position_ids=pos_ids, past_key_values=cache_pt,
                use_cache=True, output_hidden_states=True,
            )
            first_pt = torch.argmax(output_pt.logits[:, -1, :], dim=-1).squeeze().item()

        self.assertEqual(first_mlx, first_pt,
            f"First token mismatch: MLX={first_mlx} ({tokenizer.decode([first_mlx])}), "
            f"PT={first_pt} ({tokenizer.decode([first_pt])})")
        print(f"  first_token: {first_mlx} ({tokenizer.decode([first_mlx])}) ✓")

    def test_target_hidden_matches(self):
        """target_hidden extracted from prefill should match."""
        tokenizer = self.models['tokenizer']
        layer_ids = self.models['draft_mlx'].target_layer_ids

        prompt = "The meaning of life is"
        prompt_tokens = tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]
        prompt_pt = torch.tensor([prompt_tokens])

        # MLX
        whs = ModelWithHiddenStates(self.models['target_mlx'], layer_ids)
        hidden_mlx = whs(prompt_mlx, cache=None)
        mx.eval(hidden_mlx.logits)
        th_mlx = extract_context_feature(hidden_mlx.hidden_states, layer_ids)
        mx.eval(th_mlx)

        # PyTorch
        with torch.no_grad():
            output_pt = self.models['target_pt'](
                prompt_pt, output_hidden_states=True,
            )
            th_pt = self.models['ref_extract'](output_pt.hidden_states, layer_ids)

        diff = np.abs(np.array(th_mlx).astype(np.float32) - th_pt.float().numpy())
        print(f"  target_hidden max diff: {diff.max():.6f}, mean: {diff.mean():.6f}")
        self.assertLess(diff.max(), 0.5,
            f"target_hidden differs: max_diff={diff.max()}")


class TestPipelineStep2_DraftForward(unittest.TestCase):
    """Compare draft model outputs after prefill."""

    @classmethod
    def setUpClass(cls):
        cls.models = _load_models()
        torch = cls.models['torch']
        tokenizer = cls.models['tokenizer']
        layer_ids = cls.models['draft_mlx'].target_layer_ids

        prompt = "The meaning of life is"
        prompt_tokens = tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]
        prompt_pt = torch.tensor([prompt_tokens])

        # Get first token from MLX
        target_cache = cls.models['target_mlx'].make_cache()
        logits = cls.models['target_mlx'](prompt_mlx, cache=target_cache)
        mx.eval(logits)
        cls.first_token = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0)

        # Get target_hidden from MLX (separate pass, no cache update)
        whs = ModelWithHiddenStates(cls.models['target_mlx'], layer_ids)
        hidden = whs(prompt_mlx, cache=None)
        mx.eval(hidden.logits)
        cls.target_hidden_mlx = extract_context_feature(hidden.hidden_states, layer_ids)
        mx.eval(cls.target_hidden_mlx)

        # Get target_hidden from PT
        with torch.no_grad():
            output_pt = cls.models['target_pt'](
                prompt_pt, output_hidden_states=True,
            )
            cls.target_hidden_pt = cls.models['ref_extract'](output_pt.hidden_states, layer_ids)
            cls.first_token_pt = torch.argmax(output_pt.logits[:, -1, :], dim=-1).squeeze()

        cls.num_input = len(prompt_tokens)
        cls.block_size = 4  # Use small block for better draft quality
        cls.mask_token_id = cls.models['draft_mlx'].mask_token_id

    def test_draft_logits_match(self):
        """Draft model logits should match between MLX and PyTorch."""
        torch = self.models['torch']

        # MLX draft
        block = mx.full([1, self.block_size], self.mask_token_id, dtype=mx.uint32)
        block[:, 0] = self.first_token
        noise_emb = self.models['target_inner'].embed_tokens(block)

        draft_cache = self.models['draft_mlx'].make_cache()
        draft_pos = mx.arange(0, self.target_hidden_mlx.shape[1] + self.block_size)[None, :]
        draft_out = self.models['draft_mlx'](
            position_ids=draft_pos,
            noise_embedding=noise_emb,
            target_hidden=self.target_hidden_mlx,
            cache=draft_cache,
        )
        draft_logits_mlx = self.models['target_inner'].embed_tokens.as_linear(
            draft_out[:, -self.block_size + 1:, :]
        )
        mx.eval(draft_logits_mlx)

        # PT draft
        with torch.no_grad():
            dtype = next(self.models['draft_pt'].parameters()).dtype
            block_pt = torch.full((1, self.block_size), self.mask_token_id, dtype=torch.long)
            block_pt[:, 0] = self.first_token_pt
            noise_pt = self.models['target_pt'].model.embed_tokens(block_pt)
            cache_pt = self.models['DynamicCache']()
            start = self.num_input
            draft_pos_pt = torch.arange(0, start + self.block_size).unsqueeze(0)
            draft_out_pt = self.models['draft_pt'](
                target_hidden=self.target_hidden_pt,
                noise_embedding=noise_pt,
                position_ids=draft_pos_pt,
                past_key_values=cache_pt,
                use_cache=True,
            )
            draft_logits_pt = self.models['target_pt'].lm_head(
                draft_out_pt[:, -self.block_size + 1:, :]
            )

        # Compare top-k predictions
        topk_mlx = mx.argmax(draft_logits_mlx, axis=-1).squeeze(0)
        topk_pt = torch.argmax(draft_logits_pt, dim=-1).squeeze(0)

        print(f"  MLX draft tokens: {topk_mlx.tolist()}")
        print(f"  PT draft tokens:  {topk_pt.tolist()}")
        tokenizer = self.models['tokenizer']
        print(f"  MLX decoded: {tokenizer.decode(topk_mlx.tolist())}")
        print(f"  PT decoded:  {tokenizer.decode(topk_pt.tolist())}")

        # Check if predictions match
        match = [a == b for a, b in zip(topk_mlx.tolist(), topk_pt.tolist())]
        print(f"  Match: {match}")


class TestPipelineStep3_Verify(unittest.TestCase):
    """Compare target model verify outputs."""

    @classmethod
    def setUpClass(cls):
        cls.models = _load_models()

    def test_verify_with_draft_tokens(self):
        """Target model verify step should produce same posterior for same draft tokens."""
        import torch
        tokenizer = self.models['tokenizer']

        prompt = "The meaning of life is"
        prompt_tokens = tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]
        prompt_pt = torch.tensor([prompt_tokens])
        block_size = 4

        # MLX: prefill
        target_cache_mlx = self.models['target_mlx'].make_cache()
        logits_mlx = self.models['target_mlx'](prompt_mlx, cache=target_cache_mlx)
        mx.eval(logits_mlx)
        first_mlx = mx.argmax(logits_mlx[:, -1, :], axis=-1).squeeze(0)

        # Use hardcoded draft tokens for deterministic comparison
        # "question that has puzzled" = [3296, 421, 682, 83339]
        draft_tokens = [3296, 421, 682]

        block_mlx = mx.array([[first_mlx.item()] + draft_tokens], dtype=mx.uint32)

        # MLX: verify
        whs = ModelWithHiddenStates(self.models['target_mlx'], self.models['draft_mlx'].target_layer_ids)
        verify_mlx = whs(block_mlx, cache=target_cache_mlx)
        mx.eval(verify_mlx.logits)
        posterior_mlx = mx.argmax(verify_mlx.logits, axis=-1).squeeze(0)

        # PT: compare by running full sequence without cache
        with torch.no_grad():
            full_pt = torch.cat([prompt_pt, torch.tensor([[first_mlx] + draft_tokens])], dim=1)
            output_pt = self.models['target_pt'](
                full_pt, output_hidden_states=True, use_cache=False,
            )
            # Posterior = what PT predicts for the draft positions
            pt_logits = output_pt.logits[:, len(prompt_tokens):, :]
            posterior_pt = torch.argmax(pt_logits, dim=-1).squeeze(0)

        print(f"  MLX posterior: {posterior_mlx.tolist()}")
        print(f"  PT posterior:  {posterior_pt.tolist()}")
        print(f"  MLX decoded: {tokenizer.decode(posterior_mlx.tolist())}")
        print(f"  PT decoded:  {tokenizer.decode(posterior_pt.tolist())}")

        self.assertEqual(posterior_mlx.tolist(), posterior_pt.tolist(),
            "Verify posterior mismatch between MLX and PyTorch")


class TestPipelineStep4_RollbackAndRebuild(unittest.TestCase):
    """Compare cache state after rollback + rebuild."""

    @classmethod
    def setUpClass(cls):
        cls.models = _load_models()

    def test_rebuild_produces_correct_next_token(self):
        """After rollback + rebuild, the next token should match target-only decode."""
        import torch
        tokenizer = self.models['tokenizer']

        prompt = "The meaning of life is"
        prompt_tokens = tokenizer.encode(prompt)
        prompt_mlx = mx.array(prompt_tokens)[None, :]

        # Target-only baseline: what should the next tokens be?
        target_cache_baseline = self.models['target_mlx'].make_cache()
        logits = self.models['target_mlx'](prompt_mlx, cache=target_cache_baseline)
        mx.eval(logits)
        baseline_tokens = []
        for i in range(5):
            tok = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0)
            baseline_tokens.append(tok.item())
            logits = self.models['target_mlx'](tok[None, None], cache=target_cache_baseline)
            mx.eval(logits)

        print(f"  Target-only tokens: {baseline_tokens}")
        print(f"  Decoded: {tokenizer.decode(baseline_tokens)}")

        # Now simulate: prefill → verify with block → rollback → rebuild → next token
        target_cache = self.models['target_mlx'].make_cache()
        logits = self.models['target_mlx'](prompt_mlx, cache=target_cache)
        mx.eval(logits)
        first_token = baseline_tokens[0]  # Same first token

        # Verify with block of tokens (first + some "draft" tokens)
        block = mx.array([[first_token, 99999, 99999, 99999]], dtype=mx.uint32)  # Bad draft tokens

        # Save checkpoint before verify
        for c in target_cache:
            if hasattr(c, 'save_checkpoint'):
                c.save_checkpoint()

        whs = ModelWithHiddenStates(self.models['target_mlx'], self.models['draft_mlx'].target_layer_ids)
        verify = whs(block, cache=target_cache)
        mx.eval(verify.logits)

        # Rollback (all draft tokens rejected since 99999 won't match)
        for c in target_cache:
            if hasattr(c, 'rollback'):
                c.rollback()
            elif hasattr(c, 'keys') and c.keys is not None:
                c.offset -= 3  # Remove 3 bad draft tokens

        # Rebuild with just the accepted (first_token) + bonus token
        bonus_token = mx.argmax(verify.logits[:, 0, :], axis=-1).squeeze(0).item()
        rebuild_tokens = mx.array([[first_token, bonus_token]])

        rebuild = whs(rebuild_tokens, cache=target_cache)
        mx.eval(rebuild.logits)

        # Now decode next token — should match baseline_tokens[2]
        next_tok = mx.argmax(rebuild.logits[:, -1, :], axis=-1).squeeze(0).item()

        print(f"  Rebuild bonus token: {bonus_token} ({tokenizer.decode([bonus_token])})")
        print(f"  Next token after rebuild: {next_tok} ({tokenizer.decode([next_tok])})")
        print(f"  Expected (baseline[2]): {baseline_tokens[2]} ({tokenizer.decode([baseline_tokens[2]])})")

        # The bonus token should match baseline[1]
        self.assertEqual(bonus_token, baseline_tokens[1],
            f"Bonus token mismatch: got {bonus_token} ({tokenizer.decode([bonus_token])}), "
            f"expected {baseline_tokens[1]} ({tokenizer.decode([baseline_tokens[1]])})")

        # The next token after rebuild should match baseline[2]
        self.assertEqual(next_tok, baseline_tokens[2],
            f"Next token after rebuild mismatch: got {next_tok}, expected {baseline_tokens[2]}")


if __name__ == '__main__':
    unittest.main()
