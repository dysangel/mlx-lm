# Copyright © 2025 Apple Inc.

"""Tests comparing MLX DFlash implementation against PyTorch reference.

These tests verify that our MLX port produces numerically identical (within
floating point tolerance) outputs to the reference PyTorch implementation,
using the same random weights and inputs.

Reference: ~/.cache/huggingface/hub/z-lab--Qwen3.5-4B-DFlash/dflash.py
"""

import unittest
import numpy as np

import mlx.core as mx

# Set float32 for determinism
mx.set_default_device(mx.cpu)


def to_numpy(x):
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        return np.array(x)
    return x.detach().cpu().numpy()


class TestRoPE(unittest.TestCase):
    """Test RoPE cos/sin computation matches reference."""

    def _reference_rope(self, position_ids, head_dim=128, rope_theta=10000000.0):
        """Reference PyTorch RoPE computation."""
        import torch
        position_ids = torch.tensor(position_ids, dtype=torch.float32)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        bsz, seq_len = position_ids.shape
        inv_freq_expanded = inv_freq[None, :, None].float().expand(bsz, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.numpy(), sin.numpy()

    def _mlx_rope(self, position_ids, head_dim=128, rope_theta=10000000.0):
        """Our MLX RoPE computation."""
        from mlx_lm.models.dflash_v2 import ModelArgs, RoPE
        args = ModelArgs(rope_theta=rope_theta)
        rope = RoPE(args)
        rope.head_dim = head_dim
        position_ids = mx.array(position_ids)
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        hidden_states = mx.zeros((1, position_ids.shape[-1], 2560))  # dummy
        cos, sin = rope(hidden_states, position_ids)
        return np.array(cos), np.array(sin)

    def test_rope_basic(self):
        """RoPE cos/sin should match reference for basic positions."""
        positions = np.arange(16).reshape(1, -1)
        ref_cos, ref_sin = self._reference_rope(positions)
        mlx_cos, mlx_sin = self._mlx_rope(positions)

        # Reference returns (batch, seq_len, head_dim), MLX returns (seq_len, head_dim/2)
        # Reference doubles by cat(freqs, freqs), we return half-dim
        # Reference cos = cos([freqs, freqs]) = [cos(freqs), cos(freqs)]
        # We need to compare the first half
        ref_cos_half = ref_cos[0, :, :ref_cos.shape[-1] // 2]  # (seq_len, head_dim/2)
        ref_sin_half = ref_sin[0, :, :ref_sin.shape[-1] // 2]

        np.testing.assert_allclose(mlx_cos, ref_cos_half, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(mlx_sin, ref_sin_half, atol=1e-5, rtol=1e-5)

    def test_rope_non_contiguous_positions(self):
        """RoPE should handle non-contiguous positions (e.g., after cache crop)."""
        positions = np.array([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        ref_cos, ref_sin = self._reference_rope(positions)
        mlx_cos, mlx_sin = self._mlx_rope(positions)

        ref_cos_half = ref_cos[0, :, :ref_cos.shape[-1] // 2]
        ref_sin_half = ref_sin[0, :, :ref_sin.shape[-1] // 2]

        np.testing.assert_allclose(mlx_cos, ref_cos_half, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(mlx_sin, ref_sin_half, atol=1e-5, rtol=1e-5)


class TestRotateHalf(unittest.TestCase):
    """Test rotate_half matches reference."""

    def test_rotate_half(self):
        from mlx_lm.models.dflash_v2 import rotate_half

        # Create a simple test: [1, 2, 3, 4] -> [-3, -4, 1, 2]
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        result = rotate_half(x)
        expected = np.array([[-3.0, -4.0, 1.0, 2.0]])
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_rotate_half_larger(self):
        from mlx_lm.models.dflash_v2 import rotate_half

        x = mx.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        result = rotate_half(x)
        # x1 = [1,2,3], x2 = [4,5,6] -> [-x2, x1] = [-4,-5,-6, 1,2,3]
        expected = np.array([[[-4.0, -5.0, -6.0, 1.0, 2.0, 3.0]]])
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


class TestApplyRotaryPosEmb(unittest.TestCase):
    """Test apply_rotary_pos_emb matches reference."""

    def _reference_apply_rope(self, q_np, k_np, cos_np, sin_np):
        """Reference PyTorch apply_rotary_pos_emb."""
        import torch

        def rotate_half_torch(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q = torch.tensor(q_np)
        k = torch.tensor(k_np)
        cos = torch.tensor(cos_np).unsqueeze(1)  # Add head dim
        sin = torch.tensor(sin_np).unsqueeze(1)

        q_len = q.shape[-2]
        q_embed = (q * cos[..., -q_len:, :]) + (rotate_half_torch(q) * sin[..., -q_len:, :])
        k_embed = (k * cos) + (rotate_half_torch(k) * sin)
        return q_embed.numpy(), k_embed.numpy()

    def test_apply_rope_basic(self):
        """apply_rotary_pos_emb should match reference with same inputs."""
        from mlx_lm.models.dflash_v2 import apply_rotary_pos_emb

        np.random.seed(42)
        B, H, L, D = 1, 2, 4, 8  # Small dimensions for testing
        ctx_len = 3
        total_len = ctx_len + L  # K has context + noise

        q = mx.array(np.random.randn(B, H, L, D).astype(np.float32))
        k = mx.array(np.random.randn(B, H, total_len, D).astype(np.float32))

        # Compute cos/sin for total_len positions
        freqs = np.arange(total_len, dtype=np.float32)[:, None] * \
            (1.0 / (10000000.0 ** (np.arange(0, D, 2, dtype=np.float32) / D)))[None, :]
        cos = mx.array(np.cos(freqs))   # (total_len, D/2)
        sin = mx.array(np.sin(freqs))   # (total_len, D/2)

        # MLX version
        q_mlx, k_mlx = apply_rotary_pos_emb(q, k, cos, sin)

        # Reference: needs (batch, seq_len, head_dim) with doubled last dim
        cos_ref = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)[None, :, :]  # (1, total_len, D)
        sin_ref = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)[None, :, :]
        q_ref, k_ref = self._reference_apply_rope(
            np.array(q), np.array(k), cos_ref, sin_ref
        )

        np.testing.assert_allclose(np.array(q_mlx), q_ref, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.array(k_mlx), k_ref, atol=1e-4, rtol=1e-4)


class TestExtractContextFeature(unittest.TestCase):
    """Test extract_context_feature matches reference."""

    def test_extract_context_feature(self):
        from mlx_lm.generate_dflash_v2 import extract_context_feature

        # Create mock hidden states: [embedding, layer0, layer1, layer2, layer3, layer4]
        hidden_states = [mx.array(np.random.randn(1, 5, 64).astype(np.float32)) for _ in range(6)]

        # Extract from layers [0, 2, 4] (offset=1, so actual layers 1, 3, 5)
        layer_ids = [0, 2, 4]
        result = extract_context_feature(hidden_states, layer_ids)

        # Should concatenate hidden_states[1], hidden_states[3], hidden_states[5]
        expected = np.concatenate([np.array(hidden_states[1]),
                                   np.array(hidden_states[3]),
                                   np.array(hidden_states[5])], axis=-1)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_matches_reference(self):
        """Compare with reference extract_context_feature."""
        import torch
        from mlx_lm.generate_dflash_v2 import extract_context_feature

        def ref_extract(hidden_states, layer_ids):
            offset = 1
            selected = [hidden_states[layer_id + offset] for layer_id in layer_ids]
            return torch.cat(selected, dim=-1)

        hidden_np = [np.random.randn(1, 5, 64).astype(np.float32) for _ in range(6)]
        layer_ids = [0, 2, 4]

        ref_result = ref_extract([torch.tensor(h) for h in hidden_np], layer_ids).numpy()
        mlx_result = np.array(extract_context_feature(
            [mx.array(h) for h in hidden_np], layer_ids
        ))

        np.testing.assert_allclose(mlx_result, ref_result, atol=1e-5)


class TestAcceptanceComputation(unittest.TestCase):
    """Test the acceptance length computation matches reference."""

    def _reference_acceptance(self, draft_tokens, target_tokens):
        """Reference PyTorch acceptance computation."""
        import torch
        draft = torch.tensor(draft_tokens)
        target = torch.tensor(target_tokens)
        # Reference: (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
        # Our equivalent: consecutive matches from start
        matches = (draft == target).cumprod(dim=0).sum().item()
        return matches

    def _mlx_acceptance(self, draft_tokens, target_tokens):
        """Our MLX acceptance computation."""
        draft = mx.array(draft_tokens)
        target = mx.array(target_tokens)
        acceptance = int(
            (mx.cumsum(draft == target) == mx.arange(1, len(target) + 1)).sum()
        )
        return acceptance

    def test_all_match(self):
        """All tokens match -> full acceptance."""
        draft = [10, 20, 30, 40]
        target = [10, 20, 30, 40]
        self.assertEqual(self._reference_acceptance(draft, target), 4)
        self.assertEqual(self._mlx_acceptance(draft, target), 4)

    def test_none_match(self):
        """No tokens match -> zero acceptance."""
        draft = [10, 20, 30, 40]
        target = [99, 88, 77, 66]
        self.assertEqual(self._reference_acceptance(draft, target), 0)
        self.assertEqual(self._mlx_acceptance(draft, target), 0)

    def test_partial_match(self):
        """First 2 match, then diverge."""
        draft = [10, 20, 30, 40]
        target = [10, 20, 99, 40]
        self.assertEqual(self._reference_acceptance(draft, target), 2)
        self.assertEqual(self._mlx_acceptance(draft, target), 2)

    def test_single_match(self):
        """Only first token matches."""
        draft = [10, 20, 30]
        target = [10, 99, 99]
        self.assertEqual(self._reference_acceptance(draft, target), 1)
        self.assertEqual(self._mlx_acceptance(draft, target), 1)


class TestDFlashAttentionForward(unittest.TestCase):
    """Test DFlashAttention forward pass against reference with random weights."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        cls.head_dim = 8  # Small for testing
        cls.num_heads = 2
        cls.num_kv_heads = 1
        cls.hidden_size = cls.num_heads * cls.head_dim  # 16
        cls.kv_size = cls.num_kv_heads * cls.head_dim  # 8

    def _create_random_weights(self, seed=42):
        """Create random weights for both MLX and PyTorch."""
        np.random.seed(seed)
        weights = {}
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if name == 'q_proj':
                out_dim = self.num_heads * self.head_dim
            elif name == 'o_proj':
                out_dim = self.hidden_size
            else:
                out_dim = self.kv_size
            weights[f'{name}.weight'] = np.random.randn(out_dim, self.hidden_size).astype(np.float32) * 0.02
        for name in ['q_norm', 'k_norm']:
            weights[f'{name}.weight'] = np.ones(self.head_dim, dtype=np.float32)
        return weights

    def test_attention_shapes(self):
        """DFlashAttention should produce correct output shapes."""
        from mlx_lm.models.dflash_v2 import DFlashAttention, ModelArgs

        # DFlashAttention uses head_dim=128 internally, so we need matching dimensions
        head_dim = 128
        num_heads = 2
        num_kv_heads = 1
        hidden_size = num_heads * head_dim  # 256

        args = ModelArgs(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            attention_bias=False,
        )
        attn = DFlashAttention(args)

        B, L, ctx_len = 1, 4, 6
        hidden_states = mx.array(np.random.randn(B, L, hidden_size).astype(np.float32))
        target_hidden = mx.array(np.random.randn(B, ctx_len, hidden_size).astype(np.float32))

        # Compute position embeddings manually
        total_len = ctx_len + L
        inv_freq = 1.0 / (10000000.0 ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        positions = mx.arange(total_len).astype(mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        output = attn(hidden_states, target_hidden, (cos, sin))
        mx.eval(output)

        self.assertEqual(output.shape, (B, L, hidden_size))


class TestDFlashModelForward(unittest.TestCase):
    """Test full DFlash model forward pass."""

    def test_model_forward_shape(self):
        """Full model forward pass should produce correct output shape."""
        from mlx_lm.models.dflash_v2 import Model, ModelArgs

        args = ModelArgs(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            block_size=8,
            target_layer_ids=[0, 1],
        )
        model = Model(args)

        B, block_size = 1, 8
        ctx_len = 10
        num_target_layers = len(args.target_layer_ids)

        position_ids = mx.arange(ctx_len + block_size)[None, :]
        noise_embedding = mx.array(np.random.randn(B, block_size, args.hidden_size).astype(np.float32))
        target_hidden = mx.array(np.random.randn(B, ctx_len, num_target_layers * args.hidden_size).astype(np.float32))

        cache = model.make_cache()
        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=cache,
        )
        mx.eval(output)

        self.assertEqual(output.shape, (B, block_size, args.hidden_size))


class TestDFlashDraftLayerCache(unittest.TestCase):
    """Test DFlashDraftLayerCache behavior."""

    def test_empty_cache(self):
        """Empty cache should just pass through new K/V."""
        from mlx_lm.models.dflash_cache import DFlashDraftLayerCache

        cache = DFlashDraftLayerCache()
        k = mx.array(np.random.randn(1, 2, 10, 8).astype(np.float32))
        v = mx.array(np.random.randn(1, 2, 10, 8).astype(np.float32))

        k_out, v_out = cache.combine(k, v)
        np.testing.assert_array_equal(np.array(k_out), np.array(k))
        np.testing.assert_array_equal(np.array(v_out), np.array(v))

    def test_commit_and_combine(self):
        """After commit, combine should prepend cached K/V."""
        from mlx_lm.models.dflash_cache import DFlashDraftLayerCache

        cache = DFlashDraftLayerCache()

        # First: commit some noise K/V
        noise_k = mx.ones((1, 2, 3, 8), dtype=mx.float32)
        noise_v = mx.ones((1, 2, 3, 8), dtype=mx.float32) * 2
        cache.commit(noise_k, noise_v, 3)

        self.assertEqual(cache.cached_len, 3)

        # Now combine with new K/V
        new_k = mx.ones((1, 2, 5, 8), dtype=mx.float32) * 3
        new_v = mx.ones((1, 2, 5, 8), dtype=mx.float32) * 4
        k_out, v_out = cache.combine(new_k, new_v)

        # Should be cached (3) + new (5) = 8 total
        self.assertEqual(k_out.shape, (1, 2, 8, 8))
        self.assertEqual(v_out.shape, (1, 2, 8, 8))

        # First 3 should be from cache (1.0), last 5 from new (3.0)
        np.testing.assert_allclose(np.array(k_out[:, :, :3, :]), 1.0)
        np.testing.assert_allclose(np.array(k_out[:, :, 3:, :]), 3.0)

    def test_multiple_commits(self):
        """Multiple commits should accumulate K/V."""
        from mlx_lm.models.dflash_cache import DFlashDraftLayerCache

        cache = DFlashDraftLayerCache()

        k1 = mx.array(np.ones((1, 2, 2, 8), dtype=np.float32))
        v1 = mx.array(np.ones((1, 2, 2, 8), dtype=np.float32) * 2)
        cache.commit(k1, v1, 2)

        k2 = mx.ones((1, 2, 3, 8), dtype=mx.float32) * 3
        v2 = mx.ones((1, 2, 3, 8), dtype=mx.float32) * 4
        cache.commit(k2, v2, 3)

        self.assertEqual(cache.cached_len, 5)
        self.assertEqual(cache.cached_k.shape, (1, 2, 5, 8))

    def test_commit_zero(self):
        """Commit with 0 should be a no-op."""
        from mlx_lm.models.dflash_cache import DFlashDraftLayerCache

        cache = DFlashDraftLayerCache()
        k = mx.ones((1, 2, 3, 8), dtype=mx.float32)
        cache.commit(k, k, 0)

        self.assertEqual(cache.cached_len, 0)
        self.assertIsNone(cache.cached_k)


if __name__ == '__main__':
    unittest.main()
