# Copyright © 2025 Apple Inc.

"""Block diffusion speculative decoding using DFlash draft model.

Aligned with the reference PyTorch implementation:
- Persistent draft KVCache (cropped to `start` after each iteration)
- target_hidden replaced each iteration (only acceptance_length + 1 tokens)
- Position IDs span from draft cache length to block end
- Target cache cropped to `start` after each iteration
"""

from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.cache import KVCache
from .models.base import create_attention_mask, create_ssm_mask
from .sample_utils import make_sampler


def get_inner_model(model: nn.Module) -> nn.Module:
    """Get the inner model that contains embed_tokens, layers, and norm."""
    inner_model = model
    if hasattr(inner_model, 'language_model'):
        inner_model = inner_model.language_model
    if hasattr(inner_model, 'model'):
        inner_model = inner_model.model
    return inner_model


def extract_context_feature(
    hidden_states: List[mx.array],
    layer_ids: List[int],
) -> mx.array:
    """Extract and concatenate hidden states from specified target model layers."""
    offset = 1  # hidden_states[0] is embedding, [1:] are layer outputs
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return mx.concatenate(selected_states, axis=-1)


def _crop_cache(cache_list, target_len):
    """Crop all caches to keep only the first target_len positions.

    For KVCache: adjust offset (ring buffer, truncates logical end).
    For SpeculativeArraysCache: slice arrays to target_len (removes from end).
    """
    for c in cache_list:
        if hasattr(c, 'keys') and c.keys is not None:
            # KVCache: uses offset to track real length
            if c.offset > target_len:
                c.offset = target_len
        elif hasattr(c, 'crop'):
            # SpeculativeArraysCache: truncate arrays from end
            c.crop(target_len)
        elif hasattr(c, 'trim'):
            # Fallback
            current = c.offset if hasattr(c, 'offset') else 0
            if current > target_len:
                c.trim(current - target_len)


class ModelWithHiddenStates(nn.Module):
    """Wrapper to capture intermediate hidden states from target model."""

    def __init__(self, model: nn.Module, target_layer_ids: List[int]):
        super().__init__()
        self.model = model
        self.target_layer_ids = target_layer_ids
        self.hidden_states = []

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> Any:
        """Forward pass that captures intermediate hidden states."""
        self.hidden_states = []

        inner_model = self.model
        if hasattr(inner_model, 'language_model'):
            inner_model = inner_model.language_model
        lm_head_container = inner_model
        if hasattr(inner_model, 'model'):
            inner_model = inner_model.model

        h = inner_model.embed_tokens(inputs)
        self.hidden_states.append(h)

        if cache is None:
            cache = [None] * len(inner_model.layers)

        fa_idx = getattr(inner_model, 'fa_idx', 3)
        ssm_idx = getattr(inner_model, 'ssm_idx', 0)
        fa_mask = create_attention_mask(h, cache[fa_idx] if fa_idx < len(cache) else None)
        ssm_mask = create_ssm_mask(h, cache[ssm_idx] if ssm_idx < len(cache) else None)

        for layer, c in zip(inner_model.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
            self.hidden_states.append(h)

        h = inner_model.norm(h)

        if hasattr(lm_head_container, "args") and hasattr(lm_head_container.args, "tie_word_embeddings") and lm_head_container.args.tie_word_embeddings:
            logits = inner_model.embed_tokens.as_linear(h)
        elif hasattr(lm_head_container, "lm_head"):
            logits = lm_head_container.lm_head(h)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(h)
        else:
            logits = inner_model.embed_tokens.as_linear(h)

        class OutputWithHidden:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states

        return OutputWithHidden(logits, self.hidden_states)


def block_diffusion_generate_step(
    prompt: str,
    model: nn.Module,
    draft_model: nn.Module,
    tokenizer: Any,
    max_tokens: int = 256,
    **kwargs,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Generate tokens using DFlash block diffusion speculative decoding.

    Follows the reference _spec_generate from dflash.py:
    1. Persistent draft KVCache cropped to `start` after each iteration
    2. target_hidden replaced with only the latest chunk each iteration
    3. Position IDs span from draft cache length to block end
    4. Target cache cropped to `start` after each iteration

    Yields:
        Tuple of (token_id, logprobs, from_draft)
    """
    import logging
    logger = logging.getLogger(__name__)

    sampler = kwargs.get("sampler")
    if sampler is None:
        sampler = make_sampler(
            temp=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            min_p=kwargs.get("min_p", 0.0),
            min_tokens_to_keep=kwargs.get("min_tokens_to_keep", 1),
        )

    block_size = getattr(draft_model, "block_size", 16)
    mask_token_id = getattr(draft_model, "mask_token_id", None)

    if isinstance(prompt, str):
        prompt_tokens = mx.array(tokenizer.encode(prompt))
    else:
        prompt_tokens = prompt

    num_input_tokens = prompt_tokens.shape[1] if prompt_tokens.ndim == 2 else len(prompt_tokens)
    if num_input_tokens == 0:
        raise ValueError("Prompt must not be empty")

    # Initialize caches
    if hasattr(model, 'make_speculative_cache'):
        target_cache = model.make_speculative_cache()
    else:
        target_cache = model.make_cache()

    # Persistent draft KVCache — accumulates across iterations, cropped to start
    draft_cache = draft_model.make_cache()

    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)
    target_inner = get_inner_model(model)

    # === PREFILL ===
    prompt_tokens = prompt_tokens[None, :]
    prefill_output = target_model_with_hidden(prompt_tokens, cache=target_cache)
    mx.eval(prefill_output.logits)

    # Sample first token and place it in output_ids
    first_token = mx.argmax(prefill_output.logits[:, -1, :], axis=-1).squeeze(0)

    # Extract target_hidden from prefill — all prompt token hidden states
    target_hidden = extract_context_feature(
        prefill_output.hidden_states, draft_model.target_layer_ids
    )
    mx.eval(target_hidden)

    # Initialize output_ids buffer (matching reference layout)
    max_length = num_input_tokens + max_tokens + block_size
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :num_input_tokens] = prompt_tokens
    output_ids[:, num_input_tokens] = first_token

    # Yield first token
    yield first_token.item(), prefill_output.logits[:, -1, :], False
    ntoks = 1

    start = num_input_tokens  # Start at first generated token position

    # === DECODE LOOP ===
    while start < num_input_tokens + max_tokens and ntoks < max_tokens:
        remaining = num_input_tokens + max_tokens - start
        current_block_size = min(block_size, remaining)
        if current_block_size < 2:
            break

        # === DRAFT PHASE ===
        # Block: [prev_token_or_first, mask, mask, ...] (block_size tokens)
        block_output_ids = mx.array(output_ids[:, start: start + current_block_size])

        # Draft model forward pass with persistent cache
        noise_embedding = target_inner.embed_tokens(block_output_ids)

        # Position IDs: from draft cache length to start + block_size
        # This ensures cos/sin length matches K length (ctx_len + noise_len)
        draft_cache_len = draft_cache[0].offset
        draft_position_ids = mx.arange(
            draft_cache_len, draft_cache_len + target_hidden.shape[1] + current_block_size
        )[None, :]

        draft_output = draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        # Only take the last current_block_size-1 positions (skip prev_token)
        draft_logits = target_inner.embed_tokens.as_linear(
            draft_output[:, -current_block_size + 1:, :]
        )
        mx.eval(draft_logits)

        # Crop draft cache to start (discard speculative K/V beyond verified position)
        _crop_cache(draft_cache, start)

        # Sample draft tokens (replace mask tokens with predictions)
        block_output_ids[:, 1:] = mx.argmax(draft_logits, axis=-1)

        # === VERIFY PHASE ===
        # Save checkpoint before verify (for rollback on rejection)
        for c in target_cache:
            if hasattr(c, 'save_checkpoint'):
                c.save_checkpoint()

        # Run target model on block_output_ids to verify draft tokens
        verify_output = target_model_with_hidden(block_output_ids, cache=target_cache)
        mx.eval(verify_output.logits)

        # Sample posterior from target model
        posterior = mx.argmax(verify_output.logits, axis=-1)

        # Compute acceptance: consecutive matches of draft vs target
        draft_pred = block_output_ids[:, 1:]
        target_pred = posterior[:, :-1]
        acceptance_length = int(
            (draft_pred == target_pred).cumprod(axis=1).sum().squeeze().item()
        )
        logger.debug(f"Acceptance: {acceptance_length}/{current_block_size - 1}")

        # Finalize output: accepted draft tokens + bonus target token
        output_ids[:, start: start + acceptance_length] = block_output_ids[:, :acceptance_length]

        # Rollback target cache if tokens were rejected
        if acceptance_length < current_block_size - 1:
            # Some tokens rejected — rollback to checkpoint and rebuild
            for c in target_cache:
                if hasattr(c, 'rollback'):
                    # SpeculativeArraysCache: restore checkpoint
                    c.rollback()
                elif hasattr(c, 'keys') and c.keys is not None:
                    # KVCache: trim rejected tokens from end
                    num_to_trim = current_block_size - (acceptance_length + 1)
                    c.offset = c.offset - num_to_trim

            # Rebuild target cache with only accepted tokens first
            # (don't include bonus yet — we need fresh logits to determine it)
            if acceptance_length > 0:
                rebuild_tokens = block_output_ids[:, :acceptance_length]
            else:
                rebuild_tokens = None

            if rebuild_tokens is not None:
                rebuild_output = target_model_with_hidden(rebuild_tokens, cache=target_cache)
                mx.eval(rebuild_output.logits)
                # Get fresh bonus token from rebuild logits
                bonus_token = mx.argmax(rebuild_output.logits[:, -1, :], axis=-1).squeeze(0)
                bonus_logits = rebuild_output.logits[:, -1, :]

                # Now feed the bonus token to update cache and get hidden states
                bonus_input = bonus_token[None, None]
                bonus_output = target_model_with_hidden(bonus_input, cache=target_cache)
                mx.eval(bonus_output.logits)

                # Combined hidden states: rebuild tokens + bonus token
                rebuild_hidden = extract_context_feature(
                    rebuild_output.hidden_states, draft_model.target_layer_ids
                )
                bonus_hidden = extract_context_feature(
                    bonus_output.hidden_states, draft_model.target_layer_ids
                )
                target_hidden = mx.concatenate([rebuild_hidden, bonus_hidden], axis=1)
                mx.eval(target_hidden)

                saved_next_logits = bonus_output.logits[:, -1, :]
            else:
                # No accepted tokens — just get bonus token from verify logits at position 0
                # This is OK because the cache was rolled back to pre-verify state
                bonus_token = posterior[:, 0].squeeze()
                bonus_logits = verify_output.logits[:, 0, :]

                bonus_input = bonus_token[None, None]
                bonus_output = target_model_with_hidden(bonus_input, cache=target_cache)
                mx.eval(bonus_output.logits)

                target_hidden = extract_context_feature(
                    bonus_output.hidden_states, draft_model.target_layer_ids
                )
                mx.eval(target_hidden)

                saved_next_logits = bonus_output.logits[:, -1, :]

        else:
            # Full acceptance: cache already has all tokens, no rollback needed
            bonus_token = posterior[:, acceptance_length].squeeze()
            bonus_logits = verify_output.logits[:, acceptance_length, :]

            target_hidden = extract_context_feature(
                verify_output.hidden_states, draft_model.target_layer_ids
            )[:, :acceptance_length + 1, :]
            mx.eval(target_hidden)

            saved_next_logits = verify_output.logits[:, -1, :]

        output_ids[:, start + acceptance_length] = bonus_token

        # Yield accepted draft tokens
        for i in range(acceptance_length):
            token_id = block_output_ids[0, i].item()
            yield token_id, draft_logits[:, i, :], True
            ntoks += 1

        # Yield bonus target token
        yield bonus_token.item(), bonus_logits, False
        ntoks += 1

        # Crop draft cache to new start position
        new_start = start + acceptance_length + 1
        _crop_cache(draft_cache, new_start)

        start = new_start

    return
