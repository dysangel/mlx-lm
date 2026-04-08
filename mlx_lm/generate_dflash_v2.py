# Copyright © 2025 Apple Inc.

"""Block diffusion speculative decoding using DFlash draft model."""

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

    Uses cache-based verification with the "saved logits" trick:
    - After each cache update, save the logits from the last position
    - Those logits predict what the first draft token (d0) should be
    - During verification, feed only draft_tokens (no prev_token) to avoid duplication
    - Compare: d0 vs saved_logits, d1 vs logits[0], d2 vs logits[1], ...

    Yields:
        Tuple of (token_id, logprobs, from_draft)
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)
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

    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)
    target_inner = get_inner_model(model)

    # === PREFILL ===
    prompt_tokens = prompt_tokens[None, :]
    prefill_logits = model(prompt_tokens, cache=target_cache)
    mx.eval(prefill_logits)

    first_token = mx.argmax(prefill_logits[:, -1, :], axis=-1).squeeze(0).item()
    yield first_token, prefill_logits[:, -1, :], False
    ntoks = 1

    # Extract target_hidden for draft model (without cache)
    hidden_output = target_model_with_hidden(prompt_tokens, cache=None)
    mx.eval(hidden_output.logits)
    target_hidden = extract_context_feature(hidden_output.hidden_states, draft_model.target_layer_ids)
    mx.eval(target_hidden)

    # Initialize output_ids
    max_length = num_input_tokens + max_tokens + block_size
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :num_input_tokens] = prompt_tokens
    output_ids[:, num_input_tokens] = first_token

    start = num_input_tokens + 1

    # Saved logits from the last cache update - predicts what the next token should be
    # This is used for the first draft token comparison (d0)
    saved_next_logits = None

    # === DECODE LOOP ===
    iteration = 0
    while ntoks < max_tokens:
        iteration += 1
        remaining = max_tokens - ntoks
        current_block_size = min(block_size, remaining)
        logger.debug(f"Iteration {iteration}: ntoks={ntoks}, remaining={remaining}")

        if iteration > 1 and current_block_size > 1:
            # === DRAFT PHASE ===
            prev_token = output_ids[:, start - 1][:, None]
            noise_tokens = mx.full([1, max(0, current_block_size - 1)], mask_token_id, dtype=mx.uint32)
            draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1)
            noise_embedding = target_inner.embed_tokens(draft_input)
            # Position IDs must cover full context + noise range for RoPE on full K
            ctx_len = target_hidden.shape[1]
            draft_position_ids = mx.arange(ctx_len + current_block_size)[None, :]

            draft_cache = draft_model.make_cache()
            draft_output = draft_model(
                position_ids=draft_position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            mx.eval(draft_output)

            draft_logits = target_inner.embed_tokens.as_linear(draft_output)
            mx.eval(draft_logits)

            if draft_logits.shape[1] > current_block_size - 1:
                draft_tokens_block = mx.argmax(draft_logits[:, -current_block_size + 1:, :], axis=-1).squeeze(0)
            else:
                draft_tokens_block = mx.array([], dtype=mx.uint32)

            # === VERIFY PHASE (cache-based, no prev_token duplication) ===
            acceptance_length = 0
            if draft_tokens_block.size > 0:
                # Save cache state before verification (for rollback)
                for c in target_cache:
                    if hasattr(c, 'save_checkpoint'):
                        c.save_checkpoint()

                # Feed ONLY draft_tokens to model with cache (no prev_token)
                verify_logits = model(draft_tokens_block[None, :], cache=target_cache)
                mx.eval(verify_logits)

                # Build target predictions for comparison:
                # d0 prediction comes from saved_next_logits (from previous cache update)
                # d1,d2,... predictions come from verify_logits
                draft_len = len(draft_tokens_block)

                if saved_next_logits is not None:
                    # d0 comparison: saved logits predict what d0 should be
                    d0_target = mx.argmax(saved_next_logits, axis=-1).squeeze(0)
                    # d1+ comparisons: verify_logits[:-1] predict what d1,d2,... should be
                    if draft_len > 1:
                        d_rest_targets = mx.argmax(verify_logits[:, :-1, :], axis=-1).squeeze(0)
                        target_tokens = mx.concatenate([d0_target[None], d_rest_targets])
                    else:
                        target_tokens = d0_target[None]
                else:
                    # Fallback: no saved logits, skip d0 comparison
                    if draft_len > 1:
                        target_tokens = mx.argmax(verify_logits[:, :-1, :], axis=-1).squeeze(0)
                        target_tokens = mx.concatenate([mx.array([-1]), target_tokens])
                    else:
                        target_tokens = mx.array([-1])

                if logger.isEnabledFor(logging.DEBUG):
                    draft_decoded = [tokenizer.decode([t]) for t in draft_tokens_block[:5].tolist()]
                    target_decoded = [tokenizer.decode([t]) for t in target_tokens[:5].tolist()]
                    logger.debug(f"Draft: {draft_tokens_block[:5].tolist()} -> {draft_decoded}")
                    logger.debug(f"Target: {target_tokens[:5].tolist()} -> {target_decoded}")

                # Count consecutive matches from beginning
                acceptance_length = int(
                    (mx.cumsum(draft_tokens_block == target_tokens) == mx.arange(1, len(target_tokens) + 1)).sum()
                )
                logger.debug(f"Acceptance: {acceptance_length}/{len(draft_tokens_block)}")

                # Rollback cache if tokens were rejected
                if acceptance_length < len(draft_tokens_block):
                    num_to_trim = draft_len - acceptance_length
                    for c in target_cache:
                        if hasattr(c, 'rollback'):
                            # SpeculativeArraysCache: use checkpoint rollback
                            c.rollback()
                        elif hasattr(c, 'keys') and c.keys is not None:
                            # KVCache: trim arrays from the end
                            new_offset = c.offset - num_to_trim
                            c.offset = new_offset
                            c.keys = c.keys[..., :new_offset, :]
                            c.values = c.values[..., :new_offset, :]
                    logger.debug(f"Rolled back {num_to_trim} rejected tokens from cache")

                # Yield accepted draft tokens
                for i in range(acceptance_length):
                    token_id = draft_tokens_block[i].item()
                    output_ids[:, start + i] = token_id
                    yield token_id, draft_logits[:, i + 1, :], True
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break

                if ntoks >= max_tokens:
                    break

                # Yield target token from the correct position
                # - If all rejected: use saved_next_logits (prediction from previous cache state)
                # - If partial: use verify_logits at position after last accepted draft
                # - If all accepted: use verify_logits at last position (bonus token)
                if acceptance_length == 0 and saved_next_logits is not None:
                    target_logits = saved_next_logits
                elif acceptance_length < len(draft_tokens_block):
                    target_logits = verify_logits[:, acceptance_length - 1, :]
                else:
                    target_logits = verify_logits[:, -1, :]
                target_token = mx.argmax(target_logits, axis=-1).squeeze(0)
                output_ids[:, start + acceptance_length] = target_token
                yield target_token.item(), target_logits, False
                ntoks += 1

                if ntoks >= max_tokens:
                    break

                # === CACHE UPDATE ===
                # After rollback, cache is missing accepted tokens + target token
                # Rebuild cache with only the accepted tokens
                if acceptance_length > 0:
                    rebuild_tokens = mx.concatenate([
                        draft_tokens_block[:acceptance_length][None, :],
                        mx.array([[target_token.item()]])
                    ], axis=-1)
                else:
                    rebuild_tokens = mx.array([[target_token.item()]])
                rebuild_logits = model(rebuild_tokens, cache=target_cache)
                mx.eval(rebuild_logits)

                # Save logits for next iteration's d0 comparison
                saved_next_logits = rebuild_logits[:, -1, :]

                # === TARGET HIDDEN REBUILD ===
                total_tokens = start + acceptance_length + 1
                all_tokens_array = output_ids[:, :total_tokens]
                hidden_output = target_model_with_hidden(all_tokens_array, cache=None)
                mx.eval(hidden_output.logits)
                target_hidden = extract_context_feature(
                    hidden_output.hidden_states,
                    draft_model.target_layer_ids,
                )
                mx.eval(target_hidden)

                start += acceptance_length + 1
        else:
            # === ITERATION 1: Single token decode ===
            prev_token = mx.array([[first_token]])
            decode_logits = model(prev_token, cache=target_cache)
            mx.eval(decode_logits)

            target_token = mx.argmax(decode_logits[:, -1, :], axis=-1).squeeze(0)
            output_ids[:, start] = target_token
            yield target_token.item(), decode_logits[:, -1, :], False
            ntoks += 1

            if ntoks >= max_tokens:
                break

            # Add target_token to cache + save its logits for next iteration's d0 comparison
            update_logits = model(mx.array([[target_token.item()]]), cache=target_cache)
            mx.eval(update_logits)
            saved_next_logits = update_logits[:, -1, :]

            # Rebuild target_hidden
            total_tokens = start + 1
            all_tokens_array = output_ids[:, :total_tokens]
            hidden_output = target_model_with_hidden(all_tokens_array, cache=None)
            mx.eval(hidden_output.logits)
            target_hidden = extract_context_feature(
                hidden_output.hidden_states,
                draft_model.target_layer_ids,
            )
            mx.eval(target_hidden)

            start += 1

        if ntoks >= max_tokens:
            break

    return
