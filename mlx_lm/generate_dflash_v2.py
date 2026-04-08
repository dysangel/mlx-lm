# Copyright © 2025 Apple Inc.

"""Block diffusion speculative decoding using DFlash draft model - Reference implementation port."""

from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.cache import KVCache
from .models.base import create_attention_mask, create_ssm_mask
from .sample_utils import make_sampler

# Create a global stream for generation
generation_stream = mx.new_stream(mx.default_device())


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
    """Extract and concatenate hidden states from specified target model layers.

    Args:
        hidden_states: ALL hidden states from target model
                       [0] = embedding, [1:] = all layer outputs
        layer_ids: Target layer IDs (e.g., [1, 8, 15, 22, 29])

    Returns:
        Concatenated hidden states [B, seq_len, num_layers * hidden_size]
    """
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return mx.concatenate(selected_states, axis=-1)


class ModelWithHiddenStates(nn.Module):
    """Wrapper to capture intermediate hidden states from target model."""

    def __init__(self, model: nn.Module, target_layer_ids: List[int]):
        super().__init__()
        self.model = model
        self.target_layer_ids = target_layer_ids
        self.hidden_states = []

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> Any:
        """Forward pass that captures intermediate hidden states."""
        self.hidden_states = []

        # Get the inner model - need to handle different model architectures
        inner_model = self.model
        if hasattr(inner_model, 'language_model'):
            inner_model = inner_model.language_model
        lm_head_container = inner_model
        if hasattr(inner_model, 'model'):
            inner_model = inner_model.model

        # Get embeddings
        h = inner_model.embed_tokens(inputs)
        self.hidden_states.append(h)

        # Create cache if needed
        if cache is None:
            cache = [None] * len(inner_model.layers)

        # Generate proper masks for Qwen3.5 (linear attention + full attention layers)
        # Get fa_idx and ssm_idx from the model
        fa_idx = getattr(inner_model, 'fa_idx', 3)  # Default to 3 for Qwen3.5
        ssm_idx = getattr(inner_model, 'ssm_idx', 0)  # Default to 0

        fa_mask = create_attention_mask(h, cache[fa_idx] if fa_idx < len(cache) else None)
        ssm_mask = create_ssm_mask(h, cache[ssm_idx] if ssm_idx < len(cache) else None)

        # Process through layers with proper masks
        for layer, c in zip(inner_model.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
            self.hidden_states.append(h)

        # Final normalization
        h = inner_model.norm(h)

        # Get logits
        if hasattr(lm_head_container, "args") and hasattr(lm_head_container.args, "tie_word_embeddings") and lm_head_container.args.tie_word_embeddings:
            logits = inner_model.embed_tokens.as_linear(h)
        elif hasattr(lm_head_container, "lm_head"):
            logits = lm_head_container.lm_head(h)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(h)
        else:
            logits = inner_model.embed_tokens.as_linear(h)

        # Return logits with a wrapper that keeps hidden_states
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

    Args:
        prompt: Input prompt
        model: Target model
        draft_model: DFlash draft model
        tokenizer: Tokenizer
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments (sampler, logits_processors, etc.)

    Yields:
        Tuple of (token_id, logprobs, from_draft)
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f"DFlash generation started: prompt='{prompt[:50]}...', max_tokens={max_tokens}")
    logger.info(f"kwargs keys: {list(kwargs.keys())}")

    # Helper function for sampling
    def process_sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tokens = sampler(logits)
        return tokens, logprobs

    # Get sampler
    sampler = kwargs.get("sampler")
    if sampler is None:
        sampler = make_sampler(
            temp=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            min_p=kwargs.get("min_p", 0.0),
            min_tokens_to_keep=kwargs.get("min_tokens_to_keep", 1),
        )

    # Get block size
    block_size = getattr(draft_model, "block_size", 16)
    mask_token_id = getattr(draft_model, "mask_token_id", None)

    # Encode prompt
    if isinstance(prompt, str):
        prompt_tokens = mx.array(tokenizer.encode(prompt))
    else:
        prompt_tokens = prompt

    num_input_tokens = prompt_tokens.shape[1] if prompt_tokens.ndim == 2 else len(prompt_tokens)
    if num_input_tokens == 0:
        raise ValueError("Prompt must not be empty")

    # Initialize caches
    # Use SpeculativeArraysCache if available (for checkpoint/rollback support)
    if hasattr(model, 'make_speculative_cache'):
        target_cache = model.make_speculative_cache()
    else:
        target_cache = model.make_cache()
    draft_cache = draft_model.make_cache()

    logger.debug(f"Target cache types: {[type(c).__name__ for c in target_cache[:3]]}")
    logger.debug(f"Draft cache types: {[type(c).__name__ for c in draft_cache[:3]]}")

    # Prefill stage
    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)
    target_inner = get_inner_model(model)

    logger.debug(f"Target cache types: {[type(c).__name__ for c in target_cache[:3]]}")

    # Prefill with ORIGINAL model (cache-safe)
    prompt_tokens = prompt_tokens[None, :]
    prefill_logits = model(prompt_tokens, cache=target_cache)
    mx.eval(prefill_logits)

    # Get first token from prefill
    first_token = mx.argmax(prefill_logits[:, -1, :], axis=-1).squeeze(0).item()
    first_logprobs = prefill_logits[:, -1, :]
    yield first_token, first_logprobs, False
    ntoks = 1

    # Extract target_hidden WITHOUT cache (safe)
    hidden_output = target_model_with_hidden(prompt_tokens, cache=None)
    mx.eval(hidden_output.logits)
    target_hidden = extract_context_feature(
        hidden_output.hidden_states,
        draft_model.target_layer_ids,
    )
    mx.eval(target_hidden)

    # Initialize output_ids
    max_length = num_input_tokens + max_tokens + block_size
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :num_input_tokens] = prompt_tokens
    output_ids[:, num_input_tokens] = first_token

    start = num_input_tokens + 1

    # Decode loop
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
            noise_position_ids = mx.arange(start, start + current_block_size)[None, :]

            # Call draft model
            draft_cache = draft_model.make_cache()
            draft_output = draft_model(
                position_ids=noise_position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            mx.eval(draft_output)

            # Get draft logits
            draft_logits = target_inner.embed_tokens.as_linear(draft_output)
            mx.eval(draft_logits)

            # Sample draft tokens
            if draft_logits.shape[1] > current_block_size - 1:
                draft_tokens_block = mx.argmax(draft_logits[:, -current_block_size + 1:, :], axis=-1).squeeze(0)
            else:
                draft_tokens_block = mx.array([], dtype=mx.uint32)

            logger.debug(f"Draft tokens: {draft_tokens_block[:5].tolist() if draft_tokens_block.size > 0 else []}")

            # === VERIFY PHASE ===
            # Verify by passing ALL tokens (prompt + generated + draft) without cache
            # This is correct but O(n^2) - can optimize later with proper rollback
            acceptance_length = 0
            if draft_tokens_block.size > 0:
                # Build full verification input: prompt + generated + draft
                draft_len = len(draft_tokens_block)
                full_verify_input = output_ids[:, :start]  # prompt + all generated
                full_verify_input = mx.concatenate([full_verify_input, draft_tokens_block[None, :]], axis=-1)
                verify_output = target_model_with_hidden(full_verify_input, cache=None)
                logits = verify_output.logits
                mx.eval(logits)

                # Target predictions for draft positions
                # logits has one prediction per input token; we want predictions at draft positions
                draft_len = len(draft_tokens_block)
                # Predictions for draft token positions: from -draft_len-1 to -1
                target_tokens = mx.argmax(logits[:, -draft_len - 1:-1, :], axis=-1).squeeze(0)

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

                # Yield target token
                # Target token comes from the prediction AFTER the last accepted draft token,
                # NOT from the last position (which is poisoned by rejected draft tokens)
                target_logits = logits[:, start - 1 + acceptance_length, :]
                target_token = mx.argmax(target_logits, axis=-1).squeeze(0)
                output_ids[:, start + acceptance_length] = target_token
                yield target_token.item(), target_logits, False
                ntoks += 1

                if ntoks >= max_tokens:
                    break

                # === CACHE UPDATE ===
                # Always rebuild cache from scratch with accepted tokens
                # (since we verified without cache, the cache is stale)
                accepted_plus_target = output_ids[:, start:start + acceptance_length + 1]
                logger.debug(f"Updating cache with {acceptance_length + 1} tokens")
                cache_logits = model(accepted_plus_target, cache=target_cache)
                mx.eval(cache_logits)

                # === TARGET HIDDEN REBUILD (without cache) ===
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

            # Use original model for cache update
            decode_logits = model(prev_token, cache=target_cache)
            mx.eval(decode_logits)

            target_token = mx.argmax(decode_logits[:, -1, :], axis=-1).squeeze(0)
            output_ids[:, start] = target_token
            yield target_token.item(), decode_logits[:, -1, :], False
            ntoks += 1

            if ntoks >= max_tokens:
                break

            # Rebuild target_hidden WITHOUT cache
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
