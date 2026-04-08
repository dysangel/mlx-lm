# Copyright © 2025 Apple Inc.

"""Block diffusion speculative decoding using DFlash draft model - Reference implementation port."""

from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.cache import KVCache
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

        # Process through layers with cache
        for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
            h = layer(h, mask=None, cache=c)
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

    num_input_tokens = len(prompt_tokens)
    if num_input_tokens == 0:
        raise ValueError("Prompt must not be empty")

    # Initialize caches
    target_cache = model.make_cache()
    draft_cache = draft_model.make_cache()

    # Prefill stage - capture context features from target model
    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)

    # Process prompt tokens
    prompt_tokens = prompt_tokens[None, :]
    output = target_model_with_hidden(prompt_tokens, cache=target_cache)
    tokens, logprobs = process_sample(output.logits.squeeze(0))

    # Yield first token
    first_token = tokens[-1].item()
    first_logprobs = logprobs[-1]
    yield first_token, first_logprobs, False
    ntoks = 1

    # Extract context features from target model (only prompt tokens, not first generated token)
    # This matches the reference which uses only prompt context for initial materialization
    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )

    # Now process first token through cache
    first_token_input = mx.array([[first_token]])
    target_model_with_hidden(first_token_input, cache=target_cache)

    # Materialize target context into draft cache ONCE
    ctx_len = target_hidden.shape[1]
    ctx_position_ids = mx.arange(ctx_len)[None, :]
    draft_model.materialize_target_hidden(target_hidden, draft_cache, ctx_position_ids)

    # Get inner model for embeddings
    target_inner = get_inner_model(model)

    # Initialize output_ids like the reference
    max_length = num_input_tokens + max_tokens + block_size
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :num_input_tokens] = prompt_tokens
    output_ids[:, num_input_tokens] = first_token

    # Position for next token
    start = num_input_tokens + 1

    # Decode loop
    while ntoks < max_tokens:
        remaining = max_tokens - ntoks
        current_block_size = min(block_size, remaining)

        with mx.stream(generation_stream):
            # Get block of tokens from output_ids
            # Include previous token as seed (like reference)
            prev_token = output_ids[:, start - 1][:, None]  # [1] -> [1, 1]
            block_output_ids = mx.concatenate([prev_token, output_ids[:, start : start + current_block_size - 1]], axis=-1)

            # Create embeddings from tokens
            noise_embedding = target_inner.embed_tokens(block_output_ids)

            # Create position_ids for the draft model
            # Use cache manager's current size to determine starting position
            cache_size = draft_cache.total_size()
            noise_position_ids = mx.arange(cache_size - 1, cache_size - 1 + current_block_size)[None, :]

            # Draft model generates
            draft_output = draft_model(
                position_ids=noise_position_ids,
                noise_embedding=noise_embedding,
                cache=draft_cache,
            )

            # Get draft logits using target model's lm_head
            if hasattr(model, 'lm_head'):
                draft_logits = model.lm_head(draft_output)
            else:
                draft_logits = target_inner.embed_tokens.as_linear(draft_output)

            # Force evaluation
            mx.eval(draft_logits)

            # Sample draft tokens (skip first position which is the seed token)
            draft_tokens_block = mx.argmax(draft_logits[:, -current_block_size + 1:, :], axis=-1).squeeze(0)

            # Convert draft_logits to logprobs for yielding
            draft_logprobs = draft_logits - mx.logsumexp(draft_logits, axis=-1, keepdims=True)

            # For now, skip draft model entirely - just use target model
            # This establishes a clean baseline
            acceptance_length = 0

            # Predict next token from target model using current cache state
            # Use a repeat of the last token to trigger prediction
            last_token_id = output_ids[:, start - 1].item()
            token_input = mx.array([[last_token_id]])
            target_output = target_model_with_hidden(token_input, cache=target_cache)
            mx.eval(target_output.logits)
            final_token = sampler(target_output.logits[0, -1:, :])[0].item()

            # Update output_ids
            for i in range(acceptance_length):
                output_ids[:, start + i] = draft_tokens_block[i].item()
            output_ids[:, start + acceptance_length] = final_token

            # Yield accepted draft tokens
            for i in range(acceptance_length):
                yield draft_tokens_block[i].item(), draft_logprobs[:, i, :].squeeze(0), True
                ntoks += 1
                if ntoks >= max_tokens:
                    break

            if ntoks >= max_tokens:
                break

            # Yield final target token
            target_logprobs = target_output.logits - mx.logsumexp(target_output.logits, axis=-1, keepdims=True)
            yield final_token, target_logprobs[:, -1, :].squeeze(0), False
            ntoks += 1

            # Clear cache periodically to prevent memory buildup
            if ntoks % 256 == 0:
                mx.clear_cache()

            if ntoks >= max_tokens:
                break

            # Move start forward
            start += acceptance_length + 1

            # Update target_hidden to include newly generated tokens
            new_target_hidden = extract_context_feature(target_output.hidden_states, draft_model.target_layer_ids)[:, -acceptance_length - 1:, :]
            target_hidden = mx.concatenate([target_hidden, new_target_hidden], axis=1)

            # Crop noise tokens from draft cache and rematerialize updated context
            # This removes the noise tokens we just processed, keeping only context
            draft_cache.crop_noise_tokens()

            # Rematerialize updated target context (includes newly generated tokens)
            draft_model.materialize_target_hidden(target_hidden, draft_cache)

    return
