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

    # Materialize target context into draft cache ONCE
    ctx_len = target_hidden.shape[1]
    ctx_position_ids = mx.arange(ctx_len)[None, :]
    draft_model.materialize_target_hidden(target_hidden, draft_cache, ctx_position_ids)

    # Update noise_start to track where noise tokens begin (after context)
    draft_cache.update_noise_start(ctx_len)

    # Get inner model for embeddings
    target_inner = get_inner_model(model)

    # Initialize output_ids like the reference
    max_length = num_input_tokens + max_tokens + block_size
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :num_input_tokens] = prompt_tokens
    output_ids[:, num_input_tokens] = first_token

    # Position for next token
    start = num_input_tokens + 1

    # Decode loop - test draft verification
    iteration = 0
    while ntoks < max_tokens:
        iteration += 1
        remaining = max_tokens - ntoks
        current_block_size = min(block_size, remaining)
        logger.debug(f"Iteration {iteration}: ntoks={ntoks}, remaining={remaining}")

        with mx.stream(generation_stream):
            # Generate draft tokens (but don't use them yet)
            if iteration > 1:
                prev_token = output_ids[:, start - 1][:, None]
                noise_tokens = mx.zeros([1, max(0, current_block_size - 1)], dtype=mx.uint32)
                draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1) if current_block_size > 1 else prev_token
                noise_embedding = target_inner.embed_tokens(draft_input)
                cache_size = draft_cache.total_size()
                noise_position_ids = mx.arange(cache_size, cache_size + current_block_size)[None, :]
                draft_output = draft_model(
                    position_ids=noise_position_ids,
                    noise_embedding=noise_embedding,
                    cache=draft_cache,
                )
                mx.eval(draft_output)

                # Get draft logits
                if hasattr(model, 'lm_head'):
                    draft_logits = model.lm_head(draft_output)
                else:
                    draft_logits = target_inner.embed_tokens.as_linear(draft_output)
                mx.eval(draft_logits)

                # Sample draft tokens
                if draft_logits.shape[1] > 1:
                    draft_tokens_block = mx.argmax(draft_logits[:, 1:, :], axis=-1).squeeze(0)
                else:
                    draft_tokens_block = mx.array([], dtype=mx.uint32)

                logger.debug(f"Generated {len(draft_tokens_block)} draft tokens")

                # Verify draft tokens using cache with proper rollback
                acceptance_length = 0
                if draft_tokens_block.size > 0:
                    # Save cache state for rollback
                    # Cache structure: list of ArraysCache, each with .cache = [keys, values]
                    saved_cache_state = []
                    for layer_cache in target_cache:
                        if hasattr(layer_cache, 'cache'):  # ArraysCache
                            # Copy the [keys, values] list
                            saved_cache_state.append([
                                mx.array(arr) if arr is not None else None
                                for arr in layer_cache.cache
                            ])
                        else:
                            # Direct KVCache - save its state
                            saved_cache_state.append(layer_cache.state)

                    # Run draft tokens through target model WITH cache
                    target_verify_input = draft_tokens_block[None, :]
                    logits = model(target_verify_input, cache=target_cache)
                    mx.eval(logits)

                    # Get target predictions
                    if logits.ndim == 3 and logits.shape[1] >= draft_tokens_block.size:
                        target_tokens = mx.argmax(logits[:, :draft_tokens_block.size, :], axis=-1).squeeze(0)

                        # Check acceptance
                        for i in range(min(len(draft_tokens_block), len(target_tokens))):
                            if draft_tokens_block[i].item() == target_tokens[i].item():
                                acceptance_length += 1
                            else:
                                break

                        logger.debug(f"Acceptance: {acceptance_length}/{len(draft_tokens_block)}")

                        # Rollback cache if not all tokens accepted
                        if acceptance_length < len(draft_tokens_block):
                            # Restore saved state
                            for i, layer_cache in enumerate(target_cache):
                                if hasattr(layer_cache, 'cache'):  # ArraysCache
                                    layer_cache.cache = saved_cache_state[i]
                                else:
                                    layer_cache.state = saved_cache_state[i]

                    # Note: start will be updated after yielding

            # Use target model directly
            last_token_id = output_ids[:, start - 1].item()
            token_input = mx.array([[last_token_id]])
            logits = model(token_input, cache=target_cache)
            mx.eval(logits)
            token = sampler(logits[0, -1:, :])[0].item()

        # Update output_ids
        output_ids[:, start] = token

        # Yield token
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logger.debug(f"Yielding token: {token}")
        yield token, logprobs[:, -1, :].squeeze(0), False
        ntoks += 1

        # Clear cache periodically
        if ntoks % 256 == 0:
            mx.clear_cache()

        if ntoks >= max_tokens:
            break

        # Move start forward
        start += 1

    return
