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

    logger.debug(f"Target cache types: {[type(c).__name__ for c in target_cache[:3]]}")
    logger.debug(f"Draft cache types: {[type(c).__name__ for c in draft_cache[:3]]}")

    # Prefill stage - capture context features from target model
    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)

    # Check initial cache state
    initial_offset = None
    for c in target_cache:
        if hasattr(c, 'offset'):
            initial_offset = c.offset
            break
    logger.debug(f"Initial KVCache offset: {initial_offset}")

    # Process prompt tokens
    prompt_tokens = prompt_tokens[None, :]
    output = target_model_with_hidden(prompt_tokens, cache=target_cache)

    # Check cache after prefill
    prefill_offset = None
    for c in target_cache:
        if hasattr(c, 'offset'):
            prefill_offset = c.offset
            break
    logger.debug(f"After prefill KVCache offset: {prefill_offset}")

    # Save the initial prefill hidden states for accumulation
    accumulated_hidden = target_model_with_hidden.hidden_states.copy()

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

        # Check cache state at start of iteration
        kvcache_offset = None
        for c in target_cache:
            if hasattr(c, 'offset'):
                kvcache_offset = c.offset
                break
        if iteration <= 5:
            logger.debug(f"Start of iteration {iteration}: KVCache offset = {kvcache_offset}")

        with mx.stream(generation_stream):
            # Generate draft tokens (but don't use them yet)
            if iteration > 1:
                prev_token = output_ids[:, start - 1][:, None]
                noise_tokens = mx.zeros([1, max(0, current_block_size - 1)], dtype=mx.uint32)
                draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1) if current_block_size > 1 else prev_token
                noise_embedding = target_inner.embed_tokens(draft_input)

                # Use draft cache (now re-materialized with correct context)
                cache_size = draft_cache.total_size()

                # Get target position for comparison
                target_position = None
                for c in target_cache:
                    if hasattr(c, 'offset'):
                        target_position = c.offset
                        break

                noise_position_ids = mx.arange(cache_size, cache_size + current_block_size)[None, :]

                # Debug: check position IDs
                if iteration <= 3:
                    logger.debug(f"noise_position_ids: {noise_position_ids.tolist()}")
                    logger.debug(f"cache_size (draft): {cache_size}, target_offset: {target_position}")

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
                if draft_logits.shape[1] > 0:
                    draft_tokens_block = mx.argmax(draft_logits[:, :, :], axis=-1).squeeze(0)
                else:
                    draft_tokens_block = mx.array([], dtype=mx.uint32)

                # Debug: check first draft token
                if draft_logits.shape[1] > 0:
                    first_draft_token = mx.argmax(draft_logits[:, 0, :]).item()
                    logger.debug(f"First draft token from logit[0]: {first_draft_token}")

                logger.debug(f"Draft tokens (first 5): {draft_tokens_block[:5].tolist() if draft_tokens_block.size > 0 else []}")

                # Verify draft tokens
                acceptance_length = 0
                if draft_tokens_block.size > 0:
                    # Save ArraysCache conv_state before running draft tokens
                    # conv_state is a rolling window that gets overwritten - must save/restore
                    arrays_cache_state = []
                    for i, c in enumerate(target_cache):
                        if hasattr(c, 'cache') and c.cache[0] is not None:  # ArraysCache with data
                            arrays_cache_state.append((i, c.cache[0], c.cache[1]))

                    # Run draft tokens through target model WITH cache
                    # The model will write at the correct offset automatically
                    target_verify_input = draft_tokens_block[None, :]

                    # Debug: check cache state before
                    arraycache_size_before = None
                    for c in target_cache:
                        if hasattr(c, 'cache') and c.cache[0] is not None:
                            arraycache_size_before = c.cache[0].shape[1]
                            break

                    logger.debug(f"ArraysCache size before verify: {arraycache_size_before}")

                    logits = model(target_verify_input, cache=target_cache)
                    mx.eval(logits)

                    # Check cache state after
                    arraycache_size_after = None
                    for c in target_cache:
                        if hasattr(c, 'cache') and c.cache[0] is not None:
                            arraycache_size_after = c.cache[0].shape[1]
                            break

                    logger.debug(f"ArraysCache size after verify: {arraycache_size_after}")

                    # Get target predictions and check acceptance
                    if logits.ndim == 3 and logits.shape[1] >= draft_tokens_block.size:
                        target_tokens = mx.argmax(logits[:, :draft_tokens_block.size, :], axis=-1).squeeze(0)

                        # Debug: check first few tokens in detail
                        logger.debug(f"Draft[0]={draft_tokens_block[0].item()}, Target[0]={target_tokens[0].item()}, Match={draft_tokens_block[0].item() == target_tokens[0].item()}")
                        if draft_tokens_block.size > 1:
                            logger.debug(f"Draft[1]={draft_tokens_block[1].item()}, Target[1]={target_tokens[1].item()}, Match={draft_tokens_block[1].item() == target_tokens[1].item()}")

                        # Check acceptance
                        for i in range(min(len(draft_tokens_block), len(target_tokens))):
                            if draft_tokens_block[i].item() == target_tokens[i].item():
                                acceptance_length += 1
                            else:
                                break

                        logger.debug(f"Draft (first 5): {draft_tokens_block[:5].tolist()}")
                        logger.debug(f"Target (first 5): {target_tokens[:5].tolist()}")
                        logger.debug(f"Acceptance: {acceptance_length}/{len(draft_tokens_block)}")

                        # Rollback cache for rejected tokens
                        if acceptance_length < len(draft_tokens_block):
                            num_rejected = len(draft_tokens_block) - acceptance_length

                            # Trim KVCache layers
                            for c in target_cache:
                                if hasattr(c, 'trim') and not hasattr(c, 'cache'):
                                    c.trim(num_rejected)

                            # Restore ArraysCache conv_state from saved references
                            # Model creates new arrays with mx.contiguous(), so old refs are safe to restore
                            for idx, saved_k, saved_v in arrays_cache_state:
                                target_cache[idx].cache[0] = saved_k
                                target_cache[idx].cache[1] = saved_v

                            logger.debug(f"Trimmed KVCache and restored {len(arrays_cache_state)} ArraysCache layers")

            # Use target model directly (via wrapper to keep hidden_states updated)
            last_token_id = output_ids[:, start - 1].item()
            token_input = mx.array([[last_token_id]])
            output = target_model_with_hidden(token_input, cache=target_cache)
            token = sampler(output.logits[0, -1:, :])[0].item()
            logits = output.logits

            # Accumulate hidden states (only for accepted target tokens, not draft verification)
            for i, h in enumerate(target_model_with_hidden.hidden_states):
                if i < len(accumulated_hidden):
                    accumulated_hidden[i] = mx.concatenate([accumulated_hidden[i], h], axis=1)
                else:
                    accumulated_hidden.append(h)

            # Regenerate draft cache with ACCUMULATED target hidden states
            current_target_hidden = extract_context_feature(
                accumulated_hidden,
                draft_model.target_layer_ids,
            )

            # Get current context length from accumulated_hidden
            ctx_len = current_target_hidden.shape[1]
            ctx_position_ids = mx.arange(ctx_len)[None, :]

            # Reset draft cache and re-materialize
            draft_cache = draft_model.make_cache()
            draft_model.materialize_target_hidden(current_target_hidden, draft_cache, ctx_position_ids)
            draft_cache.update_noise_start(ctx_len)

            if iteration <= 3:
                target_offset = next((c.offset for c in target_cache if hasattr(c, 'offset')), None)
                logger.debug(f"Re-materialize: accumulated_hidden.shape={current_target_hidden.shape}, target_cache.offset={target_offset}")

            # Check first cache layer keys shape (always)
            first_cache = draft_cache.caches[0]
            if first_cache.keys is not None:
                if iteration <= 3:
                    logger.debug(f"First draft cache keys.shape: {first_cache.keys.shape}")
            else:
                logger.debug(f"ERROR: First draft cache keys is None after materialization!")

            if iteration <= 3:
                logger.debug(f"Re-materialized draft cache with ctx_len={ctx_len}, hidden.shape={current_target_hidden.shape}")
                logger.debug(f"Draft cache size after materialization: {draft_cache.total_size()}")

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
