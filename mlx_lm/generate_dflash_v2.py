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
    ctx_len = target_hidden.shape[1]

    # Get inner model for embeddings
    target_inner = get_inner_model(model)

    # Initialize output_ids like the reference
    max_length = num_input_tokens + max_tokens + block_size
    output_ids = mx.full([1, max_length], mask_token_id, dtype=mx.uint32)
    output_ids[:, :num_input_tokens] = prompt_tokens
    output_ids[:, num_input_tokens] = first_token

    # all_accepted_tokens is used for target_hidden reconstruction
    # It's rebuilt each iteration from output_ids which contains the full sequence
    all_accepted_tokens = []

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
            # Generate draft tokens (iteration 2+)
            # Skip draft if current_block_size <= 1 (nothing to draft)
            if iteration > 1 and current_block_size > 1:
                prev_token = output_ids[:, start - 1][:, None]
                # Use mask_token_id for noise positions (reference: output_ids initialized with mask)
                noise_tokens = mx.full([1, max(0, current_block_size - 1)], mask_token_id, dtype=mx.uint32)
                draft_input = mx.concatenate([prev_token, noise_tokens], axis=-1) if current_block_size > 1 else prev_token
                noise_embedding = target_inner.embed_tokens(draft_input)

                # Use draft cache (now re-materialized with correct context)
                # Position IDs should be ABSOLUTE positions in the full sequence
                # Use `start` (current position in output_ids) not ctx_len
                noise_position_ids = mx.arange(start, start + current_block_size)[None, :]

                # Call draft model with target_hidden
                # Re-create draft cache each iteration (reference uses cropped cache)
                draft_cache = draft_model.make_cache()
                draft_output = draft_model(
                    position_ids=noise_position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    cache=draft_cache,
                )
                mx.eval(draft_output)

                # Get draft logits
                if hasattr(model, 'lm_head'):
                    draft_logits = model.lm_head(draft_output)
                else:
                    draft_logits = target_inner.embed_tokens.as_linear(draft_output)
                mx.eval(draft_logits)

                # Sample draft tokens (skip first position which is for the seed token)
                if draft_logits.shape[1] > current_block_size - 1:
                    draft_tokens_block = mx.argmax(draft_logits[:, -current_block_size + 1:, :], axis=-1).squeeze(0)
                else:
                    draft_tokens_block = mx.array([], dtype=mx.uint32)

                logger.debug(f"Draft tokens (first 5): {draft_tokens_block[:5].tolist() if draft_tokens_block.size > 0 else []}")

                # Verify draft tokens WITH cache (using SpeculativeArraysCache checkpoint/rollback)
                acceptance_length = 0
                if draft_tokens_block.size > 0:
                    # Save checkpoint before verification (for rollback on rejection)
                    for c in target_cache:
                        if hasattr(c, 'save_checkpoint'):
                            c.save_checkpoint()

                    # Construct verification input: seed + draft tokens
                    verification_input = mx.concatenate([prev_token, draft_tokens_block[None, :]], axis=-1)
                    # Verify WITH cache - SpeculativeArraysCache supports rollback
                    output = target_model_with_hidden(verification_input, cache=target_cache)
                    logits = output.logits
                    mx.eval(logits)

                    # Get target predictions (exclude last position)
                    target_tokens = mx.argmax(logits[:, :-1, :], axis=-1).squeeze(0)

                    # Debug: compare draft vs target
                    if logger.isEnabledFor(logging.DEBUG):
                        draft_decoded = [tokenizer.decode([t]) for t in draft_tokens_block[:5].tolist()]
                        target_decoded = [tokenizer.decode([t]) for t in target_tokens[:5].tolist()]
                        logger.debug(f"Draft (first 5): {draft_tokens_block[:5].tolist()} -> {draft_decoded}")
                        logger.debug(f"Target (first 5): {target_tokens[:5].tolist()} -> {target_decoded}")

                    # Find acceptance length (compare draft_tokens_block against target_tokens)
                    acceptance_length = (
                        mx.cumsum(draft_tokens_block == target_tokens) == mx.arange(len(target_tokens))
                    ).sum()
                    acceptance_length = int(acceptance_length)

                    logger.debug(f"Acceptance: {acceptance_length}/{len(draft_tokens_block)}")

                    # Handle cache rollback/commit based on acceptance
                    if acceptance_length < len(draft_tokens_block):
                        # Some tokens rejected - rollback cache to checkpoint
                        for c in target_cache:
                            if hasattr(c, 'rollback'):
                                c.rollback()
                        num_rejected = len(draft_tokens_block) - acceptance_length
                        logger.debug(f"Rolled back {num_rejected} rejected tokens, cache at position {start}")
                    else:
                        # All tokens accepted - commit cache
                        new_position = start + acceptance_length
                        for c in target_cache:
                            if hasattr(c, 'commit'):
                                c.commit(new_position)
                        logger.debug(f"All tokens accepted, committed cache to position {new_position}")

                    logger.debug(f"Verified with cache, acceptance: {acceptance_length}")

                    # Update output_ids with accepted draft tokens
                    for i in range(acceptance_length):
                        token_id = draft_tokens_block[i].item()
                        output_ids[:, start + i] = token_id
                        # draft_tokens_block[i] corresponds to draft_logits[:, i + 1, :]
                        # because we sampled from draft_logits[:, -current_block_size + 1:, :]
                        logprobs = draft_logits[:, i + 1, :]
                        logger.debug(f"Yielding draft token: i={i}, token_id={token_id}, draft_logits.shape={draft_logits.shape}, logprobs.shape={logprobs.shape}")
                        if logprobs.ndim == 3 and logprobs.shape[1] == 1:
                            logprobs = logprobs.squeeze(1)
                        yield token_id, logprobs, True
                        ntoks += 1
                        if ntoks >= max_tokens:
                            break

                    if ntoks >= max_tokens:
                        break

                    # Sample and yield target token
                    target_token = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0)
                    output_ids[:, start + acceptance_length] = target_token
                    yield target_token.item(), logits[:, -1, :], False
                    ntoks += 1

                    if ntoks >= max_tokens:
                        break

                    # Build all_accepted_tokens for target_hidden rebuild
                    # Need ALL tokens so far: prompt + all generated tokens
                    # Use output_ids which has the complete sequence
                    total_tokens = start + acceptance_length + 1  # +1 includes the target token we just added
                    all_accepted_tokens = output_ids[:, :total_tokens].squeeze(0).tolist()

                    # Rebuild target_hidden from ALL tokens (prompt + all generated)
                    # This gives draft model full context for next iteration
                    all_tokens_array = mx.array(all_accepted_tokens)[None, :]

                    # Only rebuild cache if tokens were rejected (rollback happened)
                    # If all tokens were accepted, cache is already up-to-date from verification
                    if acceptance_length < len(draft_tokens_block):
                        # Some tokens were rejected and rolled back - need to rebuild cache
                        cache_update_output = target_model_with_hidden(all_tokens_array, cache=target_cache)
                        mx.eval(cache_update_output.logits)
                        # Commit the rebuilt cache
                        new_position = total_tokens
                        for c in target_cache:
                            if hasattr(c, 'commit'):
                                c.commit(new_position)
                        logger.debug(f"Rebuilt cache to position {new_position} after partial acceptance")
                        # Use verification output for target_hidden
                        all_hidden_output = output
                    else:
                        # All tokens accepted - cache is already up-to-date, rebuild target_hidden without cache
                        all_hidden_output = target_model_with_hidden(all_tokens_array, cache=None)
                        mx.eval(all_hidden_output.logits)

                    mx.eval(all_hidden_output.logits)

                    # Extract target_hidden from all tokens
                    target_hidden = extract_context_feature(
                        all_hidden_output.hidden_states,
                        draft_model.target_layer_ids,
                    )
                    mx.eval(target_hidden)

                    # Update ctx_len
                    ctx_len = target_hidden.shape[1]

                    # Advance start by acceptance_length + 1 (like reference)
                    start += acceptance_length + 1

                    if ntoks >= max_tokens:
                        break
            else:
                # Iteration 1: Generate one target token directly (no draft)
                prev_token = mx.array([[first_token]])
                output = target_model_with_hidden(prev_token, cache=target_cache)
                logits = output.logits
                mx.eval(logits)

                # Sample and yield target token
                target_token = mx.argmax(logits[:, -1, :], axis=-1).squeeze(0)
                output_ids[:, start] = target_token
                yield target_token.item(), logits[:, -1, :], False
                ntoks += 1

                if ntoks >= max_tokens:
                    break

                # Rebuild target_hidden from ALL tokens (prompt + first_token + this target)
                # Consistent with iteration 2+ approach
                total_tokens = start + 1  # +1 for the target token we just added
                all_accepted_tokens = output_ids[:, :total_tokens].squeeze(0).tolist()
                all_tokens_array = mx.array(all_accepted_tokens)[None, :]
                all_hidden_output = target_model_with_hidden(all_tokens_array, cache=None)
                mx.eval(all_hidden_output.logits)

                # Extract target_hidden from all tokens
                target_hidden = extract_context_feature(
                    all_hidden_output.hidden_states,
                    draft_model.target_layer_ids,
                )
                mx.eval(target_hidden)

                # Update ctx_len
                ctx_len = target_hidden.shape[1]

                # Advance start by 1 (only 1 token generated)
                start += 1

                # Crop target_cache to start (ensure consistency)
                for c in target_cache:
                    if hasattr(c, 'trim') and hasattr(c, 'offset') and c.offset > start:
                        c.trim(c.offset - start)

                if ntoks >= max_tokens:
                    break

        # Clear cache periodically
        if ntoks % 256 == 0:
            mx.clear_cache()

        if ntoks >= max_tokens:
            break

    return
