# Copyright © 2025 Apple Inc.

"""Block diffusion speculative decoding using DFlash draft model - Reference implementation port."""

from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.cache import KVCache, trim_prompt_cache
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
    # Use offset=1 to account for the embedding at index 0
    # hidden_states[layer_id + offset] gives us the output of layer at layer_id
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
    ) -> mx.array:
        """Forward pass that captures intermediate hidden states."""
        self.hidden_states = []

        # Get the inner model
        inner_model = self.model
        if hasattr(inner_model, 'language_model'):
            inner_model = inner_model.language_model
        if hasattr(inner_model, 'model'):
            inner_model = inner_model.model

        # Get embeddings
        h = inner_model.embed_tokens(inputs)
        self.hidden_states.append(h)

        # Create cache if needed
        if cache is None:
            if hasattr(inner_model, 'make_cache'):
                cache = inner_model.make_cache()
            else:
                cache = [None] * len(inner_model.layers)

        # Process through layers - capture ALL layer outputs like reference
        mask = None  # Use default causal masking
        for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
            h = layer(h, mask, c)
            self.hidden_states.append(h)  # Capture ALL layers

        # Final normalization
        h = inner_model.norm(h)

        # Get logits
        if hasattr(self.model.args, "tie_word_embeddings") and self.model.args.tie_word_embeddings:
            logits = inner_model.embed_tokens.as_linear(h)
        elif hasattr(self.model, 'lm_head'):
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

    Yields:
        Tuple of (token_id, logprobs, from_draft)
    """
    from .generation import _process_and_sample

    # Get sampler
    sampler = kwargs.get("sampler")
    if sampler is None:
        sampler = make_sampler(
            temp=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            min_p=kwargs.get("min_p", 0.0),
            min_tokens_to_keep=kwargs.get("min_tokens_to_keep", 1),
        )

    # Get cache quantization function
    quantize_cache_fn = kwargs.get("quantize_cache_fn", lambda x: None)

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
    model_cache = model.make_cache()
    draft_cache = draft_model.make_cache()

    # Prefill stage - capture context features from target model
    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)

    # Process prompt tokens
    prompt_tokens = prompt_tokens[None, :]
    logits = target_model_with_hidden(prompt_tokens, cache=model_cache)
    tokens, logprobs = _process_and_sample(None, logits.squeeze(0))

    # Yield first token
    first_token = tokens[-1].item()
    first_logprobs = logprobs[-1:]
    yield first_token, first_logprobs, False
    ntoks = 1

    # Extract context features from target model
    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )

    # Get inner model for embeddings
    target_inner = get_inner_model(model)

    # Decode loop
    while ntoks < max_tokens:
        with mx.stream(generation_stream):
            remaining = max_tokens - ntoks
            current_block_size = min(block_size, remaining)

            # Create position_ids for noise tokens
            # Position starts after the input prompt
            noise_start_pos = num_input_tokens + ntoks
            position_ids = mx.arange(noise_start_pos, noise_start_pos + current_block_size)[None, :]

            # Create mask token embeddings
            block_ids = mx.full([current_block_size], mask_token_id, dtype=mx.uint32)
            noise_embedding = target_inner.embed_tokens(block_ids)[None, ...]

            # Draft model generates
            draft_output = draft_model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            quantize_cache_fn(draft_cache)

            # Get draft logits using target model's lm_head
            if hasattr(model, 'lm_head'):
                draft_logits = model.lm_head(draft_output)
            else:
                draft_logits = target_inner.embed_tokens.as_linear(draft_output)

            # Sample draft tokens (skip first position which is for the seed token)
            draft_tokens_block = sample(draft_logits[:, -current_block_size + 1:, :], sampler).squeeze(0)

            # Prepend first token from previous iteration
            draft_tokens = mx.concatenate([mx.array([first_token]), draft_tokens_block])

            # Target model verifies draft tokens
            target_output = target_model_with_hidden(draft_tokens[None], cache=model_cache)
            quantize_cache_fn(model_cache)

            # Use argmax for verification
            target_tokens_block = mx.argmax(target_output.logits, axis=-1).squeeze(0)

            # Sample from target for actual output
            target_tokens_sampled = sample(target_output.logits, sampler).squeeze(0)

            # Find acceptance length
            draft_to_verify = draft_tokens[1:]
            target_values = target_tokens_block[:-1]
            acceptance_length = (
                mx.cumsum(draft_to_verify == target_values) == mx.arange(1, current_block_size)
            ).sum()
            acceptance_length = int(acceptance_length)

            # Trim caches
            num_rejected = current_block_size - 1 - acceptance_length
            if num_rejected > 0:
                trim_prompt_cache(model_cache, num_rejected)
                trim_prompt_cache(draft_cache, num_rejected)

            # Yield accepted draft tokens
            for i in range(acceptance_length):
                yield draft_tokens[i + 1].item(), draft_logits[:, i, :], True
                ntoks += 1
                if ntoks >= max_tokens:
                    break

            if ntoks >= max_tokens:
                break

            # Yield one target token
            target_token = target_tokens_sampled[acceptance_length]
            yield target_token.item(), target_output.logits[:, acceptance_length, :], False
            ntoks += 1

            if ntoks >= max_tokens:
                break

            # Update first_token
            first_token = target_token

            # Update target_hidden with new token
            new_token = mx.array([target_tokens_sampled[acceptance_length]])
            _ = target_model_with_hidden(new_token[None], cache=model_cache)
            target_hidden = extract_context_feature(
                target_model_with_hidden.hidden_states,
                draft_model.target_layer_ids,
            )
