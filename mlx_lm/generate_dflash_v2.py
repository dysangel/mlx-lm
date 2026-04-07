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

        # Get the inner model - need to handle different model architectures
        # Some models have: Model -> language_model(TextModel) -> model(Qwen3_5TextModel) -> layers
        # Others have: Model -> language_model(TextModel) -> layers
        inner_model = self.model
        if hasattr(inner_model, 'language_model'):
            inner_model = inner_model.language_model
        # Store reference to the potential lm_head container
        lm_head_container = inner_model
        # Navigate to the actual model with layers
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

        # Get logits - check lm_head_container for tie_word_embeddings and lm_head
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

    Yields:
        Tuple of (token_id, logprobs, from_draft)
    """
    # Helper function for sampling
    def process_sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tokens = sampler(logprobs)
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
    # Don't reuse draft_cache across blocks to avoid position mismatch
    # Each block creates its own cache
    draft_cache = None

    # Prefill stage - capture context features from target model
    target_model_with_hidden = ModelWithHiddenStates(model, draft_model.target_layer_ids)

    # Process prompt tokens
    prompt_tokens = prompt_tokens[None, :]
    output = target_model_with_hidden(prompt_tokens, cache=model_cache)
    tokens, logprobs = process_sample(output.logits.squeeze(0))

    # Yield first token
    first_token = tokens[-1].item()
    first_logprobs = logprobs[-1]
    yield first_token, first_logprobs, False
    ntoks = 1

    # Extract context features from target model
    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )

    # Get inner model for embeddings
    target_inner = get_inner_model(model)

    # Track all generated tokens for rebuilding target_hidden
    accumulated_tokens = list(prompt_tokens.squeeze(0).tolist()) + [first_token]

    # Rebuild target_hidden from full accumulated sequence (prompt + first_token)
    accumulated_tokens_mx = mx.array(accumulated_tokens)[None, :]
    _ = target_model_with_hidden(accumulated_tokens_mx, cache=None)
    target_hidden = extract_context_feature(
        target_model_with_hidden.hidden_states,
        draft_model.target_layer_ids,
    )

    # Decode loop - moved outside stream block
    while ntoks < max_tokens:
        remaining = max_tokens - ntoks
        current_block_size = min(block_size, remaining)

        with mx.stream(generation_stream):
            # Create position_ids for the draft model
            # In the reference, position_ids are GLOBAL positions in the output sequence
            # The noise tokens start at position len(accumulated_tokens)
            # The position_ids should cover ALL positions that K will attend to (context + noise)
            # But the noise tokens are at positions [len(accumulated_tokens), len(accumulated_tokens) + block_size)
            # And the context is at positions [0, len(accumulated_tokens))

            # The key insight: position_ids should start from 0 and cover all positions
            # But the noise tokens' GLOBAL position is len(accumulated_tokens)
            # So we need position_ids = [0, 1, ..., len(accumulated_tokens) + block_size - 1]

            # Actually, looking at the reference more carefully:
            # position_ids = position_ids[:, start:start + block_size] where start = len(output_ids) in global sequence
            # This gives position_ids for the noise tokens at their global positions

            # But the draft model also needs position embeddings for the context tokens (in K)
            # So we need position_ids for ALL positions: [0, len(accumulated_tokens) + block_size)

            global_pos = len(accumulated_tokens)
            total_positions = global_pos + current_block_size
            position_ids = mx.arange(0, total_positions)[None, :]

            # Create mask token embeddings
            block_ids = mx.full([current_block_size], mask_token_id, dtype=mx.uint32)
            noise_embedding = target_inner.embed_tokens(block_ids)[None, ...]

            # Draft model generates
            draft_output = draft_model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=None,  # Fresh cache for each block
            )

            # Get draft logits using target model's lm_head
            if hasattr(model, 'lm_head'):
                draft_logits = model.lm_head(draft_output)
            else:
                draft_logits = target_inner.embed_tokens.as_linear(draft_output)

            # Sample draft tokens (skip first position which is for the seed token)
            # The draft model outputs block_size positions, where position 0 is unused
            # and positions 1 to block_size-1 contain the actual draft predictions
            draft_tokens_block = sampler(draft_logits[:, -current_block_size + 1:, :]).squeeze(0)

            # For verification, we need: [first_token, draft_token_0, draft_token_1, ...]
            # The target model will predict next tokens for each position
            draft_tokens_for_target = mx.concatenate([mx.array([first_token]), draft_tokens_block])

            # Convert draft_logits to logprobs for yielding
            draft_logprobs = draft_logits - mx.logsumexp(draft_logits, axis=-1, keepdims=True)

            # Target model verifies draft tokens
            target_output = target_model_with_hidden(draft_tokens_for_target[None], cache=model_cache)
            quantize_cache_fn(model_cache)

            # Use argmax for verification
            target_tokens_block = mx.argmax(target_output.logits, axis=-1).squeeze(0)

            # Sample from target for actual output
            target_tokens_sampled = sampler(target_output.logits).squeeze(0)

            # Find acceptance length
            # Compare: draft_tokens_block[0,1,2,...] vs target_tokens_block[1,2,3,...]
            # i.e., compare each draft token with target's prediction at that position
            acceptance_length = (
                mx.cumsum(draft_tokens_block == target_tokens_block[:-1]) == mx.arange(1, len(draft_tokens_block) + 1)
            ).sum()
            acceptance_length = int(acceptance_length)

            # Trim caches
            num_rejected = current_block_size - 1 - acceptance_length
            if num_rejected > 0:
                trim_prompt_cache(model_cache, num_rejected)

            # Yield accepted draft tokens
            for i in range(acceptance_length):
                yield draft_tokens_block[i].item(), draft_logprobs[:, i, :].squeeze(0), True
                ntoks += 1
                if ntoks >= max_tokens:
                    break

            if ntoks >= max_tokens:
                break

            # Yield one target token
            target_token = target_tokens_sampled[acceptance_length]
            target_logprobs = target_output.logits - mx.logsumexp(target_output.logits, axis=-1, keepdims=True)
            yield target_token.item(), target_logprobs[:, acceptance_length, :].squeeze(0), False
            ntoks += 1

            if ntoks >= max_tokens:
                break

            # Update first_token for next iteration
            first_token = target_token

            # Accumulate all generated tokens and update target_hidden incrementally
            # Add accepted draft tokens
            for i in range(acceptance_length):
                accumulated_tokens.append(draft_tokens_block[i].item())
            # Add the target token
            accumulated_tokens.append(target_token.item())

            # Incremental update: extract only new hidden states from target_output
            # target_output.hidden_states contains hidden states for all draft_tokens positions
            # Position 0 is the seed token, positions 1..acceptance_length are accepted draft tokens
            # Position acceptance_length is the target token we just sampled

            # Extract hidden states for accepted draft tokens (positions 1 to acceptance_length)
            new_hidden_states = []
            for layer_id in draft_model.target_layer_ids:
                # hidden_states has embedding at index 0, layers at 1..33
                # We need to get the layer at layer_id
                layer_hidden = target_model_with_hidden.hidden_states[layer_id + 1]
                # Extract positions 1..acceptance_length (accepted draft tokens)
                accepted_hidden = layer_hidden[:, 1:acceptance_length + 1, :]
                # Extract position acceptance_length (target token)
                target_hidden_single = layer_hidden[:, acceptance_length:acceptance_length + 1, :]
                # Concatenate: accepted + target
                new_hidden = mx.concatenate([accepted_hidden, target_hidden_single], axis=1)
                new_hidden_states.append(new_hidden)

            # Concatenate all layers and append to target_hidden
            if new_hidden_states:
                new_target_hidden = mx.concatenate(new_hidden_states, axis=-1)
                target_hidden = mx.concatenate([target_hidden, new_target_hidden], axis=1)
