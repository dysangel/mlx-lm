#!/usr/bin/env python3
"""Direct DFlash generation test script."""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_dflash_v2 import block_diffusion_generate_step
import time

# Load model and tokenizer
print("Loading models...")
model, tokenizer = load(
    "Qwen/Qwen3.5-27B",
)
draft_model, _ = load(
    "z-lab/Qwen3.5-27B-DFlash",
)

# Test prompt
prompt = "What is 2+2?"
max_tokens = 30

print(f"\nPrompt: {prompt}")
print("Generating...\n")

start_time = time.time()
tokens = []
full_text = ""

for token_id, logprobs, from_draft in block_diffusion_generate_step(
    prompt=prompt,
    model=model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    max_tokens=max_tokens,
):
    tokens.append(token_id)
    text = tokenizer.decode(tokens)
    print(f"\r{text}", end="", flush=True)
    full_text = text

elapsed = time.time() - start_time
print(f"\n\nElapsed: {elapsed:.2f}s | Tokens: {len(tokens)} | Speed: {len(tokens)/elapsed:.1f} tok/s")
