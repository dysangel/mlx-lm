#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/ali/Projects/dflash')
import torch
import mlx.core as mx
import numpy as np
from dflash.model import DFlashDraftModel as DFlashDraftModelRef
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load as mlx_load

print("Loading models...")
with torch.no_grad():
    draft_ref = DFlashDraftModelRef.from_pretrained('z-lab/Qwen3.5-4B-DFlash')
    target_ref = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', torch_dtype='auto', device_map='cpu')

target_mlx, _ = mlx_load('Qwen/Qwen3.5-4B')
draft_mlx, _ = mlx_load('z-lab/Qwen3.5-4B-DFlash')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')

input_text = "The capital of France is"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids
mask_token_id = draft_ref.mask_token_id
block_size = 3

with torch.no_grad():
    target_output_ref = target_ref(input_ids, output_hidden_states=True)
    target_hidden_ref_raw = torch.cat([target_output_ref.hidden_states[i+1] for i in draft_ref.target_layer_ids], dim=-1)

from mlx_lm.generate_dflash_v2 import ModelWithHiddenStates, get_inner_model
target_with_hidden = ModelWithHiddenStates(target_mlx, draft_mlx.target_layer_ids)
input_ids_mlx = mx.array(input_ids[0].tolist())
target_output_mlx = target_with_hidden(input_ids_mlx[None], cache=None)

hidden_states = target_with_hidden.hidden_states
selected_states = []
for layer_id in draft_mlx.target_layer_ids:
    selected_states.append(hidden_states[layer_id + 1])  # +1 for embedding offset
target_hidden_mlx_raw = mx.concatenate(selected_states, axis=-1)

noise_ids_ref = torch.tensor([[mask_token_id] * (block_size - 1)])
with torch.no_grad():
    noise_embedding_ref = target_ref.model.embed_tokens(noise_ids_ref)

noise_ids_mlx = mx.array([mask_token_id] * (block_size - 1))
target_inner_mlx = get_inner_model(target_mlx)
noise_embedding_mlx = target_inner_mlx.embed_tokens(noise_ids_mlx)[None, ...]

position_ids = torch.arange(0, input_ids.shape[1] + block_size - 1).unsqueeze(0)
with torch.no_grad():
    output_ref = draft_ref(position_ids=position_ids, noise_embedding=noise_embedding_ref, target_hidden=target_hidden_ref_raw)
    logits_ref = target_ref.lm_head(output_ref)

position_ids_mlx = mx.arange(0, input_ids.shape[1] + block_size - 1)[None, :]
output_mlx = draft_mlx(position_ids=position_ids_mlx, noise_embedding=noise_embedding_mlx, target_hidden=target_hidden_mlx_raw, cache=None)
logits_mlx = target_inner_mlx.embed_tokens.as_linear(output_mlx)

top5_ref = torch.topk(logits_ref[0, 0, :], 5)
print(f"Reference top 5 tokens: {top5_ref.indices.tolist()} -> {[tokenizer.decode([t]) for t in top5_ref.indices]}")

logits_mlx_first = logits_mlx[0, 0, :]
top5_mlx_indices = mx.argsort(logits_mlx_first)[-5:]
print(f"MLX top 5 tokens: {top5_mlx_indices.tolist()} -> {[tokenizer.decode([int(t)]) for t in top5_mlx_indices]}")

argmax_ref = torch.argmax(logits_ref[0, 0, :]).item()
argmax_mlx = int(mx.argmax(logits_mlx_first))
print(f"\nArgmax - ref: {argmax_ref} ({tokenizer.decode([argmax_ref])}), mlx: {argmax_mlx} ({tokenizer.decode([argmax_mlx])})")

logits_ref_np = logits_ref[0, 0, :].float().cpu().numpy()
logits_mlx_np = np.array(logits_mlx_first.tolist())
corr = np.corrcoef(logits_ref_np, logits_mlx_np)[0, 1]
print(f"Logits correlation: {corr:.6f}")
