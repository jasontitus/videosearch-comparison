#!/usr/bin/env python3
"""Probe the token structure of Gemma 4 inputs to understand what to pool over."""

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load

MODEL_ID = "mlx-community/gemma-4-e4b-it-bf16"

print("Loading model...")
model, processor = load(MODEL_ID)

# --- Single image input ---
img = Image.new("RGB", (384, 384), color=(200, 50, 50))
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n=== PROMPT (raw text) ===\n{repr(prompt)}\n")

inputs = processor(text=prompt, images=[img], return_tensors="np")
input_ids = inputs["input_ids"][0]
tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

print(f"=== TOKEN ANALYSIS (single image) ===")
print(f"Total tokens: {len(tokens)}")
print(f"\nToken sequence:")
for i, (tid, tok) in enumerate(zip(input_ids, tokens)):
    marker = ""
    if "image" in tok.lower():
        marker = " <--- IMAGE TOKEN"
    elif "audio" in tok.lower():
        marker = " <--- AUDIO TOKEN"
    elif "vision" in tok.lower():
        marker = " <--- VISION"
    elif "turn" in tok.lower() or "model" in tok.lower() or "user" in tok.lower():
        marker = " <--- TEMPLATE"
    print(f"  [{i:3d}] id={tid:6d}  {tok}{marker}")

# --- After embedding: which positions are vision tokens? ---
print(f"\n=== EMBEDDING ANALYSIS ===")
input_ids_mx = mx.array(inputs["input_ids"])
pixel_values = mx.array(inputs["pixel_values"])
emb_features = model.get_input_embeddings(input_ids=input_ids_mx, pixel_values=pixel_values)
embeds = emb_features.inputs_embeds if hasattr(emb_features, "inputs_embeds") else emb_features
print(f"inputs_embeds shape: {embeds.shape}")
print(f"Original input_ids length: {len(input_ids)}")
print(f"Embedded sequence length: {embeds.shape[1]}")
print(f"Difference (added vision tokens): {embeds.shape[1] - len(input_ids)}")

# Check if the image placeholder token gets expanded
image_token_id = None
for i, tok in enumerate(tokens):
    if "image" in tok.lower():
        image_token_id = input_ids[i]
        print(f"\nImage placeholder token: id={image_token_id}, position={i}, text='{tok}'")
        break

# --- 5-image input ---
print(f"\n=== 5-IMAGE INPUT ===")
images = [Image.new("RGB", (384, 384), color=c) for c in
          [(200,50,50), (50,200,50), (50,50,200), (200,200,50), (200,50,200)]]
content = [{"type": "image"} for _ in images] + [{"type": "text", "text": "Describe."}]
messages5 = [{"role": "user", "content": content}]
prompt5 = processor.tokenizer.apply_chat_template(messages5, tokenize=False, add_generation_prompt=True)
inputs5 = processor(text=prompt5, images=images, return_tensors="np")
input_ids5 = inputs5["input_ids"][0]
tokens5 = processor.tokenizer.convert_ids_to_tokens(input_ids5.tolist())
print(f"Total tokens (5 images): {len(tokens5)}")

input_ids5_mx = mx.array(inputs5["input_ids"])
pixel_values5 = mx.array(inputs5["pixel_values"])
emb5 = model.get_input_embeddings(input_ids=input_ids5_mx, pixel_values=pixel_values5)
embeds5 = emb5.inputs_embeds if hasattr(emb5, "inputs_embeds") else emb5
print(f"Embedded length (5 images): {embeds5.shape[1]}")
print(f"Original tokens: {len(tokens5)}")
print(f"Vision tokens added: {embeds5.shape[1] - len(tokens5)}")
print(f"Vision tokens per image: {(embeds5.shape[1] - len(tokens5)) / 5:.0f}")

# Find all image token positions
img_positions = [i for i, tok in enumerate(tokens5) if "image" in tok.lower()]
print(f"\nImage placeholder positions in input_ids: {img_positions}")

# --- Text-only (for understanding query embedding) ---
print(f"\n=== TEXT-ONLY INPUT ===")
for text_fmt in [
    "dog",
    "Find video content showing: dog",
    "Describe what is happening: dog",
]:
    msgs = [{"role": "user", "content": text_fmt}]
    p = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=p, return_tensors="np")
    toks = processor.tokenizer.convert_ids_to_tokens(inp["input_ids"][0].tolist())
    print(f"  '{text_fmt}': {len(toks)} tokens")

# --- Raw text (no chat template) ---
print(f"\n=== RAW TEXT (no chat template) ===")
for text in ["dog", "running", "rifle"]:
    inp = processor(text=text, return_tensors="np")
    print(f"  '{text}': {len(inp['input_ids'][0])} tokens")
