# convert_to_gguf.py
#
# Simple model export (GGUF conversion alternative)
# Run: python convert_to_gguf.py

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# -------- CONFIG --------
HF_MODEL_DIR  = "tinyLLM"
OUTPUT_DIR    = "tinyLLM_exported"
# ------------------------

print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained(HF_MODEL_DIR)
tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_DIR)

print("Saving model in safetensors format...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Model exported successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print("\nNote: For GGUF format, you can use:")
print("  pip install gguf")
print("  Or use llama.cpp's convert_hf_to_gguf.py once it's available")
