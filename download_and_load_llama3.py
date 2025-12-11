# download_and_load_llama3.py
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

MODEL_ID = "meta-llama/Meta-Llama-3-8B"   # change if needed
TARGET_DIR = "/storage/ice1/shared/ece8803cai/andreyh/prompt_method/llama3"
os.makedirs(TARGET_DIR, exist_ok=True)

print("Downloading model files (if not present)...")
snapshot_download(repo_id=MODEL_ID, local_dir=TARGET_DIR, cache_dir=TARGET_DIR)

print("Loading tokenizer from", TARGET_DIR)
tokenizer = AutoTokenizer.from_pretrained(TARGET_DIR, use_fast=True)

print("Loading model (this will use GPU if device_map='auto')...")
model = AutoModelForCausalLM.from_pretrained(
    TARGET_DIR,
    torch_dtype=torch.float16,   # H100 supports fp16 well
    device_map="auto",           # will place weights on GPU (and CPU if needed)
    low_cpu_mem_usage=True
)

print("Model Loaded. cuda available:", torch.cuda.is_available())
# quick inference to test
inputs = tokenizer("Hello, world!", return_tensors="pt").to(next(model.parameters()).device)
out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))
