#!/usr/bin/env python3
"""
- Loads TG-ReDial dataset in JSONL format.
- Generates few-shot prompts in Chinese using guidelines + last user message,
  last topic, and ground truth movie if it exists.
- Runs the prompts through a local LLaMA 3 model.
- Saves the dataset with additional keys "llama3_prompt" and "llama3_response" in a new JSONL file.
- ONLY processes the first 5 samples per split.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# ===== CONFIG =====
MODEL_PATH = "/storage/ice1/shared/ece8803cai/andreyh/prompt_method/llama3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.9
NUM_SAMPLES = 5  # Only process first 5 samples per split

# ===== GUIDELINE + FEW-SHOT EXAMPLE =====
GUIDELINE_FEW_SHOT = """
You are a movie recommendation assistant. Your job is to recommend a movie naturally, based on the topic in "Topic". 

Example:
Ground Truth Movie: 精灵鼠小弟(1999)
Topic: 精灵鼠小弟

Dialogue:
User: 说到这，有没有值得回忆的电影推荐一下？
Assistant:
《精灵鼠小弟》怎么样？这是部经典的童年回忆电影，一场充满勇气和智慧的冒险。

Now respond to the user's message below.


"""

# ===== FUNCTIONS =====
def load_jsonl(file_path):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def build_prompt(sample):
    last_user_msg = sample["dialogue_history"][-1]["content"]
    gt_movie = sample.get("ground_truth_movie")  or "None" 
    last_topic = sample["topics_discussed"][-1] if sample["topics_discussed"] else ""

    # Full prompt: guideline + few-shot + current dialogue
    prompt = GUIDELINE_FEW_SHOT.strip() + "\n\n"

    # Insert ground truth movie and topic if they exist
    if gt_movie:
        prompt += f"Ground Truth Movie: {gt_movie}\n"
    if last_topic:
        prompt += f"Topic: {last_topic}\n\n"

    # Add current dialogue
    prompt += "Dialogue:\n"
    prompt += f"User: {last_user_msg}\n"
    prompt += "Assistant:\n\n"

    return prompt

def run_llama_on_prompt(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt_text):].strip()

    # Clean the response: take only up to first double newline
    if "\n\n" in response:
        response = response.split("\n\n")[0].strip()
    return response

def save_jsonl(samples, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ===== MAIN =====
if __name__ == "__main__":
    print(f"Loading LLaMA 3 model from {MODEL_PATH} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.to(DEVICE)

    datasets = {
        "train": "train.jsonl",
        "valid": "valid.jsonl",
        "test": "test.jsonl"
    }

    for split, path in datasets.items():
        path = Path(path)
        if not path.exists():
            print(f"{split} dataset {path} not found, skipping...")
            continue

        print(f"\n=== Processing {split} dataset (first {NUM_SAMPLES} samples only) ===")
        samples = load_jsonl(path)[:NUM_SAMPLES]

        for s in samples:
            prompt = build_prompt(s)
            llama3_resp = run_llama_on_prompt(model, tokenizer, prompt)
            s["llama3_prompt"] = prompt
            s["llama3_response"] = llama3_resp

        out_file = f"{split}_llama3.jsonl"
        save_jsonl(samples, out_file)
        print(f"Wrote {len(samples)} samples with LLaMA 3 prompts & cleaned responses to {out_file}")
