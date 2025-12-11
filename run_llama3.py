#!/usr/bin/env python3
"""
Checkpointed LLaMA runner for TG-ReDial JSONL
- Processes train/valid/test (order controllable by --order)
- Saves progress every N prompts and can resume from last checkpoint
- Appends per-split output JSONL (e.g. test_llama3.jsonl)
- Shows tqdm progress bar
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import os

# ====== CONFIG ======
MODEL_PATH = "/storage/ice1/shared/ece8803cai/andreyh/prompt_method/llama3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.9
CHECKPOINT_INTERVAL = 100  # default, can be overridden by CLI
PROMPT_SAVE_FLUSH = True

GUIDELINE_FEW_SHOT = """
你是一个电影推荐助手。你的任务是根据“Topic”自然地推荐电影，不要一开始就直接说电影名字。

示例:
Ground Truth Movie: 精灵鼠小弟(1999)
Topic: 精灵鼠小弟

对话:
User: 说到这，有没有值得回忆的电影推荐一下？
Assistant:
《精灵鼠小弟》怎么样？这是部经典的童年回忆电影，一场充满勇气和智慧的冒险。

请根据以下对话做出自然的推荐：
"""

# ====== HELPERS ======
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_prompt(sample):
    last_user_msg = sample["dialogue_history"][-1]["content"]
    gt_movie = sample.get("ground_truth_movie") or "None"
    last_topic = sample["topics_discussed"][-1] if sample.get("topics_discussed") else ""
    prompt = GUIDELINE_FEW_SHOT.strip() + "\n\n"
    prompt += f"Ground Truth Movie: {gt_movie}\n"
    prompt += f"Topic: {last_topic}\n\n"
    prompt += "对话:\n"
    prompt += f"User: {last_user_msg}\nAssistant:\n\n"
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
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract only the generated portion (after the prompt)
    response = full_text[len(prompt_text):].strip()
    # post-clean: stop at first empty line or special marker
    if "\n\n" in response:
        response = response.split("\n\n")[0].strip()
    return response

def append_jsonl_line(path: Path, obj: dict):
    # append one JSON line and flush
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if PROMPT_SAVE_FLUSH:
            f.flush()
            os.fsync(f.fileno())

def load_checkpoint(checkpoint_path: Path):
    if checkpoint_path.exists():
        try:
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_checkpoint(checkpoint_path: Path, ckpt_obj: dict):
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt_obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(checkpoint_path)

# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", nargs="+", default=["train", "valid", "test"],
                        help="Order of datasets to process (e.g. --order test train valid)")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing train.jsonl/valid.jsonl/test.jsonl")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Save progress every N samples")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Process at most N samples per split (-1 = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_interval = args.checkpoint_interval
    max_samples = args.max_samples

    print(f"Loading LLaMA model from {MODEL_PATH} on {DEVICE}...")
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
        "train": data_dir / "train.jsonl",
        "valid": data_dir / "valid.jsonl",
        "test":  data_dir / "test.jsonl"
    }

    # checkpoint file (global for all splits)
    checkpoint_path = data_dir / "llama3_checkpoints.json"
    ckpt = load_checkpoint(checkpoint_path)  # dict: {split: last_index_processed}

    for split in args.order:
        src_path = datasets.get(split)
        if not src_path or not src_path.exists():
            print(f"Skipping {split}: {src_path} not found")
            continue

        print(f"\n=== Processing {split} dataset ===")
        samples = load_jsonl(src_path)
        total = len(samples) if max_samples < 0 else min(len(samples), max_samples)

        # output file and checkpoint index for this split
        out_file = data_dir / f"{split}_llama3.jsonl"
        last_processed = ckpt.get(split, -1)  # index (inclusive) of last processed
        start_idx = last_processed + 1
        if start_idx >= total:
            print(f"{split}: already done (start {start_idx} >= total {total}), skipping.")
            continue

        # ensure output file exists (if resuming, leave existing contents)
        out_file.touch(exist_ok=True)

        # main loop with tqdm
        with tqdm(total=total, desc=f"Processing {split}", ncols=100) as pbar:
            # set initial progress to start_idx
            pbar.update(start_idx)

            try:
                for i in range(start_idx, total):
                    s = samples[i]
                    prompt = build_prompt(s)

                    # run model
                    try:
                        resp = run_llama_on_prompt(model, tokenizer, prompt)
                    except Exception as e:
                        # retry once after a short sleep (handles occasional GPU OOM/slurm hiccups)
                        print(f"\nWarning: generation failed at {split}:{i} -> {e}. Retrying after 3s...")
                        time.sleep(3)
                        resp = run_llama_on_prompt(model, tokenizer, prompt)

                    # attach and append (so partial output is on disk)
                    s_out = dict(s)  # shallow copy
                    s_out["llama3_prompt"] = prompt
                    s_out["llama3_response"] = resp
                    append_jsonl_line(out_file, s_out)

                    # update checkpoint in memory
                    ckpt[split] = i
                    # periodically persist checkpoint
                    if (i - start_idx + 1) % checkpoint_interval == 0:
                        save_checkpoint(checkpoint_path, ckpt)

                    pbar.update(1)

                # finished split: persist final checkpoint
                ckpt[split] = total - 1
                save_checkpoint(checkpoint_path, ckpt)
                print(f"{split}: completed {total} samples. output -> {out_file}")

            except KeyboardInterrupt:
                print("\nInterrupted by user — saving checkpoint and exiting.")
                save_checkpoint(checkpoint_path, ckpt)
                return
            except Exception as e:
                print(f"\nFatal error while processing {split}:{e}. Saving checkpoint and exiting.")
                save_checkpoint(checkpoint_path, ckpt)
                return

if __name__ == "__main__":
    main()
