#!/usr/bin/env python3
"""

Generate few-shot prompts for LLaMA3 from TG-ReDial JSONL datasets.
Uses:
- Last user message
- Last topic in topics_discussed
- Ground truth movie if it exists
- Adds the few-shot example
"""

import argparse
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Sample:
    conversation_id: int
    message_id: int
    last_user_msg: str
    last_topic: Optional[str]
    ground_truth_movie: Optional[str]
    ground_truth_response: str

FEW_SHOT_EXAMPLE = """你是一个电影推荐助手。你的任务是自然地引导用户到一部电影，而不是一开始就直接说电影。

规则：
- 你必须根据“Topics Discussed”里的话题来回复用户。
- 自然地回应用户的消息。
- 仅当电影与话题匹配时才推荐 ground truth 电影，否则就自然地提及话题。

对话：
User: 说到这，有没有值得回忆的电影推荐一下？

Assistant: 
# Ground Truth Movie: 精灵鼠小弟(1999)
# Ground Truth Response: 《精灵鼠小弟》怎么样？这是部经典的童年回忆电影，一场充满勇气和智慧的冒险之旅。
"""

def load_jsonl(file_path: str) -> List[Dict]:
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {file_path}")
    return samples

def extract_fewshot_sample(sample: Dict) -> Sample:
    # last user message
    dialogue = sample.get("dialogue_history", [])
    last_user_msg = ""
    for turn in reversed(dialogue):
        if turn.get("role") == "User":
            last_user_msg = turn.get("content", "")
            break
    last_topic = sample.get("topics_discussed", [])
    last_topic_str = last_topic[-1] if last_topic else None

    return Sample(
        conversation_id=sample.get("conversation_id"),
        message_id=sample.get("message_id"),
        last_user_msg=last_user_msg,
        last_topic=last_topic_str,
        ground_truth_movie=sample.get("ground_truth_movie"),
        ground_truth_response=sample.get("ground_truth_response", "")
    )

def build_prompt(sample: Sample) -> str:
    prompt = "对话：\n"
    prompt += f"User: {sample.last_user_msg}\n"
    if sample.ground_truth_movie:
        prompt += f"# Ground Truth Movie: {sample.ground_truth_movie}\n\n"
    prompt += "Assistant:\n\n"
    prompt += f"{sample.ground_truth_response}\n"
    return prompt


def save_jsonl(samples: List[Sample], out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            obj = {
                "conversation_id": s.conversation_id,
                "message_id": s.message_id,
                "last_user_msg": s.last_user_msg,
                "last_topic": s.last_topic,
                "ground_truth_movie": s.ground_truth_movie,
                "ground_truth_response": s.ground_truth_response,
                "prompt": build_prompt(s)
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} prompts to {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TG-ReDial JSONL files")
    parser.add_argument("--output_dir", type=str, default="prompts_output", help="Output directory for prompts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        file_path = os.path.join(args.data_dir, f"{split}.jsonl")
        raw_samples = load_jsonl(file_path)
        fewshot_samples = [extract_fewshot_sample(s) for s in raw_samples]
        save_jsonl(fewshot_samples, os.path.join(args.output_dir, f"{split}_fewshot.jsonl"))

if __name__ == "__main__":
    main()
