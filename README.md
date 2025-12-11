# Conversational Movie Recommender

A conversational movie recommendation system using few-shot prompting with LLaMA 3 on the TG-ReDial dataset. This is part of the final project for CS 8803 Conversational AI course at Georgia Institute of Technology.

## Overview

This project investigates prompt engineering methods for generating natural movie recommendations in conversational AI. Instead of directly outputting movie titles, the model learns to weave recommendations naturally into dialogue based on:

- User profiles and interests
- Discussion topics
- Conversation history
- Ground truth movie to recommend

## Project Structure

```
prompt_method/
├── dataset_loader.py        # Load TG-ReDial dataset → JSONL
├── fewshot_prompts.py       # Generate few-shot prompts
├── run_llama3.py            # Main LLaMA 3 inference (with checkpointing)
├── test_run_llama3.py       # Small-scale test runner
├── eval_metrics.py          # Evaluate with BLEU & Distinct scores
├── download_and_load_llama3.py  # Download LLaMA 3 model
├── train.jsonl              # Training samples
├── valid.jsonl              # Validation samples
├── test.jsonl               # Test samples
├── *_llama3.jsonl           # Model outputs
└── llama3_checkpoints.json  # Checkpoint file for resumable runs
```

## Installation

```bash
pip install torch transformers nltk tqdm
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage

### 1. Prepare Dataset

Convert TG-ReDial pickle files to JSONL format:

```bash
python dataset_loader.py --data_dir data --subset train --output train.jsonl
python dataset_loader.py --data_dir data --subset valid --output valid.jsonl
python dataset_loader.py --data_dir data --subset test --output test.jsonl
```

### 2. Run LLaMA 3 Inference

Process all splits with checkpointing:

```bash
python run_llama3.py --order test valid train --checkpoint-interval 100
```

Options:
- `--order`: Specify processing order of splits
- `--checkpoint-interval`: Save progress every N samples (default: 100)
- `--max-samples`: Limit samples per split (-1 for all)
- `--data-dir`: Directory containing JSONL files

For quick testing:
```bash
python test_run_llama3.py
```

### 3. Evaluate Results

```bash
python eval_metrics.py
```

Outputs:
- **BLEU-1/2/3**: N-gram overlap with ground truth
- **Distinct-1/2**: Lexical diversity (unique unigrams/bigrams)

## Data Format

### Input Sample (JSONL)
```json
{
  "conversation_id": 123,
  "message_id": 5,
  "user_profile": ["comedy", "action"],
  "topics_discussed": ["childhood movies", "Stuart Little"],
  "dialogue_history": [
    {"role": "User", "content": "Any nostalgic movie recommendations?"}
  ],
  "ground_truth_movie": "Stuart Little",
  "ground_truth_response": "How about Stuart Little? A classic childhood adventure film."
}
```

### Output Sample (after LLaMA 3)
```json
{
  ...original fields...,
  "llama3_prompt": "...",
  "llama3_response": "Generated recommendation response"
}
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | Meta-Llama-3-8B |
| Max New Tokens | 128 |
| Temperature | 0.8 |
| Top-p | 0.9 |
| Repetition Penalty | 1.2 |

## Prompt Strategy

The system uses Chinese guidelines with a few-shot example:

```
你是一个电影推荐助手。你的任务是根据"Topic"自然地推荐电影，不要一开始就直接说电影名字。

示例:
Ground Truth Movie: 精灵鼠小弟(1999)
Topic: 精灵鼠小弟

对话:
User: 说到这，有没有值得回忆的电影推荐一下？
Assistant:
《精灵鼠小弟》怎么样？这是部经典的童年回忆电影...
```

## Dataset

This project uses the [TG-ReDial](https://github.com/RUCAIBox/TG-ReDial) dataset, a topic-guided conversational recommendation dataset in Chinese.

