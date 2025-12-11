import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# ===== Load data =====
file_path = "test_llama3.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

smooth_fn = SmoothingFunction().method1

# Metrics accumulators
bleu1_list, bleu2_list, bleu3_list = [], [], []
all_unigrams, all_bigrams = set(), set()
total_unigrams, total_bigrams = 0, 0

# ===== Loop through samples =====
for sample in data:
    ref_text = sample.get("ground_truth_response")
    pred_text = sample.get("llama3_response")

    if ref_text and pred_text:
        ref_tokens = [word_tokenize(ref_text)]
        pred_tokens = word_tokenize(pred_text)

        # BLEU scores
        bleu1_list.append(sentence_bleu(ref_tokens, pred_tokens, weights=(1,0,0), smoothing_function=smooth_fn))
        bleu2_list.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5,0.5,0), smoothing_function=smooth_fn))
        bleu3_list.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33,0.33,0.33), smoothing_function=smooth_fn))

        # Distinct-1 (unique unigrams / total unigrams)
        all_unigrams.update(pred_tokens)
        total_unigrams += len(pred_tokens)

        # Distinct-2 (unique bigrams / total bigrams)
        bigrams = list(zip(pred_tokens, pred_tokens[1:]))
        all_bigrams.update(bigrams)
        total_bigrams += len(bigrams)

# ===== Compute averages =====
avg_bleu1 = sum(bleu1_list)/len(bleu1_list)
avg_bleu2 = sum(bleu2_list)/len(bleu2_list)
avg_bleu3 = sum(bleu3_list)/len(bleu3_list)
distinct1 = len(all_unigrams)/total_unigrams if total_unigrams>0 else 0
distinct2 = len(all_bigrams)/total_bigrams if total_bigrams>0 else 0

# ===== Print metrics =====
print(f"BLEU-1: {avg_bleu1:.4f}")
print(f"BLEU-2: {avg_bleu2:.4f}")
print(f"BLEU-3: {avg_bleu3:.4f}")
print(f"Distinct-1: {distinct1:.4f}")
print(f"Distinct-2: {distinct2:.4f}")
