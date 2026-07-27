"""
evaluate.py — Full evaluation for Arabic Summarization
Supports: Abstractive (AraT5), Hybrid (Extractive → AraT5 → Post-process)
Metrics: Standard ROUGE + Arabic-normalized ROUGE + BERTScore + Fact Score (both pipelines)
"""
import sys
import os
import json
import glob
import time
import logging
import numpy as np
from tqdm import tqdm
import re
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.utils

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from src.abstractive import AraT5Summarizer
from src.hybrid import HybridPipeline
from src.factcheck import fact_check
from src.utils import find_model_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class ArabicTokenizer:
    def tokenize(self, text):
        return text.split()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = find_model_dir(os.path.join(BASE_DIR, "models", "BEST_MODEL"))
TEST_PATH = os.path.join(BASE_DIR, "data", "test_articles_raw.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "full_evaluation_report.json")
MAX_SAMPLES = 2000


def load_test_data(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        _suggest_data_files()
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.endswith('.jsonl'):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict):
        for key in ["samples", "data", "articles", "records", "items", "test"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    if not data:
        raise ValueError("Dataset is empty")

    first = data[0]
    keys = set(first.keys())

    result_keys = {"prediction", "reference", "rouge1", "rouge2", "rougeL"}
    if result_keys.issubset(keys):
        raise ValueError(f"{path} is a results file, not a raw dataset.")

    text_candidates = ["text", "article", "document", "content", "body", "input", "source"]
    text_key = next((k for k in text_candidates if k in keys), None)
    summary_candidates = ["summary", "headline", "abstract", "target", "output", "title", "summary_text", "reference"]
    summary_key = next((k for k in summary_candidates if k in keys), None)
    id_candidates = ["id", "ID", "sample_id", "doc_id", "index"]
    id_key = next((k for k in id_candidates if k in keys), None)

    if not text_key:
        raise KeyError(f"No article field found. Keys: {sorted(keys)}")
    if not summary_key:
        raise KeyError(f"No summary field found. Keys: {sorted(keys)}")

    print(f"Auto-detected fields — article: '{text_key}', summary: '{summary_key}', id: '{id_key or 'index'}'")
    print(f"Loaded {len(data)} test articles")

    normalized = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        norm = {
            "id": str(item.get(id_key, idx)) if id_key else str(idx),
            "text": str(item.get(text_key, "")).strip(),
            "summary": str(item.get(summary_key, "")).strip()
        }
        if norm["text"]:
            normalized.append(norm)

    if not normalized:
        raise ValueError("No valid samples")

    print(f"Final valid samples: {len(normalized)}")
    return normalized


def _suggest_data_files():
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, "*.json")) + glob.glob(os.path.join(data_dir, "*.jsonl"))
        if files:
            print("Available files:")
            for f in files:
                print(f"  - {os.path.basename(f)}")


rouge_scorer_obj = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=False,
    tokenizer=ArabicTokenizer()
)


def normalize_arabic(text: str) -> str:
    """Morphology normalization for fairer Arabic ROUGE (Elsaid et al., 2023)."""
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)   # diacritics + tatweel
    text = re.sub(r'[أإآ]', 'ا', text)                   # alef variants
    text = re.sub(r'ى', 'ي', text)                       # alef maksura
    text = re.sub(r'ة', 'ه', text)                       # ta marbuta
    text = re.sub(r'\bال', '', text)                     # strip "al-" prefix
    return text


def compute_rouge(pred, ref):
    scores = rouge_scorer_obj.score(ref, pred)
    norm_scores = rouge_scorer_obj.score(normalize_arabic(ref), normalize_arabic(pred))
    return {
        "rouge1": round(scores['rouge1'].fmeasure * 100, 4),
        "rouge2": round(scores['rouge2'].fmeasure * 100, 4),
        "rougeL": round(scores['rougeL'].fmeasure * 100, 4),
        "rouge1_norm": round(norm_scores['rouge1'].fmeasure * 100, 4),
        "rouge2_norm": round(norm_scores['rouge2'].fmeasure * 100, 4),
        "rougeL_norm": round(norm_scores['rougeL'].fmeasure * 100, 4),
    }


def avg_rouge_block(scores):
    """Average all 6 ROUGE variants from a list of per-sample dicts."""
    keys = ["rouge1", "rouge2", "rougeL", "rouge1_norm", "rouge2_norm", "rougeL_norm"]
    return {f"avg_{k}": round(np.mean([s[k] for s in scores]), 4) for k in keys}


def avg_fact_block(fact_scores, faithful_flags):
    """Average fact metrics — same computation for both pipelines (fair comparison)."""
    valid = [f for f in fact_scores if f is not None]
    rate = sum(1 for f in faithful_flags if f) / max(len(faithful_flags), 1) * 100
    return {
        "avg_fact_score": round(np.mean(valid), 4) if valid else None,
        "faithful_rate_pct": round(rate, 2),
    }


def compute_bertscore(preds, refs):
    try:
        P, R, F1 = bert_score_fn(
            preds, refs, lang="ar",
            model_type="bert-base-multilingual-cased",
            num_layers=9, verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return [round(f * 100, 4) for f in F1.tolist()]
    except Exception as e:
        print(f"BERTScore skipped: {e}")
        return [None] * len(preds)


def print_rouge_block(block):
    """Pretty print standard + normalized ROUGE side by side."""
    print(f"ROUGE-1:        {block['avg_rouge1']}   (norm: {block['avg_rouge1_norm']})")
    print(f"ROUGE-2:        {block['avg_rouge2']}   (norm: {block['avg_rouge2_norm']})")
    print(f"ROUGE-L:        {block['avg_rougeL']}   (norm: {block['avg_rougeL_norm']})")


def run_abstractive(articles, model_path):
    summarizer = AraT5Summarizer(model_path)
    preds, fact_scores, faithful_flags = [], [], []
    for article in tqdm(articles, desc="Abstractive"):
        pred = summarizer.summarize(article)
        preds.append(pred)
        fc = fact_check(pred, article)   # same verification as hybrid — fair comparison
        fact_scores.append(fc.get("fact_score"))
        faithful_flags.append(fc.get("faithful"))
    return preds, fact_scores, faithful_flags


def run_hybrid(articles, model_path):
    pipeline = HybridPipeline(model_path)
    preds, fact_scores, faithful_flags, retries = [], [], [], 0
    for article in tqdm(articles, desc="Hybrid"):
        result = pipeline.run(article, max_summary_words=60)
        preds.append(result["final_summary"])
        fc = result.get("fact_check", {})
        fact_scores.append(fc.get("fact_score"))
        faithful_flags.append(fc.get("faithful"))
        if result.get("regenerated"):
            retries += 1
    return preds, fact_scores, faithful_flags, retries


def main():
    print("=" * 70)
    print("ARABIC SUMMARIZATION — FULL EVALUATION")
    print("=" * 70)

    data = load_test_data(TEST_PATH)
    if MAX_SAMPLES:
        data = data[:MAX_SAMPLES]
        print(f"Evaluating on {MAX_SAMPLES} samples (subset mode)")
    else:
        print(f"Evaluating on {len(data)} samples (full test set)")

    articles = [d["text"] for d in data]
    refs = [d["summary"] for d in data]

    results = {
        "metadata": {
            "total_samples": len(data),
            "model_path": MODEL_PATH,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "pipelines": {}
    }

    # ═══════════════ ABSTRACTIVE ═══════════════
    print("\n" + "=" * 70)
    print("ABSTRACTIVE PIPELINE (AraT5 direct)")
    print("=" * 70)
    abs_preds, abs_fact_scores, abs_faithful_flags = run_abstractive(articles, MODEL_PATH)

    abs_scores = [compute_rouge(p, r) for p, r in zip(abs_preds, refs)]
    abs_bert = compute_bertscore(abs_preds, refs)

    results["pipelines"]["abstractive"] = {
        **avg_rouge_block(abs_scores),
        "avg_bertscore": round(np.mean([b for b in abs_bert if b is not None]), 4) if any(b is not None for b in abs_bert) else None,
        **avg_fact_block(abs_fact_scores, abs_faithful_flags),
        "samples": [
            {"id": data[i]["id"], "pred": abs_preds[i], "ref": refs[i],
             **abs_scores[i], "bertscore": abs_bert[i],
             "fact_score": abs_fact_scores[i], "faithful": abs_faithful_flags[i]}
            for i in range(len(data))
        ]
    }
    print_rouge_block(results["pipelines"]["abstractive"])
    print(f"BERTScore:      {results['pipelines']['abstractive']['avg_bertscore']}")
    print(f"Fact Score:     {results['pipelines']['abstractive']['avg_fact_score']}")
    print(f"Faithful:       {results['pipelines']['abstractive']['faithful_rate_pct']}%")

    # ═══════════════ HYBRID ═══════════════
    print("\n" + "=" * 70)
    print("HYBRID PIPELINE (Extractive → AraT5 → Post-process → Fact-check)")
    print("=" * 70)
    hybrid_preds, fact_scores, faithful_flags, retries = run_hybrid(articles, MODEL_PATH)

    hybrid_scores = [compute_rouge(p, r) for p, r in zip(hybrid_preds, refs)]
    hybrid_bert = compute_bertscore(hybrid_preds, refs)

    results["pipelines"]["hybrid"] = {
        **avg_rouge_block(hybrid_scores),
        "avg_bertscore": round(np.mean([b for b in hybrid_bert if b is not None]), 4) if any(b is not None for b in hybrid_bert) else None,
        **avg_fact_block(fact_scores, faithful_flags),
        "regenerations": retries,
        "samples": [
            {"id": data[i]["id"], "pred": hybrid_preds[i], "ref": refs[i],
             **hybrid_scores[i], "bertscore": hybrid_bert[i],
             "fact_score": fact_scores[i], "faithful": faithful_flags[i]}
            for i in range(len(data))
        ]
    }
    print_rouge_block(results["pipelines"]["hybrid"])
    print(f"BERTScore:      {results['pipelines']['hybrid']['avg_bertscore']}")
    print(f"Fact Score:     {results['pipelines']['hybrid']['avg_fact_score']}")
    print(f"Faithful:       {results['pipelines']['hybrid']['faithful_rate_pct']}%")
    print(f"Regenerated:    {retries} summaries")

    # ═══════════════ COMPARISON SUMMARY ═══════════════
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    a, h = results["pipelines"]["abstractive"], results["pipelines"]["hybrid"]
    print(f"{'Metric':<22}{'Direct AraT5':>15}{'Hybrid':>12}")
    print("-" * 49)
    print(f"{'ROUGE-1':<22}{a['avg_rouge1']:>15}{h['avg_rouge1']:>12}")
    print(f"{'ROUGE-2':<22}{a['avg_rouge2']:>15}{h['avg_rouge2']:>12}")
    print(f"{'ROUGE-L':<22}{a['avg_rougeL']:>15}{h['avg_rougeL']:>12}")
    print(f"{'BERTScore':<22}{a['avg_bertscore']:>15}{h['avg_bertscore']:>12}")
    print(f"{'Fact Score':<22}{a['avg_fact_score']:>15}{h['avg_fact_score']:>12}")
    print(f"{'Faithful %':<22}{a['faithful_rate_pct']:>15}{h['faithful_rate_pct']:>12}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()