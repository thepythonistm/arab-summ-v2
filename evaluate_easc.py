"""
evaluate_easc.py — Zero-shot EASC evaluation.
NO TRAINING. BEST_MODEL (XL-Sum trained) summarizes EASC articles.
Multi-reference: max ROUGE across 5 human summaries per article.
"""
import os
import json
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer

from src.abstractive import AraT5Summarizer
from src.factcheck import fact_check
from src.utils import find_model_dir

class ArabicTokenizer:
    def tokenize(self, text):
        return text.split()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = find_model_dir(os.path.join(BASE_DIR, "models", "BEST_MODEL"))
DATA_PATH = os.path.join(BASE_DIR, "data", "easc_multiref.json")
OUT_PATH = os.path.join(BASE_DIR, "easc_zeroshot_report.json")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                  use_stemmer=False, tokenizer=ArabicTokenizer())

def rouge_vs_refs(pred, refs):
    """Max F1 across all human references (standard multi-ref practice)."""
    best = None
    for ref in refs:
        s = scorer.score(ref, pred)
        cur = {k: s[k].fmeasure * 100 for k in ['rouge1', 'rouge2', 'rougeL']}
        if best is None or cur['rouge1'] > best['rouge1']:
            best = cur
    return best

def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Zero-shot on {len(data)} EASC articles (no training)")

    summarizer = AraT5Summarizer(MODEL_PATH)
    results = []
    for item in tqdm(data, desc="EASC zero-shot"):
        pred = summarizer.summarize(item["text"])
        scores = rouge_vs_refs(pred, item["summaries"])
        fc = fact_check(pred, item["text"])
        results.append({"id": item["id"], "pred": pred, **scores,
                        "fact_score": fc["fact_score"], "faithful": fc["faithful"]})

    report = {
        "experiment": "zero-shot XL-Sum→EASC, article-level, multi-reference max",
        "n_articles": len(data),
        "avg_rouge1": round(np.mean([r["rouge1"] for r in results]), 4),
        "avg_rouge2": round(np.mean([r["rouge2"] for r in results]), 4),
        "avg_rougeL": round(np.mean([r["rougeL"] for r in results]), 4),
        "avg_fact_score": round(np.mean([r["fact_score"] for r in results]), 4),
        "faithful_rate_pct": round(sum(r["faithful"] for r in results) / len(results) * 100, 2),
        "samples": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nROUGE-1: {report['avg_rouge1']} | ROUGE-2: {report['avg_rouge2']} | ROUGE-L: {report['avg_rougeL']}")
    print(f"Fact Score: {report['avg_fact_score']} | Faithful: {report['faithful_rate_pct']}%")
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()