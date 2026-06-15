#!/usr/bin/env python3
"""
Evaluate full hybrid pipeline on test set and produce evaluation_report.json.
"""
import sys
import os
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rouge_score import rouge_scorer
from hybrid import HybridPipeline
from utils import load_test_articles, save_report


class ArabicTokenizer:
    def tokenize(self, text):
        return text.split()


def main():
    # Config
    MODEL_PATH = "./models/arat5-xlsum-best"
    TEST_PATH = "./data/test_articles.json"
    OUTPUT_PATH = "./evaluation_report.json"

    print("Loading model...")
    pipe = HybridPipeline(MODEL_PATH)

    print("Loading test data...")
    articles = load_test_articles(TEST_PATH)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=ArabicTokenizer()
    )

    results = []
    rouge1, rouge2, rougeL = [], [], []

    for item in tqdm(articles, desc="Evaluating"):
        article = item["text"]
        ref = item["summary"]

        pred = pipe.run(article)["final_summary"]

        scores = scorer.score(ref, pred)
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure

        rouge1.append(r1)
        rouge2.append(r2)
        rougeL.append(rl)

        results.append({
            "id": item.get("id", ""),
            "reference": ref,
            "prediction": pred,
            "rouge1": round(r1 * 100, 4),
            "rouge2": round(r2 * 100, 4),
            "rougeL": round(rl * 100, 4)
        })

    report = {
        "model": MODEL_PATH,
        "total_samples": len(articles),
        "avg_rouge1": round(np.mean(rouge1) * 100, 4),
        "avg_rouge2": round(np.mean(rouge2) * 100, 4),
        "avg_rougeL": round(np.mean(rougeL) * 100, 4),
        "samples": results
    }

    save_report(report, OUTPUT_PATH)
    print(f"\n{'='*60}")
    print(f"ROUGE-1: {report['avg_rouge1']}")
    print(f"ROUGE-2: {report['avg_rouge2']}")
    print(f"ROUGE-L: {report['avg_rougeL']}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()