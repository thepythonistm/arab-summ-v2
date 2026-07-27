import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.abstractive import AraT5Summarizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "arat5-xlsum-best")


def main():
    parser = argparse.ArgumentParser(description="Arabic Summarization")
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--output", "-o", default="output.json", help="Output JSON file")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--max-chars", "-c", type=int, default=4000, help="Max article chars")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        article = f.read().strip()

    original_len = len(article)
    if original_len > args.max_chars:
        article = article[:args.max_chars]

    print(f"Loading model from: {args.model}")
    summarizer = AraT5Summarizer(args.model)

    print("Generating summary...")
    summary = summarizer.summarize("summarize: " + article)

    result = {
        "original_length": original_len,
        "truncated_length": len(article),
        "final_summary": summary,
        "compression_ratio": round((1 - len(summary) / original_len) * 100, 2) if original_len > 0 else 0
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY:")
    print(summary)
    print(f"\nCompression: {result['compression_ratio']}%")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()