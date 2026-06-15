"""
CLI runner for the hybrid summarization pipeline.
Usage:
    python pipeline.py --input article.txt --output summary.json
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybrid import HybridPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "arat5-xlsum-best", "arat5-xlsum-best")


def main():
    parser = argparse.ArgumentParser(description="Arabic Hybrid Summarization")
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--output", "-o", default="output.json", help="Output JSON file")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--sentences", "-s", type=int, default=5, help="Max extractive sentences")
    parser.add_argument("--words", "-w", type=int, default=60, help="Max summary words")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        article = f.read().strip()

    print(f"Loading model from: {args.model}")
    pipe = HybridPipeline(args.model)

    print("Generating summary...")
    result = pipe.run(article, max_sentences=args.sentences, max_summary_words=args.words)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY:")
    print(result["final_summary"])
    print(f"\nCompression: {result['compression_ratio']}%")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()