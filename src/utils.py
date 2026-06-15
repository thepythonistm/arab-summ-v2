"""
Utility helpers + Windows path fix for transformers.
"""
import os
import json

# ═══════════════════════════════════════════════════════
# CRITICAL FIX: Disable huggingface_hub validation ONLY
# for local Windows paths. Preserves HF Hub downloads.
# ═══════════════════════════════════════════════════════
import huggingface_hub.utils._validators as _validators
_original_validate = _validators.validate_repo_id

def _patched_validate(repo_id):
    # Skip validation only for existing local paths
    if isinstance(repo_id, str) and (os.path.exists(repo_id) or os.path.isdir(repo_id)):
        return
    _original_validate(repo_id)

_validators.validate_repo_id = _patched_validate


def find_model_dir(base_path: str) -> str:
    """Auto-discover model directory. Handles nested folders from zip extraction."""
    base_path = os.path.abspath(base_path)
    
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path
    
    if os.path.isdir(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                return item_path
    
    return base_path


def load_test_articles(path: str) -> list[dict]:
    """Load test articles from JSON or JSONL."""
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        f.seek(0)
        if first.startswith("["):
            articles = json.load(f)
        else:
            for line in f:
                articles.append(json.loads(line.strip()))
    return articles


def save_report(results: list[dict], path: str):
    """Save evaluation results to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)