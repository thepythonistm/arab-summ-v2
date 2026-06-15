"""
Extractive Module: Trained AraBERT with TF-IDF fallback.
"""
import re
import os
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import utils FIRST to trigger monkey-patch
from .utils import find_model_dir

# ── Try to load trained AraBERT extractive model ──
USE_TRAINED = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    EXTRACTIVE_PATH = find_model_dir("./models/extractive-best")
    print(f"Looking for extractive model at: {EXTRACTIVE_PATH}")
    
    if not os.path.exists(os.path.join(EXTRACTIVE_PATH, "config.json")):
        raise FileNotFoundError(f"config.json not found in {EXTRACTIVE_PATH}")
    
    _tokenizer = AutoTokenizer.from_pretrained(EXTRACTIVE_PATH, local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(EXTRACTIVE_PATH, local_files_only=True)
    _model.eval()
    USE_TRAINED = True
    print("✅ Loaded trained AraBERT extractive model.")
except Exception as e:
    print(f"⚠️ Trained model not found ({e}). Using TF-IDF fallback.")


def _segment(text: str) -> list[str]:
    raw = re.split(r'[.!?۔،؛]', text)
    return [s.strip() for s in raw if len(s.strip()) > 20] or [text]


def _tfidf_compress(article: str, max_sentences: int = 5) -> str:
    sentences = _segment(article)
    if len(sentences) <= max_sentences:
        return article
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    sim = cosine_similarity(tfidf)
    scores = sim.sum(axis=1)
    top_idx = np.argsort(scores)[-max_sentences:]
    return ' . '.join([sentences[i] for i in sorted(top_idx)])


def _trained_compress(article: str, max_sentences: int = 5) -> str:
    sentences = _segment(article)
    if len(sentences) <= max_sentences:
        return article
    
    device = next(_model.parameters()).device
    scores = []
    for sent in sentences:
        inputs = _tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = _model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0][1].item()
        scores.append(prob)
    
    top_idx = np.argsort(scores)[-max_sentences:]
    return ' . '.join([sentences[i] for i in sorted(top_idx)])


def extractive_compress(article: str, max_sentences: int = 5) -> str:
    """Uses trained AraBERT if available, else TF-IDF."""
    if USE_TRAINED:
        return _trained_compress(article, max_sentences)
    return _tfidf_compress(article, max_sentences)