"""
Extractive Module: AraBERT sentence scoring + MMR selection.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_PATH = "models/extractive-best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None


def _load_model():
    """Lazy load on first use."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.to(DEVICE)   # 🔧 FIX: model was never moved to GPU before
        _model.eval()


def split_sentences(text):
    delimiters = r'[.!?۔!\?]+'
    sentences = re.split(delimiters, text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def get_model_scores_batched(sentences, text):
    if not sentences:
        return np.array([])

    _load_model()
    encodings = _tokenizer(
        [text] * len(sentences),
        sentences,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)   # 🔧 FIX: inputs also moved to same device

    with torch.no_grad():
        outputs = _model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

    return probs


def mmr_select(sentences, scores, lambda_param=0.6, max_sentences=5):
    """
    Maximal Marginal Relevance — no training needed.
    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
    n = len(sentences)
    if n <= max_sentences:
        return list(range(n))

    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    vectorizer = TfidfVectorizer()
    try:
        sent_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.argsort(scores)[-max_sentences:].tolist()

    selected = [int(np.argmax(scores))]
    remaining = set(range(n)) - set(selected)

    while len(selected) < max_sentences and remaining:
        mmr_scores = {}
        for idx in remaining:
            relevance = scores[idx]
            sims = [
                cosine_similarity(sent_vectors[idx], sent_vectors[s])[0, 0]
                for s in selected
            ]
            redundancy = max(sims)
            mmr_scores[idx] = lambda_param * relevance - (1 - lambda_param) * redundancy

        best_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def rank_sentences(sentences, model_scores):
    if not sentences:
        return []

    model_scores = np.array(model_scores)

    position_scores = np.array([
        1.0 if i < 3 else 0.5 if i < 6 else 0.2 / (i - 5)
        for i in range(len(sentences))
    ])

    lengths = np.array([len(s.split()) for s in sentences])
    length_scores = np.where(
        (lengths >= 15) & (lengths <= 40), 1.0,
        np.where(lengths < 15, lengths / 15.0, 40.0 / lengths)
    )

    def norm(arr):
        arr = arr - arr.min()
        return arr / (arr.max() + 1e-8) if arr.max() > 0 else arr

    model_scores = norm(model_scores)
    position_scores = norm(position_scores)
    length_scores = norm(length_scores)

    final = 0.50 * model_scores + 0.35 * position_scores + 0.15 * length_scores

    return list(zip(sentences, final))


def extractive_compress(text, max_sentences=5, max_words=120, lambda_mmr=0.6):
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    model_scores = get_model_scores_batched(sentences, text)
    ranked = rank_sentences(sentences, model_scores)

    final_scores = [score for _, score in ranked]

    selected_indices = mmr_select(sentences, final_scores,
                                  lambda_param=lambda_mmr,
                                  max_sentences=max_sentences)

    selected_indices = sorted(selected_indices)
    selected_sentences = [sentences[i] for i in selected_indices]

    result = " ".join(selected_sentences)

    words = result.split()
    if len(words) > max_words:
        result = " ".join(words[:max_words]) + "."

    return result