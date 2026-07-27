"""
Post-processing: Deduplication, punctuation fix, length compression.
"""
import re

SENTENCE_SPLIT_RE = re.compile(r'([.!?؟۔])\s*')

def split_sentences(text: str) -> list[str]:
    """Split on Arabic + Latin terminators, keep delimiter attached."""
    parts = SENTENCE_SPLIT_RE.sub(r'\1|SPLIT|', text)
    return [p.strip() for p in parts.split('|SPLIT|') if p.strip()]


def remove_duplicate_sentences(text: str) -> str:
    """Remove repeated sentences while preserving order and punctuation."""
    sentences = split_sentences(text)
    seen = set()
    unique = []
    for s in sentences:
        # Normalize for comparison (remove trailing punctuation + whitespace)
        normalized = re.sub(r'\s+', ' ', s).strip()
        normalized = re.sub(r'[.!?؟۔]+$', '', normalized).strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)
    return ' '.join(unique)


def fix_arabic_punctuation(text: str) -> str:
    """Fix spacing around Arabic and Latin punctuation."""
    # Remove space before punctuation
    text = re.sub(r'\s+([،؛؟!.])', r'\1', text)
    # Add space after punctuation (if followed by a word)
    text = re.sub(r'([،؛؟!.])(?=\S)', r'\1 ', text)
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compress_length(text: str, max_words: int = 60) -> str:
    """Hard compress if summary exceeds target word count."""
    words = text.split()
    if len(words) <= max_words:
        return text

    sentences = split_sentences(text)
    result = []
    count = 0
    for s in sentences:
        s_words = len(s.split())
        if count + s_words <= max_words:
            result.append(s)
            count += s_words
        else:
            break

    return ' '.join(result) if result else ' '.join(words[:max_words])


def postprocess_summary(text: str, max_words: int = 60) -> str:
    """
    Full post-processing pipeline:
    1. Remove duplicate sentences
    2. Fix Arabic punctuation
    3. Compress to target length
    """
    text = remove_duplicate_sentences(text)
    text = fix_arabic_punctuation(text)
    text = compress_length(text, max_words=max_words)
    return text