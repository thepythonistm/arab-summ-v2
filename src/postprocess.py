"""
Post-processing: Deduplication, punctuation fix, length compression.
"""
import re


def remove_duplicate_sentences(text: str) -> str:
    """Remove repeated sentences while preserving order."""
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    seen = set()
    unique = []
    for s in sentences:
        normalized = re.sub(r'\s+', ' ', s).strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)
    return ' . '.join(unique)


def fix_arabic_punctuation(text: str) -> str:
    """Fix spacing around Arabic punctuation."""
    text = re.sub(r'\s+([،؛؟!])', r'\1', text)
    text = re.sub(r'([،؛؟!])\s+', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compress_length(text: str, max_words: int = 60) -> str:
    """Hard compress if summary exceeds target word count."""
    words = text.split()
    if len(words) <= max_words:
        return text

    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    result = []
    count = 0
    for s in sentences:
        s_words = len(s.split())
        if count + s_words <= max_words:
            result.append(s)
            count += s_words
        else:
            break

    return ' . '.join(result) if result else ' '.join(words[:max_words])


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