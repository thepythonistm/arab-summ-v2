"""
Fact Checking Module: verifies that critical facts in the generated
summary (numbers, dates, percentages, Latin entities, quoted phrases)
actually exist in the source document. No training needed.
"""
import re

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize(text: str) -> str:
    return text.translate(AR_DIGITS)


def extract_facts(text: str) -> dict:
    """Extract checkable facts from text."""
    t = _normalize(text)
    return {
        "numbers": set(re.findall(r'\d+(?:[.,]\d+)?', t)),
        "percents": set(re.findall(r'\d+(?:[.,]\d+)?\s*%', t)),
        "years": set(re.findall(r'\b(?:19|20)\d{2}\b', t)),
        # Latin proper nouns (names, orgs) if present
        "latin_entities": set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)),
        # Quoted phrases (Arabic or Latin quotes)
        "quotes": set(re.findall(r'[«"]([^»"]{5,60})[»"]', text)),
    }


def fact_check(summary: str, source: str) -> dict:
    """
    Compare facts in summary vs source document.
    Returns faithfulness verdict + score + unsupported facts.
    """
    sum_facts = extract_facts(summary)
    src_facts = extract_facts(source)

    unsupported = []
    for category in ["numbers", "percents", "years", "latin_entities"]:
        # years are a subset of numbers — skip double counting
        if category == "numbers":
            diff = (sum_facts[category] - src_facts[category]) - sum_facts["years"]
        else:
            diff = sum_facts[category] - src_facts[category]
        for item in diff:
            unsupported.append({"type": category, "value": item})

    # Quotes: substring check against full source text (not just quoted set)
    for q in sum_facts["quotes"]:
        if q.strip() not in source:
            unsupported.append({"type": "quotes", "value": q})

    total = sum(len(v) for v in sum_facts.values())
    total = max(total - len(sum_facts["years"]), 1)  # avoid double count
    score = 1.0 - (len(unsupported) / total)

    return {
        "faithful": len(unsupported) == 0,
        "fact_score": round(max(score, 0.0), 3),
        "unsupported_facts": unsupported,
        "facts_checked": total
    }