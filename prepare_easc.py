"""
prepare_easc.py — Convert EASC corpus to evaluation JSON.
Pairs each article (Articles/TopicN/) with its 5 MTurk human summaries (MTurk/TopicN/).
"""
import os
import json

EASC_ROOT = "EASC/EASC-UTF-8"
OUT_PATH = "data/easc_multiref.json"

def read(path):
    for enc in ("utf-8", "cp1256", "windows-1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read().strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""

def collect(root_dir, ext_filter=".txt"):
    """Returns {topic_name: [file paths]} for a given subtree."""
    topics = {}
    sub_root = os.path.join(EASC_ROOT, root_dir)
    if not os.path.isdir(sub_root):
        return topics
    for topic in os.listdir(sub_root):
        topic_dir = os.path.join(sub_root, topic)
        if not os.path.isdir(topic_dir):
            continue
        files_found = []
        for dirpath, _, files in os.walk(topic_dir):
            for f in files:
                if ext_filter is None or f.lower().endswith(ext_filter):
                    files_found.append(os.path.join(dirpath, f))
        if files_found:
            topics[topic] = sorted(files_found)
    return topics

articles = collect("Articles", ext_filter=".txt")
summaries = collect("MTurk", ext_filter=None)

print(f"Articles: {len(articles)} topics | Summaries: {len(summaries)} topics")

data = []
skipped = []
for topic, art_files in sorted(articles.items()):
    text = read(art_files[0])          # one article per topic
    refs = [read(p) for p in summaries.get(topic, [])]
    refs = [r for r in refs if r]
    if not text or not refs:
        skipped.append(topic)
        continue
    data.append({"id": topic, "text": text, "summaries": refs})

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

ref_counts = [len(d["summaries"]) for d in data]
print(f"✅ Saved {len(data)} articles to {OUT_PATH}")
print(f"   Refs per article: min={min(ref_counts)}, max={max(ref_counts)}, first 10: {ref_counts[:10]}")
if skipped:
    print(f"   ⚠️ Skipped {len(skipped)} topics (no match): {skipped[:10]}")