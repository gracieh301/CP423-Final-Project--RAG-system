import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "spongebob_corpus.json", encoding="utf-8") as f:
    original = json.load(f)

with open(DATA_DIR / "spongebob_corpus_clean.json", encoding="utf-8") as f:
    cleaned = json.load(f)

# grab the first episode present in both files
orig_doc = next(d for d in original["documents"] if d["type"] == "episode")
clean_doc = next(d for d in cleaned["documents"] if d["id"] == orig_doc["id"])

print(f"=== {orig_doc['title']} ===\n")
print(f"--- ORIGINAL ({len(orig_doc['text'])} chars) ---")
print(orig_doc["text"][:1000], "...\n")
print(f"--- CLEANED ({len(clean_doc['text'])} chars) ---")
print(clean_doc["text"])