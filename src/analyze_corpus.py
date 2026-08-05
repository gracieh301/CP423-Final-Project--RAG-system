
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = PROJECT_ROOT / "data" / "spongebob_corpus.json"



def main():
    if not CORPUS_FILE.exists():
        print(f"Couldn't find {CORPUS_FILE} -- update CORPUS_FILE to the right path.")
        return

    with open(CORPUS_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    documents = corpus.get("documents", [])
    total_docs = len(documents)

   
    ids = [doc["id"] for doc in documents]
    unique_ids = set(ids)

 
    titles = [doc["title"] for doc in documents]
    unique_titles = set(titles)

    # Breakdown by type
    type_counts = Counter(doc["type"] for doc in documents)

    print("=" * 50)
    print("CORPUS ANALYSIS")
    print("=" * 50)
    print(f"Total documents:        {total_docs}")
    print(f"Unique IDs:              {len(unique_ids)}")
    print(f"Unique titles:           {len(unique_titles)}")
    print()
    print("By type:")
    for doc_type, count in type_counts.items():
        print(f"  {doc_type:<12} {count}")

    
    if len(unique_ids) != total_docs:
        dupe_ids = [id_ for id_, count in Counter(ids).items() if count > 1]
        print(f"\n[!] WARNING: {total_docs - len(unique_ids)} duplicate ID(s) found: {dupe_ids}")

    if len(unique_titles) != total_docs:
        dupe_titles = [t for t, count in Counter(titles).items() if count > 1]
        print(f"\n[!] WARNING: {total_docs - len(unique_titles)} duplicate title(s) found: {dupe_titles}")

    if len(unique_ids) == total_docs and len(unique_titles) == total_docs:
        print("\nNo duplicates found -- every document is unique.")


if __name__ == "__main__":
    main()