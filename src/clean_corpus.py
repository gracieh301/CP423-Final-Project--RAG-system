"""
Clean spongebob_corpus.json so each EPISODE document's "text" field only
contains the Synopsis and Trivia sections (character documents are left
untouched).

Why this approach: the plaintext extracted from each episode page is one
long blob with section headers as bare lines (e.g. "Synopsis", "Trivia",
"Production", ...). We split on those known top-level headers and keep
only the content under "Synopsis" and "Trivia".

Usage:
    python clean_corpus.py
      (reads spongebob_corpus.json, writes spongebob_corpus_clean.json)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "spongebob_corpus.json"
OUTPUT_FILE = DATA_DIR / "spongebob_corpus_clean.json"

# Top-level section headers that show up as their own line on episode
# pages, in the order they typically appear. Anything not in this set
# (Act 1, Act 2, General, Dub facts, Errors, Cultural references, etc.)
# is treated as a SUB-heading nested inside whichever top-level section
# it falls under, and gets kept/dropped along with that section.
TOP_LEVEL_HEADERS = [
    "Characters",
    "Synopsis",
    "Production",
    "Release",
    "Reception",
    "Trivia",
    "Videos",
    "Names in other languages",
    "References",
    "Notes",
]

SECTIONS_TO_KEEP = ("Synopsis", "Trivia")


def split_into_sections(text):
    """Split episode plaintext into {header: content} using the known
    top-level headers as section boundaries."""

    lines = text.split("\n")
    sections = {}
    current_header = None
    buffer = []

    for line in lines:
        stripped = line.strip()

        if stripped in TOP_LEVEL_HEADERS:
            # flush whatever we were collecting into the previous header
            if current_header is not None:
                sections[current_header] = "\n".join(buffer).strip()
            current_header = stripped
            buffer = []
        else:
            if current_header is not None:
                buffer.append(line)

    if current_header is not None:
        sections[current_header] = "\n".join(buffer).strip()

    return sections


def clean_episode_text(text):
    sections = split_into_sections(text)

    parts = []
    for header in SECTIONS_TO_KEEP:
        content = sections.get(header, "").strip()
        if content:
            parts.append(f"{header}\n\n{content}")

    return "\n\n".join(parts).strip()


def main():
    if not INPUT_FILE.exists():
        print(f"Couldn't find {INPUT_FILE} -- update INPUT_FILE to the right path.")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    documents = corpus.get("documents", [])

    cleaned_count = 0
    empty_count = 0

    for doc in documents:
        if doc.get("type") != "episode":
            continue

        original_text = doc.get("text", "")
        cleaned_text = clean_episode_text(original_text)

        if not cleaned_text:
            empty_count += 1
            # keep original text as fallback rather than wiping it out,
            # in case this episode's page didn't match expected headers
            print(f"  [!] No Synopsis/Trivia found for: {doc.get('title')} "
                  f"(keeping original text as fallback)")
            continue

        doc["text"] = cleaned_text
        doc["char_count"] = len(cleaned_text)
        cleaned_count += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\nCleaned {cleaned_count} episode documents.")
    if empty_count:
        print(f"{empty_count} episode(s) had no matching Synopsis/Trivia "
              f"sections and were left as-is -- check these manually.")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()