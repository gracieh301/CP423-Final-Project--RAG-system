import json
import re

INPUT_FILE = "data/spongebob_corpus_clean.json"
OUTPUT_FILE = "data/processed_corpus.json"

CHUNK_SIZE = 250
OVERLAP = 50

REMOVE_HEADINGS = {
    "gallery",
    "references",
    "see also",
    "external links",
    "navigation",
    "categories",
    "images",
    "videos",
    "trivia"
}


def clean_text(text):

    if not text:
        return ""

    # Remove citations like [1]
    text = re.sub(r"\[\d+\]", "", text)

    # Remove "Main article: ..."
    text = re.sub(r"Main article:\s*[^.\n]+\.?", "", text, flags=re.IGNORECASE)

    # Remove image/file references
    text = re.sub(r"File:[^\n]+", "", text, flags=re.IGNORECASE)

    # Remove image captions (short standalone sentences)
    text = re.sub(
        r"\b[A-Z][A-Za-z' -]{1,40}\s(?:watching|arguing|holding|looking|standing|running|sitting|smiling|laughing)[^.]*\.",
        "",
        text
    )

    # Remove extra whitespace/newlines
    text = re.sub(r"\n+", "\n", text)

    cleaned = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        if line.lower() in REMOVE_HEADINGS:
            continue

        cleaned.append(line)

    text = " ".join(cleaned)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def chunk_document(text):

    words = text.split()

    if len(words) <= CHUNK_SIZE:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        # If this isn't the last chunk, try to end at the previous sentence
        if end < len(words):

            for i in range(end, max(start + CHUNK_SIZE // 2, start), -1):

                if words[i - 1].endswith((".", "!", "?")):
                    end = i
                    break

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = max(end - OVERLAP, start + 1)

    return chunks


# -------------------------
# Main preprocessing
# -------------------------
if __name__ == "__main__":

    # Load corpus
    with open(INPUT_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    processed = []

    documents = corpus["documents"]
    total_docs = len(documents)

    print(f"Loaded {total_docs} documents.")
    print("Starting preprocessing...\n")

    for index, doc in enumerate(documents, start=1):

        cleaned = clean_text(doc["text"])
        chunks = chunk_document(cleaned)

        for i, chunk in enumerate(chunks, start=1):

            processed.append({

                "chunk_id": f"{doc['id']}_{i:03}",

                "doc_id": doc["id"],

                "title": doc["title"],

                "type": doc["type"],

                "url": doc["url"],

                "text": chunk

            })

        # Print progress every 200 documents
        if index % 200 == 0 or index == total_docs:
            print(
                f"Processed {index}/{total_docs} documents "
                f"({100 * index / total_docs:.1f}%) "
                f"- Total chunks: {len(processed)}"
            )

    output = {

        "corpus_name": corpus["corpus_name"],

        "num_chunks": len(processed),

        "documents": processed

    }

    print("\nWriting processed corpus...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nFinished preprocessing!")
    print(f"Documents processed: {total_docs}")
    print(f"Chunks created: {len(processed)}")
    print(f"Saved to: {OUTPUT_FILE}")