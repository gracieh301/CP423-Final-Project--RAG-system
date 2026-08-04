import json
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi


# -------------------------
# Configuration
# -------------------------

INPUT_FILE = "data/processed_corpus.json"

TOP_K = 5


# -------------------------
# BM25 preprocessing
# -------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def tokenize(text):
    """
    Convert text into BM25 tokens:
    - lowercase
    - remove punctuation
    - remove stopwords
    - stem words
    """

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    tokens = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return tokens


# -------------------------
# Load corpus
# -------------------------

def load_corpus():

    with open(INPUT_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    return corpus["documents"]


# -------------------------
# Build BM25 index
# -------------------------

def build_bm25(documents):

    print("Tokenizing documents...")

    tokenized_documents = []

    for doc in documents:

        # Repeat title to increase its importance
        text_for_indexing = (
            doc["title"] + " " +
            doc["title"] + " " +
            doc["text"]
)

        tokens = tokenize(text_for_indexing)

        tokenized_documents.append(tokens)


    print("Building BM25 index...")

    bm25 = BM25Okapi(
        tokenized_documents,
        k1=1.5,
        b=0.75
    )

    print("BM25 index ready.")

    return bm25



# -------------------------
# Retrieval
# -------------------------

def retrieve(query, bm25, documents, top_k=TOP_K):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    # Get highest scoring document indexes
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]


    results = []

    for idx in ranked_indexes:

        results.append({

            "score": float(scores[idx]),

            "chunk_id": documents[idx]["chunk_id"],

            "doc_id": documents[idx]["doc_id"],

            "title": documents[idx]["title"],

            "url": documents[idx]["url"],

            "text": documents[idx]["text"]

        })


    return results



# -------------------------
# Main test
# -------------------------

if __name__ == "__main__":

    documents = load_corpus()

    print(f"Loaded {len(documents)} chunks.")

    bm25 = build_bm25(documents)


    # Test query
    query = "Which episode does spongebob discover fire?"

    results = retrieve(
        query,
        bm25,
        documents
    )


    print("\nTop results:\n")

    for i, result in enumerate(results, start=1):

        print(f"Rank {i}")
        print(f"Title: {result['title']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}...")
        print("-" * 50)
