import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util


# -------------------------
# Configuration
# -------------------------

INPUT_FILE = "data/processed_corpus.json"

TOP_K = 5

MODEL_NAME = "all-MiniLM-L6-v2"

# Cached embeddings get saved here after the first run
EMBEDDINGS_CACHE_FILE = "data/embeddings_cache.npy"


# -------------------------
# Load corpus
# -------------------------

def load_corpus():

    with open(INPUT_FILE, encoding="utf-8") as f:
        corpus = json.load(f)

    return corpus["documents"]


# -------------------------
# Build dense index
# -------------------------

def build_dense_index(documents, model_name=MODEL_NAME, cache_file=EMBEDDINGS_CACHE_FILE):

    print(f"Loading embedding model '{model_name}'...")

    model = SentenceTransformer(model_name)

    cache_path = Path(cache_file)

    # If a cached embeddings file exists and matches the corpus size,
    # load it instead of re-embedding everything.
    if cache_path.exists():
        print(f"Found cached embeddings at {cache_path}, loading...")
        embeddings = np.load(cache_path)

        if embeddings.shape[0] == len(documents):
            print("Cache matches corpus size -- skipping re-embedding.")
            return model, embeddings
        else:
            print(
                f"Cache has {embeddings.shape[0]} vectors but corpus has "
                f"{len(documents)} documents -- cache is stale, re-embedding."
            )

    print("Embedding documents...")

    # No tokenizing/stemming/stopword removal here -- the embedding model
    # wants natural, unmodified text. We still repeat the title to boost
    # its weight in the combined text, same idea as the BM25 version.
    texts_for_embedding = [
        doc["title"] + ". " + doc["title"] + ". " + doc["text"]
        for doc in documents
    ]

    embeddings = model.encode(
        texts_for_embedding,
        convert_to_tensor=False,   # numpy array, easier to cache with np.save
        show_progress_bar=True,
        batch_size=32,
    )

    embeddings = np.asarray(embeddings)

    print(f"Saving embeddings to {cache_path} for future runs...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)

    print("Dense index ready.")

    return model, embeddings


# -------------------------
# Retrieval
# -------------------------

def retrieve(query, model, embeddings, documents, top_k=TOP_K):

    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, embeddings)[0]

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

    model, embeddings = build_dense_index(documents)


    # Test query
    query = "Does squidward like his job?"

    results = retrieve(
        query,
        model,
        embeddings,
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