"""
Evaluate — single command to reproduce all experimental results.

Runs every test question through THREE pipelines:
  1. LLM only          (no retrieval)
  2. BM25 + LLM         (classical retrieval RAG)
  3. Dense + LLM        (embedding retrieval RAG)

...and prints/saves a neat comparison table.

Reproducibility:
  - random / numpy seeds fixed
  - Ollama generation uses temperature=0 and a fixed seed (set in generate.py)
  - BM25 and dense retrieval are both deterministic given a fixed corpus

Usage (single command, reproduces everything with defaults):
    python evaluate.py

Optional flags:
    python evaluate.py --questions data/test_questions.json --top_k 5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from BM25 import load_corpus, build_bm25, retrieve as bm25_retrieve
import dense_retrieval as dense_module
from generate import generate_answer, generate_plain_answer

# -------------------------
# Reproducibility
# -------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass


# -------------------------
# Configuration
# -------------------------

# Resolve paths relative to THIS FILE, not the current working directory --
# this way `python evaluate.py` works the same whether run from the
# project root, from src/, or with a full path, on any OS.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "data" / "test_questions.json"
DEFAULT_TOP_K = 5

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"


# -------------------------
# Retrieval metrics
# -------------------------

def hit_at_k(retrieved_ids, relevant_ids):
    """1 if ANY relevant document appears in the retrieved list, else 0."""
    if not relevant_ids:
        return None
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def precision_at_k(retrieved_ids, relevant_ids):
    """Fraction of retrieved documents that are actually relevant."""
    if not relevant_ids or not retrieved_ids:
        return None
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids, relevant_ids):
    """Fraction of relevant documents that were successfully retrieved."""
    if not relevant_ids:
        return None
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1 / rank of the FIRST relevant document found (0 if none found)."""
    if not relevant_ids:
        return None
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


REFUSAL_PHRASE = "I don't know"  # must match generate.py's required refusal text


def is_refusal(answer_text):
    """Check whether an answer contains the required refusal phrase."""
    return REFUSAL_PHRASE in answer_text.strip().lower()


def refusal_correctness(answer_text, expected_refusal):
    """
    For out-of-scope questions (expected_refusal=True), correct behavior
    is refusing to answer. For normal questions (expected_refusal=False),
    correct behavior is NOT refusing. Returns None if expected_refusal
    wasn't specified (question wasn't marked as an abstention test case).
    """
    if expected_refusal is None:
        return None
    return float(is_refusal(answer_text) == expected_refusal)


def compute_retrieval_metrics(retrieved_chunks, relevant_ids):
    """
    Compute all metrics for one query's retrieval results, at the
    DOCUMENT level -- a retrieved chunk counts as a match if its doc_id
    is in relevant_ids, regardless of which specific chunk from that
    document was retrieved. This is more forgiving than chunk-level
    matching when a single document is split into multiple chunks.
    """
    retrieved_ids = [c["doc_id"] for c in retrieved_chunks]

    return {
        "hit": hit_at_k(retrieved_ids, relevant_ids),
        "precision": precision_at_k(retrieved_ids, relevant_ids),
        "recall": recall_at_k(retrieved_ids, relevant_ids),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids),
    }


def is_abstention(answer):
    """Alias for is_refusal, kept for readability where 'abstention' fits better."""
    return is_refusal(answer)


# -------------------------
# Load test questions
# -------------------------

def load_test_questions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


# -------------------------
# Run one question through all 3 pipelines
# -------------------------

def run_question(question, bm25_index, dense_model, dense_embeddings, documents, top_k,
                  relevant_ids=None, expected_refusal=None):

    # --- LLM only ---
    llm_only_answer = generate_plain_answer(question)

    # --- BM25 + LLM ---
    bm25_chunks = bm25_retrieve(question, bm25_index, documents, top_k=top_k)
    bm25_answer = generate_answer(question, bm25_chunks)
    bm25_metrics = compute_retrieval_metrics(bm25_chunks, relevant_ids)
    bm25_metrics["refusal_correct"] = refusal_correctness(bm25_answer, expected_refusal)

    # --- Dense + LLM ---
    dense_chunks = dense_module.retrieve(question, dense_model, dense_embeddings, documents, top_k=top_k)
    dense_answer = generate_answer(question, dense_chunks)
    dense_metrics = compute_retrieval_metrics(dense_chunks, relevant_ids)
    dense_metrics["refusal_correct"] = refusal_correctness(dense_answer, expected_refusal)

    return {
        "llm_only_answer": llm_only_answer,
        "bm25_top_chunk": bm25_chunks[0]["chunk_id"] if bm25_chunks else "",
        "bm25_answer": bm25_answer,
        "bm25_metrics": bm25_metrics,
        "dense_top_chunk": dense_chunks[0]["chunk_id"] if dense_chunks else "",
        "dense_answer": dense_answer,
        "dense_metrics": dense_metrics,
    }


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-only vs BM25-RAG vs Dense-RAG")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE,
                         help="path to test questions JSON")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    print("Loading test questions...")
    test_set = load_test_questions(args.questions)
    print(f"Loaded {len(test_set)} test questions.\n")

    print("Loading corpus...")
    documents = load_corpus()
    print(f"Loaded {len(documents)} chunks.\n")

    print("Building BM25 index...")
    bm25_index = build_bm25(documents)

    print("\nBuilding dense index (cached after first run)...")
    dense_model, dense_embeddings = dense_module.build_dense_index(documents)

    rows = []

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        relevant_ids = item.get("relevant_doc_ids")
        expected_refusal = item.get("expected_refusal")  # True / False / None (not tested)

        print(f"\n[{i}/{len(test_set)}] {question}")
        start = time.time()

        outputs = run_question(
            question, bm25_index, dense_model, dense_embeddings, documents,
            args.top_k, relevant_ids=relevant_ids, expected_refusal=expected_refusal
        )

        elapsed = time.time() - start
        print(f"  done in {elapsed:.1f}s")

        bm25_m = outputs["bm25_metrics"]
        dense_m = outputs["dense_metrics"]

        rows.append({
            "Question": question,
            "Ground Truth": ground_truth,
            "Expected Refusal": expected_refusal,
            "LLM Only": outputs["llm_only_answer"],
            "BM25+LLM": outputs["bm25_answer"],
            "BM25 Top Chunk": outputs["bm25_top_chunk"],
            "BM25 Hit": bm25_m["hit"],
            "BM25 Precision": bm25_m["precision"],
            "BM25 Recall": bm25_m["recall"],
            "BM25 RR": bm25_m["rr"],
            "BM25 Refusal Correct": bm25_m["refusal_correct"],
            "Dense+LLM": outputs["dense_answer"],
            "Dense Top Chunk": outputs["dense_top_chunk"],
            "Dense Hit": dense_m["hit"],
            "Dense Precision": dense_m["precision"],
            "Dense Recall": dense_m["recall"],
            "Dense RR": dense_m["rr"],
            "Dense Refusal Correct": dense_m["refusal_correct"],
        })

    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)

    # Print a readable version in the terminal (full CSV has full text).
    display_df = df.copy()
    for col in ["LLM Only", "BM25+LLM", "Dense+LLM", "Ground Truth"]:
        display_df[col] = display_df[col].str.slice(0, 60) + "..."

    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.width", 200)
    print(display_df.to_string(index=False))

    # --- Retrieval metrics summary (averaged across all questions) ---
    metric_cols = [
        "BM25 Hit", "BM25 Precision", "BM25 Recall", "BM25 RR",
        "Dense Hit", "Dense Precision", "Dense Recall", "Dense RR",
    ]
    has_metrics = df[metric_cols].notna().any().any()

    if has_metrics:
        print("\n" + "=" * 80)
        print(f"RETRIEVAL METRICS SUMMARY (averaged over {len(df)} questions, top_k={args.top_k})")
        print("=" * 80)

        summary = pd.DataFrame({
            "Method": ["BM25", "Dense"],
            "Hit Rate": [df["BM25 Hit"].mean(), df["Dense Hit"].mean()],
            "Precision@k": [df["BM25 Precision"].mean(), df["Dense Precision"].mean()],
            "Recall@k": [df["BM25 Recall"].mean(), df["Dense Recall"].mean()],
            "MRR": [df["BM25 RR"].mean(), df["Dense RR"].mean()],
        })
        print(summary.to_string(index=False))
    else:
        print("\n(No 'relevant_doc_ids' found in test questions -- "
              "add them to compute retrieval metrics like Precision/Recall/MRR.)")

    # --- Refusal / abstention accuracy (for out-of-scope questions) ---
    has_refusal_tests = df["Expected Refusal"].notna().any()

    if has_refusal_tests:
        print("\n" + "=" * 80)
        print("REFUSAL ACCURACY (out-of-scope / unanswerable questions)")
        print("=" * 80)
        print("Correct behavior: refuse ('I don't know') when expected_refusal=True,")
        print("answer normally when expected_refusal=False.\n")

        refusal_summary = pd.DataFrame({
            "Method": ["BM25+LLM", "Dense+LLM"],
            "Refusal Accuracy": [
                df["BM25 Refusal Correct"].mean(),
                df["Dense Refusal Correct"].mean(),
            ],
            "N tested": [
                df["BM25 Refusal Correct"].notna().sum(),
                df["Dense Refusal Correct"].notna().sum(),
            ],
        })
        print(refusal_summary.to_string(index=False))

    print(f"\nFull results saved to:")
    print(f"  {RESULTS_CSV}")


if __name__ == "__main__":
    main()