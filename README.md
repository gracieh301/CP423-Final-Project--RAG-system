# CP423-Final-Project--RAG-system

A Retrieval-Augmented Generation (RAG) system built for course CP423. This project explores and compares two information retrieval approaches for answering questions over a custom SpongeBob SquarePants knowledge corpus:

- **BM25** sparse lexical retrieval
- **Dense Retrieval** semantic retrieval using Sentence Transformers

The system crawls and preprocesses information from the SpongeBob SquarePants Fandom Wiki, builds a retrieval corpus, retrieves relevant documents for user questions, and evaluates retrieval performance using standard information-retrieval metrics.

---
## Prerequisites
1. Make sure you have Meta Llama 3.2 installed and running

## How to run
1. use the command pip install requirements.txt in your python environment
2. run crawler2.py to generate the corpus (if you already have the corpus, skip to step 5.)
3. Next run clean_corpus.py 
4. After that run preprocess.py
5. To test individual questions, run either run_bm25.py or run_dense.py
6. To run and evaluate both systems on all test questions navigate to src and run "evaluate.py" in terminal
7. This will generate a csv file containing all evaluation results




