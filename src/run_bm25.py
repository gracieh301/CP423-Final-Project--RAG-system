from BM25 import load_corpus, build_bm25, retrieve
from generate import generate_answer


def main():

    print("Loading corpus...")

    documents = load_corpus()

    print(f"Loaded {len(documents)} chunks.\n")


    bm25 = build_bm25(documents)


    while True:

        query = input("\nQuestion (or 'quit'): ")

        if query.lower() == "quit":
            break


        results = retrieve(
            query,
            bm25,
            documents
        )


        print("\nRetrieved chunks:\n")

        for i, result in enumerate(results, start=1):

            print(f"Rank {i}")
            print(f"Title: {result['title']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Score: {result['score']:.4f}")
            print("-" * 40)


        print("\nGenerating answer...\n")

        answer = generate_answer(query, results)

        print(answer)


if __name__ == "__main__":
    main()