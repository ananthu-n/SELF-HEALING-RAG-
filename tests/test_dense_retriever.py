from app.retrieval.dense_retriever import DenseRetriever


def main():

    retriever = DenseRetriever()

    results = retriever.retrieve(
        query="What is Self-RAG?",
        top_k=5,
    )

    print("\nQUERY")
    print(results.query)

    print("\nTOTAL")
    print(results.total_results)

    print("\nRESULTS")

    for i, chunk in enumerate(results.retrieved_chunks, start=1):

        print("-" * 80)

        print(f"Rank      : {i}")
        print(f"Score     : {chunk.score:.4f}")
        print(f"Paper ID  : {chunk.paper_id}")
        print(f"Page      : {chunk.page_number}")
        print(f"Chunk ID  : {chunk.chunk_id}")

        preview = chunk.text.replace("\n", " ")

        print(f"Text      : {preview[:300]}...")


if __name__ == "__main__":
    main()