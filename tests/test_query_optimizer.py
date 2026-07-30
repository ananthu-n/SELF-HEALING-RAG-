from app.self_healing.query_optimizer import QueryOptimizer


def main():

    optimizer = QueryOptimizer()

    queries = [

        "What is Self-RAG?",

        "Please explain Self-RAG.",

        "Can you explain GraphRAG?",

        "Tell me about CRAG.",

    ]

    for query in queries:

        plan = optimizer.optimize(

            query=query,

            reason="Testing",

        )

        print()

        print("=" * 60)

        print("Original :", plan.original_query)

        print("Optimized:", plan.optimized_query)


if __name__ == "__main__":

    main()