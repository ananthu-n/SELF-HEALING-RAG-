# app/ingestion/test_arxiv.py
from pprint import pprint

from app.ingestion.arxiv_client import ArxivClient


def main():
    client = ArxivClient(max_results=3)

    papers = client.search("Retrieval Augmented Generation")

    print(f"\nFound {len(papers)} papers\n")

    for i, paper in enumerate(papers, start=1):
        print("=" * 80)
        print(f"Paper {i}")
        pprint(paper)
        print()


if __name__ == "__main__":
    main()