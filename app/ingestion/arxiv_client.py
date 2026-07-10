import arxiv


class ArxivClient:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.client = arxiv.Client()

    def search(self, query: str):
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = self.client.results(search)

        from app.models.paper import Paper

        papers = []

        for result in results:
            papers.append(
                Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    summary=result.summary,
                    published=result.published,
                    pdf_url=result.pdf_url,
                    entry_id=result.entry_id,
                    categories=result.categories,
                )
            )

        return papers