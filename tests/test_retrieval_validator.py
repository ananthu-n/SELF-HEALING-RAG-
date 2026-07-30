from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.validator import RetrievalValidator


retriever = DenseRetriever()

validator = RetrievalValidator()

result = retriever.retrieve(
    "What is Self-RAG?",
    top_k=5,
)

validation = validator.validate(result)

print()

print(validation.model_dump())