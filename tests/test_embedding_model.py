from app.embeddings.model import EmbeddingModel


model = EmbeddingModel()

vectors = model.encode(
    [
        "Self Healing RAG",
        "Large Language Models"
    ]
)

print(vectors.shape)