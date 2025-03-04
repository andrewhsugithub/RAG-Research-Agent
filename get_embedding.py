from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function(
    embedding_model_name: str = "thenlper/gte-small",
    device: str = "cuda:0",
) -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        multi_process=False,
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    return embeddings
