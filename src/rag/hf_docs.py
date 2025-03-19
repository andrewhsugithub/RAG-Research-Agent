from loader import Loader
from retriever import QdrantRetriever
from get_embedding import get_embedding_function


docs = Loader().load_hf_docs()

# if want to use chroma, u need to split into nodes first
# nodes = Splitter(chunk_size=125).split(docs)
# retriever = ChromaBM25Retriever(
#     collection_name="hf_docs",
#     embedding_func=get_embedding_function(),
#     path="../../chroma",
# ).populate(nodes, sparse_top_k=30)

retriever = QdrantRetriever(
    collection_name="hf_docs",
    embedding_func=get_embedding_function(),
    path="../../qdrant",
).populate(documents=docs)
