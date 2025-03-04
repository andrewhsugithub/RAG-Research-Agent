from model.llm import AnswerLLM
from retriever import QdrantRetriever, ChromaBM25Retriever

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from get_embedding import get_embedding_function

import os, torch

import argparse


def main(query, path="tmp", collection="hf_docs", qdrant=True, chroma=False):
    if qdrant and chroma:
        raise ValueError("Only one of Qdrant or Chroma can be used at a time")

    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

    torch.set_default_device(DEVICE)

    query_text = query
    print("query_text:", query_text)

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = get_embedding_function(EMBEDDING_MODEL_NAME, DEVICE)

    if qdrant:
        retriever = QdrantRetriever(
            collection_name=collection,
            embedding_func=embedding_model,
            path=path,
        ).retrieve(top_k=20)
    elif chroma:
        retriever = ChromaBM25Retriever(
            collection_name=collection,
            embedding_func=embedding_model,
            path=path,
        ).retrieve(top_k=20)

    print("✅ Retrieved Documents")

    if torch.cuda.is_available():
        # to disable multi-processing in reranker
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE.split(":")[-1]
    RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
    rerank = FlagEmbeddingReranker(
        model=RERANKER_NAME,
        top_n=5,
        use_fp16=True,
    )
    query_engine = RetrieverQueryEngine(retriever, node_postprocessors=[rerank])

    response = query_engine.query(query_text)
    relevant_docs = response.source_nodes
    print("✅ Reranked Documents")

    context = "\n\n---\n\n".join([doc.node.get_content() for doc in relevant_docs])

    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    # MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    model = AnswerLLM(MODEL_NAME, device=DEVICE)
    answer = model.generate(query_text, context)
    print("==================================Answer==================================")
    print(f"{answer}")
    print(
        "==================================Source docs=================================="
    )
    for i, doc in enumerate(relevant_docs):
        print(
            f"Document {i+1}------------------------------------------------------------"
        )
        print(doc.node.get_content())
        print(doc.node.get_metadata_str())
        print("score:", doc.get_score())
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query text to search for in the document collection",
    )
    parser.add_argument("--path", type=str, help="Path to the document collection")
    parser.add_argument(
        "--collection", type=str, help="Name of the document collection"
    )
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant retriever")
    parser.add_argument(
        "--chroma", action="store_true", help="Use ChromaBM25 retriever"
    )
    # TODO: support Pinecone
    args = parser.parse_args()

    main(**vars(args))
