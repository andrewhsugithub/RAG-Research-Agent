from abc import ABC
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.core import Settings
from typing import Optional

import torch


class Retriever(ABC):
    def __init__(
        self,
        collection_name: str,
        embedding_func,
        path: str = "./chroma_test",
        chunk_size: int = 512,
        device: str = "cuda:2",
    ):
        torch.set_default_device(device)
        Settings.llm = None
        Settings.chunk_size = chunk_size

        self.path = path
        self.collection_name = collection_name
        self.embedding = embedding_func

        # Set in subclass
        self.vector_store = None
        self.storage_context = None

    def populate(self, documents) -> None:
        if not self.storage_context or not self.vector_store:
            raise ValueError(
                "Vector store and storage context must be initialized in the subclass."
            )

        VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embedding,
            show_progress=True,
        )
        print(f"✅ Documents successfully added to {self.__class__.__name__}!")

    def retrieve(self, top_k: int) -> None:
        if not self.vector_store:
            raise ValueError("Vector store must be initialized in the subclass.")

        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embedding,
        )
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            sparse_top_k=top_k + 10,
            vector_store_query_mode="hybrid",
        )
        return retriever


class QdrantRetriever(Retriever):
    def __init__(
        self,
        qdrant_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant_client = QdrantClient(path=self.path)

        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.qdrant_client,
            enable_hybrid=True,
            batch_size=100,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )


class ChromaBM25Retriever(Retriever):
    def __init__(self, bm25_path: str = "./bm25_retriever", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bm25_path = bm25_path
        client = chromadb.PersistentClient(path=self.path)
        collection = client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def populate(self, nodes, sparse_top_k: int, lang: str = "english") -> None:
        VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embedding,
            show_progress=True,
        )
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=sparse_top_k,
            stemmer=Stemmer.Stemmer(lang),
            language=lang,
        )
        bm25_retriever.persist(self.bm25_path)
        print(f"✅ Documents successfully added to ChromaDB and BM25!")

    def retrieve(self, top_k: int):
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embedding,
        )
        bm25_retriever = BM25Retriever.from_persist_dir(self.bm25_path)

        retriever = QueryFusionRetriever(
            [index.as_retriever(similarity_top_k=top_k), bm25_retriever],
            similarity_top_k=top_k,
            llm=None,
            num_queries=5,
            retriever_weights=[0.75, 0.25],
            use_async=False,
        )
        return retriever


class PineconeRetriever(Retriever):
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        # TODO: haven't tested this yet
        # Note: Pinecone doesn't support on disk storage, if local development is needed, use Pinecone's docker image
        # Here, we are using the Pinecone's cloud
        super().__init__(*args, **kwargs)

        pc = Pinecone(api_key=pinecone_api_key)
        pc.create_index(
            name=self.collection_name,
            dimension=384,  # depends on the embedding model
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        pc_index = pc.Index(self.collection_name)
        self.vector_store = PineconeVectorStore(
            pinecone_index=pc_index, add_sparse_vector=True, batch_size=100
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
