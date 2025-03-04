from abc import ABC
import datasets
from llama_index.core import Document
from tqdm import tqdm
from typing import List


class Loader(ABC):
    def __init__(self):
        pass

    def load(self, documents=None) -> List[Document]:
        docs = [
            Document(text=doc["text"], metadata={"timestamp": doc["timestamp"]})
            for doc in documents
        ]

        return docs

    def load_hf_docs(self) -> List[Document]:
        ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

        docs = [
            Document(text=doc["text"], metadata={"source": doc["source"]})
            for doc in tqdm(ds)
        ]

        return docs
