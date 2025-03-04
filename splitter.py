from abc import ABC
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from typing import List

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


class Splitter(ABC):
    def __init__(self, chunk_size: int):
        self.parser = MarkdownNodeParser(chunk_size=chunk_size)

    def split(self, documents: List[Document]) -> List[BaseNode]:
        nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)
        print(f"Number of nodes after splitting: {len(nodes)}")
        return nodes
