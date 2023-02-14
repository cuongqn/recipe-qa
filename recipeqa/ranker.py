from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class Ranker(ABC):
    @abstractmethod
    def __call__(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        pass


class LangChainFAISSIndexRanker(Ranker):
    def __init__(self, text_splitter=None):
        self.text_splitter = text_splitter
        if text_splitter is None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500, chunk_overlap=50
            )

    def _split_text(self, docs: List[Document]) -> Tuple[List[str], List[dict]]:
        split_docs = []
        split_metadata = []
        for doc in docs:
            texts = self.text_splitter.split_text(doc.page_content)
            split_docs.extend(texts)
            split_metadata.extend([doc.metadata] * len(texts))
        return split_docs, split_metadata

    def __call__(
        self, query: str, docs: List[Document], top_k: int = 5
    ) -> List[Document]:
        split_texts, split_metadata = self._split_text(docs=docs)
        faiss_index = FAISS.from_texts(
            texts=split_texts, embedding=OpenAIEmbeddings(), metadatas=split_metadata
        )
        top_texts = faiss_index.max_marginal_relevance_search(
            query, k=top_k, fetch_k=10 * top_k
        )
        return top_texts
