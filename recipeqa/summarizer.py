from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document

import chains


class Summarizer(ABC):
    def __call__(self, docs: List[Document], **kwargs) -> str:
        pass


class LangChainDistillSummarizer(Summarizer):
    """Distill chain from Andrew White paper-qa
    https://github.com/whitead/paper-qa/blob/main/paperqa/qaprompts.py
    """

    distill_chain = chains.distill_chain

    def __call__(self, docs: List[Document], query: str) -> str:
        docs_summary = self.distill_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{"context_str": d.page_content, "question": query} for d in docs]
        )
        docs_summary = [s["text"] for s in docs_summary]

        summary = "\n\n".join(
            [
                f"{d.metadata['id']}: {s}"
                for d, s in zip(docs, docs_summary)
                if "not applicable" not in s.lower()
            ]
        )
        return summary
