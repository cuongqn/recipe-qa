from typing import List, Optional

from langchain.chains import LLMChain
from langchain.docstore.document import Document

import chains
import fetcher
import ranker
import summarizer
import utils


class Agent:
    def __init__(
        self,
        doc_fetcher: Optional[fetcher.Fetcher] = None,
        doc_ranker: Optional[ranker.Ranker] = None,
        doc_summarizer: Optional[summarizer.Summarizer] = None,
        qa_chain: Optional[LLMChain] = None,
        refine_chain: Optional[LLMChain] = None,
    ):
        self.fetcher = doc_fetcher
        self.ranker = doc_ranker
        self.summarizer = doc_summarizer
        self.qa_chain = qa_chain
        self.refine_chain = refine_chain

        if self.fetcher is None:
            self.fetcher = fetcher.YummlyFetcher()
        if self.ranker is None:
            self.ranker = ranker.LangChainFAISSIndexRanker()
        if self.summarizer is None:
            self.summarizer = summarizer.LangChainDistillSummarizer()
        if qa_chain is None:
            self.qa_chain = chains.qa_chain
        if refine_chain is None:
            self.refine_chain = chains.refine_chain

    def __call__(self, query: str, fetch_k: int = 20, top_k: int = 5):
        docs = self.fetcher(query=query, k=fetch_k)
        top_docs = self.ranker(query=query, docs=docs, top_k=top_k)
        context_str = self.summarizer(docs=top_docs, query=query)
        answer = self._answer(query=query, context_str=context_str)

        bib = self._generate_bib(answer=answer, docs=top_docs)
        return f"Answer: {answer}\n\n\nReferences:\n\n{bib}"

    def _answer(self, query: str, context_str: str) -> str:
        answer = self.qa_chain.run(
            question=query, context_str=context_str, length="about 1000 words"
        )[1:]
        count = 0
        while utils.maybe_is_truncated(answer):
            answer = self.refine_chain.run(
                question=query, existing_answer=answer, context_str=context_str
            )
            count += 1
            if count == 5:
                break
        return answer

    def _generate_bib(self, answer: str, docs: List[Document]) -> str:
        sources = {d.metadata["id"]: d.metadata["url"] for d in docs}
        bib = {}
        for k, s in sources.items():
            if k[:10] in answer:
                bib[k] = s
        bib = "\n\n".join([f"{i+1}. {k}: {c}" for i, (k, c) in enumerate(bib.items())])
        return bib


if __name__ == "__main__":
    agent = Agent()
    answer = agent(
        query="How to combine miso paste with brussel sprouts?", fetch_k=20, top_k=5
    )

