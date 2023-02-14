from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from langchain.docstore.document import Document
import chains


class Fetcher(ABC):
    @abstractmethod
    def __call__(self, query: str, k: int) -> List[Document]:
        pass


class YummlyFetcher(Fetcher):
    URL = """https://mapi.yummly.com/mapi/v19/content/search?solr.seo_boost=new&q={query}&ignore-taste-pref%3F=true&start=0&maxResult=36&fetchUserCollections=false&allowedContent=single_recipe&allowedContent=suggested_search&allowedContent=related_search&allowedContent=article&allowedContent=generic_cta&guided-search=true&solr.view_type=search_internal"""

    def __call__(self, query: str, k: int = 20) -> List[Document]:
        query = chains.query_chain.run(query)
        query = query.replace(" ", "+")
        url = self.URL.replace("{query}", query)
        page = requests.get(url).json()
        docs = []
        for feed in page["feed"][:k]:
            url = feed["content"]["details"]["directionsUrl"]
            author = feed["content"]["moreContent"]["queryParams"]["authorId"]
            name = feed["content"]["moreContent"]["queryParams"]["id"]
            doc = Document(
                page_content=_url2doc(url),
                metadata={"id": f"{author}-{name}", "url": url},
            )
            docs.append(doc)
        return docs


def _url2doc(url):
    try:
        body = requests.get(url).content
    except:
        return ""

    doc = _html2doc(body)
    return doc


def _html2doc(body):
    soup = BeautifulSoup(body, "html.parser")
    texts = soup.findAll(text=True)
    visible_texts = filter(_tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)


def _tag_visible(element):
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True
