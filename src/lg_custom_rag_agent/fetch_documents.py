"""Document fetching and retrieval for LG Custom RAG Agent."""

import os
from typing import Optional

# Disable Chroma's anonymized telemetry — its bundled posthog client is
# incompatible with newer posthog versions and spams stderr with
# "Failed to send telemetry event" errors. Must be set before chromadb loads.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import bs4  # noqa: E402
import requests
from chromadb.config import Settings  # noqa: E402
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_jina.embeddings import LateChunkEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from lg_custom_rag_agent.config import get_jina_api_key


class JinaEmbeddingsNoLateChunk(LateChunkEmbeddings):
    """Jina embeddings with late_chunking disabled by default."""

    def embed_documents(
        self, texts: list[str], task: str = "retrieval.passage", late_chunking: bool = False
    ) -> list[list[float]]:
        return super().embed_documents(texts, task=task, late_chunking=late_chunking)

    def embed_query(
        self, text: str, task: str = "retrieval.query", late_chunking: bool = False
    ) -> list[float]:
        return super().embed_query(text, task=task, late_chunking=late_chunking)


DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "lilian_weng_posts"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def load_web_page(url: str, bs_kwargs: Optional[dict] = None) -> list[Document]:
    """Fetch and parse a web page into a Document."""
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


def load_documents(urls: Optional[list[str]] = None) -> list[Document]:
    """Load multiple web pages into Documents."""
    urls = urls or DEFAULT_URLS
    docs = [load_web_page(url) for url in urls]
    return [item for sublist in docs for item in sublist]


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks using recursive character splitting."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(documents)


def _get_embeddings() -> JinaEmbeddingsNoLateChunk:
    """Get Jina embeddings instance.

    session=None is safe: LateChunkEmbeddings declares `session` as a required
    field, but its before-mode validator always replaces it with an
    authenticated requests.Session built from the API key.
    """
    return JinaEmbeddingsNoLateChunk(
        session=None, jina_api_key=SecretStr(get_jina_api_key())
    )


def _get_vectorstore(force_reindex: bool = False) -> Chroma:
    """Get or create the Chroma vector store.

    If the collection exists and is non-empty, open it without re-indexing.
    When force_reindex=True, delete the collection and rebuild.
    """
    embeddings = _get_embeddings()

    if force_reindex:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=CHROMA_SETTINGS)
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        documents = load_documents()
        doc_splits = split_documents(documents)
        return Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            client_settings=CHROMA_SETTINGS,
        )

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=CHROMA_SETTINGS,
    )

    if vectorstore._collection.count() == 0:
        documents = load_documents()
        doc_splits = split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            client_settings=CHROMA_SETTINGS,
        )

    return vectorstore


def get_retriever(k: int = 4, force_reindex: bool = False):
    """Get a retriever for the blog posts."""
    vectorstore = _get_vectorstore(force_reindex=force_reindex)
    return vectorstore.as_retriever(search_kwargs={"k": k})


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


retriever_tool = retrieve_blog_posts


def main() -> None:
    """Demo: run a sample query."""
    result = retriever_tool.invoke({"query": "types of reward hacking"})
    print(result)


if __name__ == "__main__":
    main()
