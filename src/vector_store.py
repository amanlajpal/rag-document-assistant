import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_HF_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
)


def get_embedding_function(
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
):
    """
    Returns the appropriate LangChain embedding function based on provider.
    - 'openai': Uses OpenAIEmbeddings with text-embedding-3-small (as per resume).
    - 'huggingface': Uses local HuggingFace embeddings for a 100% free option.
    """
    if provider.lower() == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the sidebar or in your .env file."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=DEFAULT_OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
        )
    elif provider.lower() == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=DEFAULT_HF_EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


class VectorStoreManager:
    """
    Manages persistent Chroma storage for RAG document retrieval.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_DIR,
        collection_name: str = "rag_documents",
        embedding_function=None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.vector_store: Optional[Chroma] = None

    def initialize_store(self, embedding_function=None):
        if embedding_function:
            self.embedding_function = embedding_function

        if not self.embedding_function:
            raise ValueError("Embedding function must be provided to initialize vector store.")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> int:
        """
        Adds split document chunks to the Chroma vector store.
        Returns the number of documents added.
        """
        if not self.vector_store:
            self.initialize_store()

        self.vector_store.add_documents(documents=documents)
        return len(documents)

    def get_retriever(self, k: int = DEFAULT_TOP_K):
        """
        Returns a LangChain retriever interface for semantic similarity search.
        """
        if not self.vector_store:
            self.initialize_store()

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def count_documents(self) -> int:
        """
        Returns the count of indexed items in the collection.
        """
        if not self.vector_store:
            return 0
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def reset_store(self):
        """
        Deletes and resets persistent Chroma storage.
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        self.vector_store = None
