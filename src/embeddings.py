from pathlib import Path
from typing import List, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import config
from src.utils import get_logger

import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    allow_reset=True
))
logger = get_logger(__name__)


class EmbeddingManager:
    """manages embedding models and vector stores"""

    def __init__(self) -> None:
        """init embedding manager"""
        self.embedding_model = self._initialize_embedding_model()
        self.vector_store = None

    def _initialize_embedding_model(self) -> Any:
        """init appropriate embedding model

        returns:
            inited embedding model
        """
        if config.embedding_model.startswith("text-embedding"):
            logger.info(f"using openai embeddings: {config.embedding_model}")
            return OpenAIEmbeddings(
                model=config.embedding_model,
                show_progress_bar=True
            )

        logger.info(f"using huggingface embeddings: {config.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """create and store vector index

        args:
            chunks: List of doc chunks to index
        """
        logger.info("creating vector store")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=str(Path(config.cache_dir) / "chroma_db")
        )
        logger.info("vector store created successfully")

    def load_vector_store(self) -> None:
        """load vector store from disk"""
        self.vector_store = Chroma(
            persist_directory=str(Path(config.cache_dir) / "chroma_db"),
            embedding_function=self.embedding_model
        )
        logger.info("Vector store loaded from disk")

    def get_retriever(self) -> Any:
        """get a retriever from vector store

        returns:
            confged retriever instance

        raises:
            ValueError: if vector store is not inited
        """
        if not self.vector_store:
            raise ValueError("vector store is not initialized")

        return self.vector_store.as_retriever(
            search_kwargs={"k": config.top_k}
        )