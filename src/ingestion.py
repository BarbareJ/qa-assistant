from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_documents() -> List[Dict[str, Any]]:
    """load and process docs from confged dir

    returns:
        list of loaded docs
    """
    documents = []

    for file_path in Path(config.docs_dir).glob("*"):
        try:
            if file_path.suffix == ".txt":
                loader = TextLoader(str(file_path))
            elif file_path.suffix == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                continue

            documents.extend(loader.load())
            logger.info(f"loaded doc: {file_path.name}")
        except Exception as e:
            logger.warning(f"failed to load {file_path}: {str(e)}")

    logger.info(f"tot docs loaded: {len(documents)}")
    return documents


def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """split docs into chunks for processing

    args:
        docs: list of docs to split

    returns:
        list of doc chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks