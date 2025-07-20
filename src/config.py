from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """config class for qa assistant"""
    docs_dir: str = "docs"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    top_k: int = 3
    cache_dir: Optional[str] = ".cache"


config = Config()


def update_config(**kwargs) -> None:
    """update config with new values
    args:
        **kwargs: key-value pairs
    """
    global config
    config = Config(**{**vars(config), **kwargs})