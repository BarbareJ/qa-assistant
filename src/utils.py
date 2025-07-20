import os
import logging
from pathlib import Path
from src.config import config


def get_logger(name: str) -> logging.Logger:
    """get a configured logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_environment() -> str:
    """load and verify environment variables"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        try:
            load_dotenv(env_path, override=True)
        except Exception as e:
            get_logger(__name__).warning(f"error loading .env: {str(e)}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "please add to .env file like: OPENAI_API_KEY=\"sk-...\""
        )
    return api_key


def validate_config() -> None:
    """validate config"""
    if not Path(config.docs_dir).exists():
        raise ValueError(f"doc dir {config.docs_dir} does not exist")
    if config.chunk_size <= config.chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")