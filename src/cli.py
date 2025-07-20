import click
from pathlib import Path
from dotenv import load_dotenv
from src.ingestion import load_documents, split_documents
from src.embeddings import EmbeddingManager
from src.llm import AnswerGenerator
from src.config import config, update_config
from src.utils import get_logger, validate_config, load_environment

try:
    load_environment()
except ValueError as e:
    get_logger(__name__).error(str(e))
    raise

logger = get_logger(__name__)

class QAAssistant:
    def __init__(self):
        validate_config()
        self.embedding_manager = EmbeddingManager()
        self.answer_generator = AnswerGenerator()
        self.vector_store_path = Path(config.cache_dir) / "chroma_db"

    def initialize(self, recreate=False):
        if recreate or not self.vector_store_path.exists():
            documents = load_documents()
            chunks = split_documents(documents)
            self.embedding_manager.create_vector_store(chunks)
            self.vector_store_path.parent.mkdir(exist_ok=True)
        else:
            self.embedding_manager.load_vector_store()

    def ask_question(self, question):
        retriever = self.embedding_manager.get_retriever()
        relevant_chunks = retriever.invoke(question)
        return self.answer_generator.generate_answer(relevant_chunks, question)

@click.command()
@click.option("--question", "-q", required=True, help="Question to answer")
@click.option("--recreate", is_flag=True, help="Recreate vector store")
def main(question, recreate):
    try:
        assistant = QAAssistant()
        assistant.initialize(recreate=recreate)
        answer = assistant.ask_question(question)
        click.echo(f"\nAnswer: {answer}\n")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        click.echo(f"Error: {str(e)}", err=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo("Error: Failed to generate answer", err=True)

if __name__ == "__main__":
    main()