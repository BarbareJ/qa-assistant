from src.cli import main
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    main()