import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import config
from src.utils import get_logger, load_environment

try:
    api_key = load_environment()
except ValueError as e:
    get_logger(__name__).error(str(e))
    raise

logger = get_logger(__name__)


class AnswerGenerator:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt_template = self._get_prompt_template()
        self.chain = self._build_chain()

    def _initialize_llm(self):
        """init llm with proper api key handling"""
        logger.info(f"Using OpenAI model: {config.llm_model}")
        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            openai_api_key=api_key
        )

    def _get_prompt_template(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on context.
            If you don't know the answer, say you don't know. Keep answers concise.

            Context: {context}"""),
            ("human", "{question}")
        ])

    def _build_chain(self):
        return (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
        )

    def generate_answer(self, context, question):
        try:
            return self.chain.invoke({
                "context": context,
                "question": question
            })
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error."