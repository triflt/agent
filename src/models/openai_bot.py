from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import json
from .prompts.prompt_v2 import SYSTEM_PROMPT
from .rag_engine import RAGEngine
from ..schemas.response_models import AssistantResponse, ContextInfo, PredictionResponse
from ..config import config


class OpenAIBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing OpenAI bot...")

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found")

        self.client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized successfully")

        self.model = config.llm.main_model
        self.system_prompt = SYSTEM_PROMPT
        self.logger.info(f"Using model: {self.model}")

        # Initialize RAG engine
        self.logger.info("Initializing RAG engine...")
        try:
            self.rag_engine = RAGEngine()
            # Initialize RAG engine with documents
            self.rag_engine.load_and_process_documents()
            self.logger.info("✅ RAG engine initialized successfully")
            self.logger.info("🤖 Bot is ready to handle requests")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize RAG engine: {str(e)}", exc_info=True
            )
            raise

    def generate_response(self, query: str, request_id: int) -> PredictionResponse:
        """
        Generate a response using the OpenAI model with RAG.

        Args:
            query (str): The user's query
            request_id (int): Request identifier

        Returns:
            PredictionResponse: Structured response with answer and reasoning
        """
        try:
            self.logger.info(f"Generating response for query: {query}")

            # Get relevant context and URLs from RAG
            self.logger.info("Retrieving relevant context from RAG...")
            context, urls = self.rag_engine.get_relevant_context(query, num_chunks=5)
            self.logger.info(
                f"Retrieved {len(context)} context chunks and {len(urls)} unique URLs"
            )
            self.logger.info(f"Retrieved chunk: \n{context}")

            # Format context for the model
            context_info = [
                ContextInfo(
                    text=ctx, source_url=urls[i] if i < len(urls) else "unknown"
                )
                for i, ctx in enumerate(context)
            ]

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "system",
                    "content": "Ты должен отдавать ответ следуя схеме AssistantResponse.",
                },
                {
                    "role": "system",
                    "content": f"Найденные документы:\n\n{json.dumps([c.model_dump() for c in context_info], indent=2)}",
                },
                {"role": "user", "content": query},
            ]

            self.logger.info(f"Sending request to OpenAI API using model: {self.model}")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=AssistantResponse,
                temperature=config.llm.main_temperature,
                max_tokens=config.llm.max_tokens,
            )

            parsed_response = response.choices[0].message.parsed
            if not parsed_response:
                raise ValueError("Failed to parse structured response from model")

            self.logger.info(
                f"Received structured response: {parsed_response.model_dump_json(indent=2)}"
            )

            return PredictionResponse(
                id=request_id,
                answer=parsed_response.answer,
                reasoning=parsed_response.reasoning,
                sources=urls,
            )

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    def set_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt"""
        self.system_prompt = new_prompt

    def set_model(self, model_name: str) -> None:
        """Update the model being used"""
        self.model = model_name
