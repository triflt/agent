from typing import Dict, Any
from pydantic import BaseModel


class RAGConfig(BaseModel):
    """RAG Engine configuration"""

    num_chunks: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_doc_length: int = 400
    min_chunk_length: int = 300
    embeddings_model: str = "intfloat/multilingual-e5-small"
    max_links: int = 3
    use_query_expansion: bool = False


class LLMConfig(BaseModel):
    """LLM configuration"""

    main_model: str = "gpt-4o-2024-08-06"
    expander_model: str = "gpt-4o-2024-08-06"
    main_temperature: float = 0.1
    expander_temperature: float = 0.3
    max_tokens: int = 500


class Config(BaseModel):
    """Global configuration"""

    rag: RAGConfig = RAGConfig()
    llm: LLMConfig = LLMConfig()


# Default configuration
config = Config()
