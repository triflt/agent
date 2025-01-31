from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class CustomEmbeddings:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing embeddings model: {model_name}")
        
        # Initialize the model with progress bar enabled
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            show_progress=True  # Enable progress bar
        )
        self.logger.info("✅ Embeddings model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.embed_documents(texts)
            self.logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating document embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            embedding = self.model.embed_query(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise 