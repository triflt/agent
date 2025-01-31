from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import os
from pathlib import Path
import logging
import pandas as pd
from pydantic import HttpUrl
from .embeddings import CustomEmbeddings


class RAGEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing custom embeddings model...")
        self.embeddings = CustomEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.vector_store = None
        self.persist_directory = "data/chroma_db"
        self.min_doc_length = 400  # Minimum document length
        self.min_chunk_length = 300  # Minimum chunk length

    def load_and_process_documents(
        self, csv_path: str = "data/processed/texts.csv"
    ) -> None:
        """Load documents from CSV, split into chunks, and create vector store"""

        # Try to load existing vector store
        if os.path.exists(self.persist_directory):
            self.logger.info("Found existing vector store, loading...")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                self.logger.info("Successfully loaded existing vector store")
                return
            except Exception as e:
                self.logger.warning(
                    f"Error loading existing vector store: {e}. Creating new one..."
                )

        self.logger.info(f"Loading documents from CSV: {csv_path}")

        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Filter out short documents
            original_count = len(df)
            df = df[df['content'].str.len() >= self.min_doc_length]
            filtered_count = len(df)
            
            if filtered_count < original_count:
                self.logger.info(
                    f"Filtered out {original_count - filtered_count} documents shorter than "
                    f"{self.min_doc_length} characters"
                )
            
        except FileNotFoundError:
            # If CSV doesn't exist, process raw texts first
            self.logger.info("CSV not found. Processing raw texts first !!!!!!!!")
            from ..utils.text_processor import process_texts_to_csv

            df = process_texts_to_csv()
            # Apply filtering to processed texts
            original_count = len(df)
            df = df[df['content'].str.len() >= self.min_doc_length]
            filtered_count = len(df)
            
            if filtered_count < original_count:
                self.logger.info(
                    f"Filtered out {original_count - filtered_count} documents shorter than "
                    f"{self.min_doc_length} characters"
                )

        documents = []

        # Process each row
        for _, row in df.iterrows():
            # Create document with metadata including both URL and source
            doc = Document(
                page_content=row["content"],
                metadata={"url": row["url"], "source": row["url"]},
            )
            documents.append(doc)

        self.logger.info(f"Loaded {len(documents)} documents")

        # Split documents into chunks while preserving metadata
        chunks = self.text_splitter.split_documents(documents)
        original_chunk_count = len(chunks)
        
        # Filter out short chunks
        chunks = [chunk for chunk in chunks if len(chunk.page_content) >= self.min_chunk_length]
        filtered_chunk_count = len(chunks)
        
        if filtered_chunk_count < original_chunk_count:
            self.logger.info(
                f"Filtered out {original_chunk_count - filtered_chunk_count} chunks shorter than "
                f"{self.min_chunk_length} characters"
            )
        
        self.logger.info(f"Created {len(chunks)} valid chunks from documents")

        # Create vector store
        self.logger.info("Creating new vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.logger.info("Vector store created and persisted successfully")

    def get_relevant_context(
        self, query: str, num_chunks: int = 3
    ) -> tuple[List[str], List[str]]:
        """
        Retrieve relevant document chunks for a query

        Returns:
            tuple: (contexts, urls)
        """
        self.logger.info(f"Searching for relevant context for query: {query}")

        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call load_and_process_documents first."
            )

        # Search for relevant documents
        docs = self.vector_store.similarity_search(query, k=num_chunks)
        self.logger.info(f"Found {len(docs)} relevant chunks")

        # Extract content and URLs
        contexts = [doc.page_content for doc in docs]

        # Get unique URLs and ensure they're valid
        unique_urls = set()
        for doc in docs:
            url = doc.metadata.get("url", "https://itmo.ru")
            if url.startswith("http"):  # Only add valid URLs
                unique_urls.add(url)

        urls = list(unique_urls)  # Just use strings instead of HttpUrl objects

        self.logger.debug(f"Retrieved contexts: {contexts}")
        self.logger.debug(f"Retrieved URLs: {urls}")

        return contexts, urls
