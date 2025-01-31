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
from ..tools.query_expander import QueryExpander
from openai import OpenAI
from ..config import config


class RAGEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing custom embeddings model...")
        self.embeddings = CustomEmbeddings(model_name=config.rag.embeddings_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.vector_store = None
        self.persist_directory = "data/chroma_db"
        self.min_doc_length = config.rag.min_doc_length
        self.min_chunk_length = config.rag.min_chunk_length
        self.query_expander = QueryExpander(OpenAI())

    def load_and_process_documents(
        self, csv_path: str = "data/processed/texts_final.csv"
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
            df = df[df["content"].str.len() >= self.min_doc_length]
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
            df = df[df["content"].str.len() >= self.min_doc_length]
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
        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.page_content) >= self.min_chunk_length
        ]
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
        self, query: str, num_chunks: int = None
    ) -> tuple[List[str], List[str]]:
        """
        Retrieve relevant document chunks for a query using expanded queries
        """
        num_chunks = num_chunks or config.rag.num_chunks
        self.logger.info(f"Processing query: {query}")

        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call load_and_process_documents first."
            )

        all_docs = []
        if config.rag.use_query_expansion:
            # Use query expansion
            expanded = self.query_expander.expand_query(query)
            self.logger.info(f"Generated {len(expanded.queries)} expanded queries")
            self.logger.debug(f"Search strategy: {expanded.search_strategy}")

            for expanded_query in expanded.queries:
                docs = self.vector_store.similarity_search(expanded_query, k=num_chunks)
                all_docs.extend(docs)
        else:
            # Use original query directly
            self.logger.info("Query expansion disabled, using original query")
            docs = self.vector_store.similarity_search(query, k=num_chunks)
            all_docs.extend(docs)

        # Remove duplicates and get top chunks
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        top_docs = unique_docs[:num_chunks]

        self.logger.info(f"Found {len(top_docs)} unique relevant chunks")

        # Extract content and URLs
        contexts = [doc.page_content for doc in top_docs]

        # Get unique URLs with their frequency count
        url_frequency = {}
        for doc in all_docs:  # Use all_docs to count frequency across all results
            url = doc.metadata.get("url", "https://itmo.ru")
            if url.startswith("http"):
                url_frequency[url] = url_frequency.get(url, 0) + 1

        # Sort URLs by frequency and take top N
        urls = sorted(url_frequency.items(), key=lambda x: x[1], reverse=True)[
            : config.rag.max_links
        ]
        urls = [url for url, _ in urls]

        return contexts, urls
