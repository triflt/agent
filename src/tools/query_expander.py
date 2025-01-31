from typing import List, Optional
from openai import OpenAI
import logging
from pydantic import BaseModel
from .prompt_v1 import QUERY_EXPANSION_PROMPT
from ..config import config

logger = logging.getLogger(__name__)


class ExpandedQueries(BaseModel):
    """Structure for expanded queries"""

    queries: List[str]
    search_strategy: str


class QueryExpander:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.logger = logging.getLogger(__name__)

    def expand_query(self, original_query: str) -> ExpandedQueries:
        """
        Expand original query into multiple search queries

        Args:
            original_query: User's original query

        Returns:
            ExpandedQueries object containing list of queries and search strategy
        """
        try:
            self.logger.info(f"Expanding query: {original_query}")

            response = self.client.beta.chat.completions.parse(
                model=config.llm.expander_model,
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                    {"role": "user", "content": original_query},
                ],
                response_format=ExpandedQueries,
                temperature=config.llm.expander_temperature,
                max_tokens=config.llm.max_tokens,
            )

            expanded = response.choices[0].message.parsed
            if not expanded:
                raise ValueError("Failed to parse structured response")

            self.logger.info(f"Generated {len(expanded.queries)} expanded queries")
            self.logger.debug(f"Expanded queries: {expanded.queries}")
            self.logger.debug(f"Search strategy: {expanded.search_strategy}")

            return expanded

        except Exception as e:
            self.logger.error(f"Error expanding query: {str(e)}")
            # Return original query if expansion fails
            return ExpandedQueries(
                queries=[original_query],
                search_strategy="Using original query due to expansion error",
            )


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize query expander
    expander = QueryExpander(client)

    # Test queries
    test_queries = [
        "В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?\n1. ARWU (Shanghai Ranking)\n2. Times Higher Education (THE)\n3. QS World University Rankings\n4. U.S. News & World Report",
        "В каком году Университет ИТМО был включён в число Национальных исследовательских университетов России?\n1. 2007\n2. 2009\n3. 2011\n4. 2015",
    ]

    for query in test_queries:
        print(f"\nOriginal query: {query}")
        expanded = expander.expand_query(query)
        print("\nExpanded queries:")
        for i, eq in enumerate(expanded.queries, 1):
            print(f"{i}. {eq}")
        print(f"\nSearch strategy: {expanded.search_strategy}")
