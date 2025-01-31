from typing import List, Optional
from openai import OpenAI
import logging
from pydantic import BaseModel
from ..config import config

logger = logging.getLogger(__name__)


class ExpandedQueries(BaseModel):
    """Structure for expanded queries"""

    queries: List[str]
    search_strategy: str


QUERY_EXPANSION_PROMPT = """
Ты эксперт по информационному поиску в базе документов ИТМО. 
Твоя задача - переформулировать запрос пользователя для поиска релевантной информации.

Проанализируй запрос и создай 2-3 поисковых запроса, которые:
1. Охватывают разные аспекты вопроса
2. Используют синонимы и связанные термины
3. Учитывают возможные формулировки в документах

Правила:
- Если в вопросе есть варианты ответов, включи их ключевые термины
- Используй ключевые слова, где уместно
- Каждый запрос должен быть на русском языке (за исключением английских названий)
- Запросы должны быть краткими и точными

Верни ответ в формате JSON:
{
    "queries": ["запрос1", "запрос2", "запрос3"],
    "search_strategy": "краткое описание стратегии поиска"
}

Пример запроса с вариантами ответов:
"В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?
1. ARWU (Shanghai Ranking)
2. Times Higher Education (THE)
3. QS World University Rankings
4. U.S. News & World Report"

Пример ответа:
{
    "queries": [
        "ИТМО топ-400 мировой рейтинг 2021",
        "ИТМО ARWU Shanghai Ranking Times Higher Education QS",
        "университет ИТМО рейтинг университетов 2021"
    ],
    "search_strategy": "Поиск упоминаний о достижении ИТМО топ-400 в 2021 году и проверка всех основных рейтингов"
}
"""


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
