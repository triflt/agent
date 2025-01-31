from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field
import requests
from bs4 import BeautifulSoup
import logging
import json
import re
from datetime import datetime
from urllib.parse import urljoin
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Individual search result from ITMO news/pages"""

    title: str
    url: HttpUrl
    description: Optional[str] = None
    date: Optional[datetime] = None
    content: Optional[str] = None


class SearchResponse(BaseModel):
    """Complete search response"""

    query: str
    total_results: int
    results: List[SearchResult]
    search_time: datetime = Field(default_factory=datetime.now)


class ITMOSearchTool:
    """Tool for searching and parsing ITMO website content"""

    def __init__(self):
        self.base_url = "https://news.itmo.ru"
        self.search_url = f"{self.base_url}/ru/search"  # Changed URL
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _parse_search_results(self, html_content: str) -> List[dict]:
        """Parse search results from HTML content"""
        soup = BeautifulSoup(html_content, "html.parser")
        results = []

        # Find all news items (using the correct selector from search_itmo_news.py)
        news_items = soup.select("article.news-list__item-wrapper")
        logger.info(f"Found {len(news_items)} news items")

        for item in news_items:
            try:
                # Extract title and URL
                title_elem = item.select_one("a.news-list__item-link")
                if not title_elem:
                    continue

                url = urljoin(self.base_url, title_elem.get("href", ""))
                title = title_elem.get_text(strip=True)

                # Extract date
                date_elem = item.select_one("span.news-list__item-date")
                date_str = date_elem.get_text(strip=True) if date_elem else None

                # Extract description
                desc_elem = item.select_one("div.news-list__item-lead")
                description = desc_elem.get_text(strip=True) if desc_elem else None

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "description": description,
                        "date": date_str,
                    }
                )
                logger.debug(f"Parsed news item: {title}")

            except Exception as e:
                logger.error(f"Error parsing news item: {str(e)}", exc_info=True)
                continue

        return results

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string into datetime object"""
        if not date_str:
            return None

        try:
            # Using the date format from the original code
            months_ru = {
                "января": "01",
                "февраля": "02",
                "марта": "03",
                "апреля": "04",
                "мая": "05",
                "июня": "06",
                "июля": "07",
                "августа": "08",
                "сентября": "09",
                "октября": "10",
                "ноября": "11",
                "декабря": "12",
            }

            # Split the date string and clean it
            date_parts = date_str.lower().strip().split()
            if len(date_parts) >= 3:
                day = date_parts[0]
                month = months_ru.get(date_parts[1])
                year = date_parts[2]

                if month:
                    date_str = f"{year}-{month}-{day.zfill(2)}"
                    return datetime.fromisoformat(date_str)

        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {str(e)}")

        return None

    def _parse_page_content(self, url: str) -> Optional[str]:
        """Parse detailed content from a specific page"""
        try:
            logger.info(f"Fetching content from URL: {url}")
            response = requests.get(url, headers=self.headers, verify=False)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Using selectors from parse_itmo_page.py
            article = soup.select_one("article.news-item__content")
            if not article:
                logger.warning("No article content found")
                return None

            # Remove unnecessary elements
            for elem in article.select("div.news-item__share, div.news-item__tags"):
                elem.decompose()

            # Get text content
            text_content = article.get_text(separator="\n", strip=True)
            cleaned_text = self._clean_text(text_content)

            logger.debug(f"Extracted content length: {len(cleaned_text)} chars")
            return cleaned_text

        except Exception as e:
            logger.error(f"Error parsing page {url}: {str(e)}", exc_info=True)
            return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text content"""
        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def search(
        self, query: str, max_results: int = 5, parse_content: bool = True
    ) -> SearchResponse:
        """
        Search ITMO news and pages
        """
        try:
            params = {"q": query, "type": "news"}  # Added type parameter

            logger.info(f"Performing search with params: {params}")

            # Make search request
            response = requests.get(
                self.search_url, params=params, headers=self.headers, verify=False
            )
            response.raise_for_status()

            # Parse HTML response
            search_results = self._parse_search_results(response.text)
            logger.info(f"Parsed {len(search_results)} results from HTML")

            results = []
            for item in search_results[:max_results]:
                try:
                    # Parse date
                    date = self._parse_date(item.get("date"))

                    # Create result object
                    result = SearchResult(
                        title=item["title"],
                        url=item["url"],
                        description=item.get("description"),
                        date=date,
                        content=None,
                    )

                    # Optionally parse full content
                    if parse_content:
                        content = self._parse_page_content(item["url"])
                        if content:
                            result.content = content

                    results.append(result)
                    logger.debug(f"Successfully processed result: {result.title}")

                except Exception as e:
                    logger.error(
                        f"Error processing search result: {str(e)}", exc_info=True
                    )
                    continue

            # Create response object
            search_response = SearchResponse(
                query=query, total_results=len(search_results), results=results
            )

            logger.info(f"Search completed. Found {len(results)} results")
            return search_response

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}", exc_info=True)
            return SearchResponse(query=query, total_results=0, results=[])


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    search_tool = ITMOSearchTool()

    # Example search
    query = "yandex"
    response = search_tool.search(query, max_results=3, parse_content=True)

    # Print results
    print(f"\nSearch results for '{query}':")
    print(f"Total results: {response.total_results}")
    print("\nResults:")

    for i, result in enumerate(response.results, 1):
        print(f"\n{i}. {result.title}")
        print(f"URL: {result.url}")
        print(f"Date: {result.date}")
        print(f"Description: {result.description}")
        print(f"Content length: {len(result.content) if result.content else 0} chars")
