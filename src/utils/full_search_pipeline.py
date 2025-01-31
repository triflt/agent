import re
import requests
from bs4 import BeautifulSoup, Tag
import csv
import sys
from tqdm import tqdm
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

# Ignore only the InsecureRequestWarning
warnings.simplefilter("ignore", InsecureRequestWarning)

class SearchResult(BaseModel):
    """
    Individual search result from ITMO news/pages
    """
    title: str
    url: HttpUrl
    description: Optional[str] = None
    date: Optional[datetime] = None
    content: Optional[str] = None

class SearchResponse(BaseModel):
    """
    Complete search response
    """
    query: str
    total_results: int
    results: List[SearchResult]
    search_time: datetime = Field(default_factory=datetime.now)

def clean_article_content(content_block: Tag) -> str:
    """
    Remove unnecessary elements and prepare text:
      - Remove <script> and <style>
      - Convert non-breaking spaces (\xa0) to regular spaces
      - Collapse multiple spaces/newlines
    """
    # 1. Remove <script> and <style> tags
    for unwanted_tag in content_block(["script", "style"]):
        unwanted_tag.decompose()

    # 2. Extract text
    text = content_block.get_text(separator="\n", strip=True)

    # 3. Remove non-breaking spaces
    text = text.replace("\xa0", " ")

    # 4. Collapse multiple newlines and extra spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # 5. Final strip
    text = text.strip()
    return text

def parse_itmo_article_page(html_content: str) -> dict:
    """
    Extracts data from a single ITMO article page:
      - title
      - publication_datetime (e.g. 2025-01-22T11:48:35+03:00)
      - views
      - authors (list of strings)
      - tags (list of strings)
      - cleaned article_text
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Title
    title_tag = soup.select_one("div.article h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Date/time in time[datetime], e.g. 2025-01-22T11:48:35+03:00
    time_tag = soup.select_one("div.news-info-wrapper time")
    publication_datetime_str = time_tag.get("datetime", "").strip() if time_tag else None
    publication_datetime = None
    if publication_datetime_str:
        try:
            publication_datetime = datetime.fromisoformat(publication_datetime_str)
        except ValueError:
            publication_datetime = None

    # Views (example: <span class="icon eye">1049</span>)
    views_span = time_tag.select_one("span.icon.eye") if time_tag else None
    views = views_span.get_text(strip=True) if views_span else None

    # Authors
    authors = []
    for author_block in soup.select(".author-block .author-item"):
        name_tag = author_block.select_one(".about h6 a[rel='author']")
        if name_tag:
            authors.append(name_tag.get_text(strip=True))

    # Tags
    tags = []
    for tag_item in soup.select("ul.tags li a"):
        tags.append(tag_item.get_text(strip=True))

    # Main article text
    article_block = soup.select_one(".content.js-mediator-article")
    article_text = clean_article_content(article_block) if article_block else ""

    return {
        "title": title,
        "publication_datetime": publication_datetime,
        "views": views,
        "authors": authors,
        "tags": tags,
        "article_text": article_text
    }

def parse_itmo_search_page(html_content: str):
    """
    Parses the main ITMO search result page, returning:
      - total_results (integer)
      - articles: list of {title, link, snippet, date (string)}
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Find total results (e.g. "152 результата")
    total_results_str = None
    h2_element = soup.select_one(".weeklyevents h2 span")
    if h2_element:
        total_results_str = h2_element.get_text(strip=True)

    # Convert e.g. "152 результата" -> 152 (if possible)
    total_results = 0
    if total_results_str:
        match = re.search(r"(\d+)", total_results_str)
        if match:
            total_results = int(match.group(1))

    # Parse each result (li.weeklyevent)
    results = []
    for item in soup.select(".weeklyevents ul li.weeklyevent"):
        h4 = item.find("h4")
        if not h4 or not h4.find("a"):
            continue

        # Title & Link
        title_tag = h4.find("a")
        title = title_tag.get_text(strip=True)
        link = title_tag["href"]
        if link.startswith("/"):
            link = "https://news.itmo.ru" + link

        # paragraphs typically contain snippet & date
        paragraphs = item.find_all("p")
        snippet = paragraphs[0].get_text(strip=True) if len(paragraphs) > 0 else None
        date_str = paragraphs[-1].get_text(strip=True) if len(paragraphs) > 1 else None

        results.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "date": date_str  # date as string for now
        })

    return {
        "total_results": total_results,
        "articles": results
    }

def get_search_results(
    query: str,
    max_articles: int = 5,
    max_pages: int = 1
) -> SearchResponse:
    """
    High-level function that:
      - Searches ITMO news for `query`
      - Parses multiple pages if max_pages > 1
      - Collects search results across pages
      - Fetches each article's content (up to max_articles total)
      - Returns a Pydantic SearchResponse
    """
    base_url = "https://news.itmo.ru/ru/search/"
    all_articles = []
    total_found = 0

    for page_num in range(1, max_pages + 1):
        params = {
            "search": query,
            "page": page_num,
            "section": "news"  # важный параметр, учитывая пример ссылок
        }

        response = requests.get(base_url, params=params, verify=False)
        response.raise_for_status()

        search_data = parse_itmo_search_page(response.text)
        
        if page_num == 1:
            total_found = search_data["total_results"]

        # Добавляем все статьи с текущей страницы
        all_articles.extend(search_data["articles"])

        # Если статей на странице нет — прерываем цикл (больше страниц нет)
        if not search_data["articles"]:
            break

    # Обрезаем список статей, если хотим получить не более max_articles
    articles_to_fetch = all_articles[:max_articles]

    results_list = []
    for article_info in articles_to_fetch:
        url = article_info["link"]
        snippet = article_info["snippet"]
        date_str = article_info["date"]

        article_resp = requests.get(url, verify=False)
        article_resp.raise_for_status()
        article_data = parse_itmo_article_page(article_resp.text)

        final_date = article_data["publication_datetime"]

        # Если не удалось вытащить ISO-дату, пробуем распарсить date_str
        if not final_date and date_str:
            match = re.search(r"(\d{2}\.\d{2}\.\d{4})", date_str)
            if match:
                try:
                    final_date = datetime.strptime(match.group(1), "%d.%m.%Y")
                except ValueError:
                    final_date = None

        search_result = SearchResult(
            title=article_data["title"] or article_info["title"],
            url=url,
            description=snippet,
            date=final_date,
            content=article_data["article_text"]
        )
        results_list.append(search_result)

    return SearchResponse(
        query=query,
        total_results=total_found,
        results=results_list
    )