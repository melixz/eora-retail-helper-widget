import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import time


class WebCrawler:
    """Класс для парсинга сайта eora.ru"""

    def __init__(self, base_url: str = "https://eora.ru", delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay
        self.visited_urls = set()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def crawl_page(self, url: str) -> Dict[str, Any]:
        """Парсинг одной страницы"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            title = soup.find("title")
            title_text = title.get_text().strip() if title else ""

            for script in soup(["script", "style"]):
                script.decompose()

            text_content = soup.get_text()
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return {
                "url": url,
                "title": title_text,
                "content": text,
                "metadata": {"source": "web", "url": url, "title": title_text},
            }

        except Exception as e:
            print(f"Ошибка при парсинге {url}: {e}")
            return None

    def get_links(self, url: str) -> List[str]:
        """Получение ссылок со страницы"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            links = []

            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                if self.is_valid_url(full_url):
                    links.append(full_url)

            return links

        except Exception as e:
            print(f"Ошибка при получении ссылок с {url}: {e}")
            return []

    def is_valid_url(self, url: str) -> bool:
        """Проверка валидности URL"""
        parsed = urlparse(url)
        return (
            parsed.netloc == urlparse(self.base_url).netloc
            and url not in self.visited_urls
            and not any(
                ext in url.lower()
                for ext in [".pdf", ".jpg", ".png", ".gif", ".css", ".js"]
            )
        )

    def crawl_site(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Парсинг всего сайта"""
        pages_data = []
        urls_to_visit = [self.base_url]

        while urls_to_visit and len(pages_data) < max_pages:
            url = urls_to_visit.pop(0)

            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)

            page_data = self.crawl_page(url)
            if page_data:
                pages_data.append(page_data)

                new_links = self.get_links(url)
                urls_to_visit.extend(new_links)

            time.sleep(self.delay)

        return pages_data
