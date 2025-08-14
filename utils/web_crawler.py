import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import time
from utils.error_handler import ErrorHandler, handle_webcrawler_errors


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
        ErrorHandler.log_info(f"Инициализирован WebCrawler для {base_url}")

    def crawl_page(self, url: str) -> Dict[str, Any]:
        """Парсинг одной страницы"""
        max_retries = 3
        timeout = 30

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                title = soup.find("title")
                title_text = title.get_text().strip() if title else ""

                for script in soup(["script", "style"]):
                    script.decompose()

                text_content = soup.get_text()
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                if len(text.strip()) < 50:
                    ErrorHandler.log_warning(f"Мало контента на странице {url}")
                    return None

                return {
                    "url": url,
                    "title": title_text,
                    "content": text,
                    "metadata": {"source": "web", "url": url, "title": title_text},
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    ErrorHandler.log_warning(
                        f"Попытка {attempt + 1} не удалась для {url}: {e}. Повторяем..."
                    )
                    time.sleep(2**attempt)
                else:
                    ErrorHandler.log_warning(
                        f"Не удалось загрузить {url} после {max_retries} попыток: {e}"
                    )
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
            ErrorHandler.log_warning(f"Ошибка при получении ссылок с {url}: {e}")
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

    @handle_webcrawler_errors
    def crawl_site(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Парсинг всего сайта"""
        pages_data = []
        urls_to_visit = [self.base_url]

        # Добавляем специфичные URL из файла
        specific_urls = self._load_specific_urls()
        urls_to_visit.extend(specific_urls)

        ErrorHandler.log_info(
            f"Начинаем парсинг сайта {self.base_url}, максимум {max_pages} страниц"
        )
        ErrorHandler.log_info(
            f"Добавлено {len(specific_urls)} специфичных URL для парсинга"
        )

        while urls_to_visit and len(pages_data) < max_pages:
            url = urls_to_visit.pop(0)

            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)

            page_data = self.crawl_page(url)
            if page_data:
                pages_data.append(page_data)

                # Получаем новые ссылки только с основной страницы
                if url == self.base_url:
                    new_links = self.get_links(url)
                    urls_to_visit.extend(new_links)

            time.sleep(self.delay)

        ErrorHandler.log_info(f"Парсинг завершен. Обработано {len(pages_data)} страниц")
        return pages_data

    def _load_specific_urls(self) -> List[str]:
        """Загрузка специфичных URL из файла"""
        urls = []
        try:
            import os

            urls_file = os.path.join("data", "eora_cases_urls.txt")
            if os.path.exists(urls_file):
                with open(urls_file, "r", encoding="utf-8") as f:
                    urls = [line.strip() for line in f if line.strip()]
                ErrorHandler.log_info(f"Загружено {len(urls)} URL из файла {urls_file}")
            else:
                ErrorHandler.log_warning(f"Файл {urls_file} не найден")
        except Exception as e:
            ErrorHandler.log_warning(f"Ошибка при загрузке URL из файла: {e}")

        return urls
