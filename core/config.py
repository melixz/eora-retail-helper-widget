import os
import logging
from dotenv import load_dotenv
from core.exceptions import ConfigurationError

load_dotenv()


class Config:
    """Конфигурация приложения"""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    DATA_PATH = os.getenv("DATA_PATH", "./data")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "20"))
    CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))

    ENABLE_WEB_CRAWLING = os.getenv("ENABLE_WEB_CRAWLING", "true").lower() == "true"
    WEB_CRAWL_TIMEOUT = int(os.getenv("WEB_CRAWL_TIMEOUT", "30"))

    SEARCH_K = int(os.getenv("SEARCH_K", "5"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        errors = []

        if not cls.OPENAI_API_KEY and cls.MODEL_PROVIDER == "openai":
            errors.append("OPENAI_API_KEY не установлен")

        if not cls.GIGACHAT_API_KEY and cls.MODEL_PROVIDER == "gigachat":
            errors.append("GIGACHAT_API_KEY не установлен")

        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE должен быть положительным числом")

        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP не может быть отрицательным")

        if cls.CRAWL_MAX_PAGES <= 0:
            errors.append("CRAWL_MAX_PAGES должен быть положительным числом")

        if cls.SEARCH_K <= 0:
            errors.append("SEARCH_K должен быть положительным числом")

        if errors:
            raise ConfigurationError(f"Ошибки конфигурации: {'; '.join(errors)}")

        return True

    @classmethod
    def setup_logging(cls):
        """Настройка логирования"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("eora_rag.log")],
        )
