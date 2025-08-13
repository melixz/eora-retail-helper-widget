import os
from dotenv import load_dotenv

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

    SEARCH_K = int(os.getenv("SEARCH_K", "5"))

    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        if not cls.OPENAI_API_KEY and cls.MODEL_PROVIDER == "openai":
            raise ValueError("OPENAI_API_KEY не установлен")

        if not cls.GIGACHAT_API_KEY and cls.MODEL_PROVIDER == "gigachat":
            raise ValueError("GIGACHAT_API_KEY не установлен")

        return True
