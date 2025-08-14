import logging
import functools
from typing import Any, Callable, Optional
from core.exceptions import (
    EoraRAGException,
    DocumentLoadError,
    VectorStoreError,
    LLMError,
    WebCrawlerError,
)

logger = logging.getLogger(__name__)


def handle_errors(
    exception_type: type = EoraRAGException,
    default_return: Any = None,
    log_level: str = "error",
):
    """Декоратор для обработки ошибок"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_func = getattr(logger, log_level.lower())
                log_func(f"Ошибка в {func.__name__}: {e}")

                if isinstance(e, EoraRAGException):
                    raise
                else:
                    raise exception_type(f"Ошибка в {func.__name__}: {e}")

        return wrapper

    return decorator


def handle_document_load_errors(func: Callable) -> Callable:
    """Декоратор для обработки ошибок загрузки документов"""
    return handle_errors(DocumentLoadError, [])(func)


def handle_vectorstore_errors(func: Callable) -> Callable:
    """Декоратор для обработки ошибок векторного хранилища"""
    return handle_errors(VectorStoreError, [])(func)


def handle_llm_errors(func: Callable) -> Callable:
    """Декоратор для обработки ошибок LLM"""
    return handle_errors(
        LLMError, {"answer": "Ошибка генерации ответа", "sources": []}
    )(func)


def handle_webcrawler_errors(func: Callable) -> Callable:
    """Декоратор для обработки ошибок веб-краулера"""
    return handle_errors(WebCrawlerError, [])(func)


class ErrorHandler:
    """Класс для централизованной обработки ошибок"""

    @staticmethod
    def log_and_raise(
        error: Exception, context: str, exception_type: type = EoraRAGException
    ):
        """Логирование и поднятие исключения"""
        logger.error(f"{context}: {error}")
        if isinstance(error, EoraRAGException):
            raise error
        else:
            raise exception_type(f"{context}: {error}")

    @staticmethod
    def log_warning(message: str, context: Optional[str] = None):
        """Логирование предупреждения"""
        full_message = f"{context}: {message}" if context else message
        logger.warning(full_message)

    @staticmethod
    def log_info(message: str, context: Optional[str] = None):
        """Логирование информации"""
        full_message = f"{context}: {message}" if context else message
        logger.info(full_message)

    @staticmethod
    def safe_execute(func: Callable, default_return: Any = None, context: str = ""):
        """Безопасное выполнение функции с возвратом значения по умолчанию"""
        try:
            return func()
        except Exception as e:
            logger.error(f"Ошибка при выполнении {context}: {e}")
            return default_return
