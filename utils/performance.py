import time
import functools
import logging
from typing import Callable
import streamlit as st

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """Декоратор для измерения времени выполнения функции"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} выполнена за {execution_time:.2f} секунд")
        return result

    return wrapper


class PerformanceMonitor:
    """Класс для мониторинга производительности"""

    @staticmethod
    def track_memory_usage():
        """Отслеживание использования памяти"""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Использование памяти: {memory_info.rss / 1024 / 1024:.2f} MB")
            return memory_info.rss / 1024 / 1024
        except ImportError:
            logger.warning("psutil не установлен, мониторинг памяти недоступен")
            return None

    @staticmethod
    def optimize_session_state():
        """Оптимизация session state"""
        if hasattr(st, "session_state"):
            state_size = len(st.session_state)
            logger.info(f"Размер session state: {state_size} элементов")

            if "messages" in st.session_state and len(st.session_state.messages) > 50:
                st.session_state.messages = st.session_state.messages[-30:]
                logger.info("Очищены старые сообщения из session state")

    @staticmethod
    def cache_stats():
        """Статистика кэша"""
        try:
            cache_info = {}
            if hasattr(st.cache_data, "get_stats"):
                cache_info["data_cache"] = st.cache_data.get_stats()
            if hasattr(st.cache_resource, "get_stats"):
                cache_info["resource_cache"] = st.cache_resource.get_stats()

            logger.info(f"Статистика кэша: {cache_info}")
            return cache_info
        except Exception as e:
            logger.warning(f"Не удалось получить статистику кэша: {e}")
            return {}


def with_performance_monitoring(func: Callable) -> Callable:
    """Декоратор для комплексного мониторинга производительности"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Измерение времени
        start_time = time.time()

        memory_before = PerformanceMonitor.track_memory_usage()

        try:
            result = func(*args, **kwargs)

            memory_after = PerformanceMonitor.track_memory_usage()

            execution_time = time.time() - start_time
            logger.info(f"{func.__name__}: время={execution_time:.2f}с")

            if memory_before and memory_after:
                memory_diff = memory_after - memory_before
                logger.info(f"{func.__name__}: изменение памяти={memory_diff:.2f}MB")

            return result

        except Exception as e:
            logger.error(f"Ошибка в {func.__name__}: {e}")
            raise
        finally:
            PerformanceMonitor.optimize_session_state()

    return wrapper
