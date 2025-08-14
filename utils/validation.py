import re
from typing import List, Dict, Any


class InputValidator:
    """Класс для валидации пользовательского ввода"""

    @staticmethod
    def validate_query(query: str) -> bool:
        """Валидация пользовательского запроса"""
        if not query or not query.strip():
            raise ValueError("Запрос не может быть пустым")

        if len(query.strip()) < 3:
            raise ValueError("Запрос слишком короткий (минимум 3 символа)")

        if len(query) > 1000:
            raise ValueError("Запрос слишком длинный (максимум 1000 символов)")

        dangerous_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"exec\s*\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Запрос содержит недопустимые элементы")

        return True

    @staticmethod
    def validate_complexity_level(level: str) -> bool:
        """Валидация уровня сложности"""
        valid_levels = ["easy", "medium", "hard"]
        if level not in valid_levels:
            raise ValueError(
                f"Недопустимый уровень сложности. Допустимые: {valid_levels}"
            )
        return True

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Очистка пользовательского запроса"""
        query = re.sub(r"\s+", " ", query.strip())

        query = re.sub(r'[<>"\']', "", query)

        return query


class ResponseValidator:
    """Класс для валидации ответов системы"""

    @staticmethod
    def validate_response(response: Dict[str, Any]) -> bool:
        """Валидация структуры ответа"""
        required_fields = ["answer", "sources", "complexity_level"]

        for field in required_fields:
            if field not in response:
                raise ValueError(f"Отсутствует обязательное поле: {field}")

        if not isinstance(response["answer"], str):
            raise ValueError("Поле 'answer' должно быть строкой")

        if not isinstance(response["sources"], list):
            raise ValueError("Поле 'sources' должно быть списком")

        if not isinstance(response["complexity_level"], str):
            raise ValueError("Поле 'complexity_level' должно быть строкой")

        return True

    @staticmethod
    def validate_sources(sources: List[Dict[str, Any]]) -> bool:
        """Валидация источников"""
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ValueError(f"Источник {i} должен быть словарем")

            if not any(key in source for key in ["source_file", "url", "title"]):
                raise ValueError(f"Источник {i} должен содержать идентификатор")

        return True


class DataValidator:
    """Класс для валидации данных документов"""

    @staticmethod
    def validate_document_content(content: str) -> bool:
        """Валидация содержимого документа"""
        if not content or not content.strip():
            return False

        if len(content.strip()) < 10:
            return False

        if len(content) > 100000:  # 100KB текста
            return False

        return True

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Валидация метаданных документа"""
        if not isinstance(metadata, dict):
            raise ValueError("Метаданные должны быть словарем")

        if "source_file" not in metadata and "url" not in metadata:
            raise ValueError("Метаданные должны содержать source_file или url")

        return True
