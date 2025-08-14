import pytest
from utils.validation import InputValidator, ResponseValidator, DataValidator


class TestInputValidator:
    """Тесты для валидатора пользовательского ввода"""

    def test_validate_query_success(self):
        """Тест успешной валидации запроса"""
        valid_query = "Что вы можете сделать для ритейлеров?"
        assert InputValidator.validate_query(valid_query) is True

    def test_validate_query_empty(self):
        """Тест валидации пустого запроса"""
        with pytest.raises(ValueError, match="Запрос не может быть пустым"):
            InputValidator.validate_query("")

        with pytest.raises(ValueError, match="Запрос не может быть пустым"):
            InputValidator.validate_query("   ")

    def test_validate_query_too_short(self):
        """Тест валидации слишком короткого запроса"""
        with pytest.raises(ValueError, match="Запрос слишком короткий"):
            InputValidator.validate_query("Hi")

    def test_validate_query_too_long(self):
        """Тест валидации слишком длинного запроса"""
        long_query = "a" * 1001
        with pytest.raises(ValueError, match="Запрос слишком длинный"):
            InputValidator.validate_query(long_query)

    def test_validate_query_dangerous_content(self):
        """Тест валидации опасного контента"""
        dangerous_queries = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onclick=alert('xss')",
            "eval(malicious_code)",
            "exec(dangerous_code)",
        ]

        for query in dangerous_queries:
            with pytest.raises(ValueError, match="содержит недопустимые элементы"):
                InputValidator.validate_query(query)

    def test_validate_complexity_level_success(self):
        """Тест успешной валидации уровня сложности"""
        valid_levels = ["easy", "medium", "hard"]
        for level in valid_levels:
            assert InputValidator.validate_complexity_level(level) is True

    def test_validate_complexity_level_invalid(self):
        """Тест валидации неверного уровня сложности"""
        with pytest.raises(ValueError, match="Недопустимый уровень сложности"):
            InputValidator.validate_complexity_level("invalid")

    def test_sanitize_query(self):
        """Тест очистки запроса"""
        dirty_query = "  Что   вы   можете<script>   сделать?  "
        clean_query = InputValidator.sanitize_query(dirty_query)
        assert clean_query == "Что вы можетеscript сделать?"
        assert not any(char in clean_query for char in "<>\"'")


class TestResponseValidator:
    """Тесты для валидатора ответов"""

    def test_validate_response_success(self):
        """Тест успешной валидации ответа"""
        valid_response = {
            "answer": "Тестовый ответ",
            "sources": [{"source_file": "test.txt"}],
            "complexity_level": "easy",
        }
        assert ResponseValidator.validate_response(valid_response) is True

    def test_validate_response_missing_fields(self):
        """Тест валидации ответа с отсутствующими полями"""
        incomplete_response = {"answer": "Тестовый ответ"}

        with pytest.raises(ValueError, match="Отсутствует обязательное поле"):
            ResponseValidator.validate_response(incomplete_response)

    def test_validate_response_wrong_types(self):
        """Тест валидации ответа с неверными типами"""
        wrong_answer_type = {"answer": 123, "sources": [], "complexity_level": "easy"}

        with pytest.raises(ValueError, match="должно быть строкой"):
            ResponseValidator.validate_response(wrong_answer_type)

        wrong_sources_type = {
            "answer": "test",
            "sources": "not a list",
            "complexity_level": "easy",
        }

        with pytest.raises(ValueError, match="должно быть списком"):
            ResponseValidator.validate_response(wrong_sources_type)

    def test_validate_sources_success(self):
        """Тест успешной валидации источников"""
        valid_sources = [
            {"source_file": "test1.txt"},
            {"url": "http://example.com"},
            {"title": "Test Document"},
        ]
        assert ResponseValidator.validate_sources(valid_sources) is True

    def test_validate_sources_invalid(self):
        """Тест валидации неверных источников"""
        invalid_sources = ["not a dict", {"invalid": "source"}]

        with pytest.raises(ValueError, match="должен быть словарем"):
            ResponseValidator.validate_sources(invalid_sources)


class TestDataValidator:
    """Тесты для валидатора данных"""

    def test_validate_document_content_success(self):
        """Тест успешной валидации содержимого документа"""
        valid_content = (
            "Это валидное содержимое документа с достаточным количеством текста."
        )
        assert DataValidator.validate_document_content(valid_content) is True

    def test_validate_document_content_empty(self):
        """Тест валидации пустого содержимого"""
        assert DataValidator.validate_document_content("") is False
        assert DataValidator.validate_document_content("   ") is False

    def test_validate_document_content_too_short(self):
        """Тест валидации слишком короткого содержимого"""
        short_content = "Короткий"
        assert DataValidator.validate_document_content(short_content) is False

    def test_validate_document_content_too_long(self):
        """Тест валидации слишком длинного содержимого"""
        long_content = "a" * 100001
        assert DataValidator.validate_document_content(long_content) is False

    def test_validate_metadata_success(self):
        """Тест успешной валидации метаданных"""
        valid_metadata = {"source_file": "test.txt", "other_field": "value"}
        assert DataValidator.validate_metadata(valid_metadata) is True

        valid_metadata_url = {"url": "http://example.com"}
        assert DataValidator.validate_metadata(valid_metadata_url) is True

    def test_validate_metadata_invalid_type(self):
        """Тест валидации метаданных неверного типа"""
        with pytest.raises(ValueError, match="должны быть словарем"):
            DataValidator.validate_metadata("not a dict")

    def test_validate_metadata_missing_required(self):
        """Тест валидации метаданных без обязательных полей"""
        invalid_metadata = {"other_field": "value"}
        with pytest.raises(ValueError, match="должны содержать source_file или url"):
            DataValidator.validate_metadata(invalid_metadata)
