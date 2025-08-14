import pytest
from unittest.mock import Mock, patch
from utils.error_handler import (
    ErrorHandler,
    handle_errors,
    handle_document_load_errors,
    handle_llm_errors,
    handle_webcrawler_errors,
)
from core.exceptions import DocumentLoadError, LLMError, WebCrawlerError


class TestErrorHandler:
    """Тесты для обработчика ошибок"""

    def test_log_and_raise_custom_exception(self):
        """Тест логирования и поднятия кастомного исключения"""
        error = ValueError("Test error")

        with pytest.raises(DocumentLoadError):
            ErrorHandler.log_and_raise(error, "Test context", DocumentLoadError)

    def test_log_and_raise_eora_exception(self):
        """Тест логирования и поднятия EORA исключения"""
        error = DocumentLoadError("Test error")

        with pytest.raises(DocumentLoadError):
            ErrorHandler.log_and_raise(error, "Test context")

    @patch("utils.error_handler.logger")
    def test_log_warning(self, mock_logger):
        """Тест логирования предупреждения"""
        ErrorHandler.log_warning("Test message", "Test context")
        mock_logger.warning.assert_called_once_with("Test context: Test message")

    @patch("utils.error_handler.logger")
    def test_log_info(self, mock_logger):
        """Тест логирования информации"""
        ErrorHandler.log_info("Test message", "Test context")
        mock_logger.info.assert_called_once_with("Test context: Test message")

    def test_safe_execute_success(self):
        """Тест успешного выполнения функции"""
        func = Mock(return_value="success")
        result = ErrorHandler.safe_execute(func, "default", "test context")
        assert result == "success"
        func.assert_called_once()

    def test_safe_execute_failure(self):
        """Тест выполнения функции с ошибкой"""
        func = Mock(side_effect=Exception("Test error"))
        result = ErrorHandler.safe_execute(func, "default", "test context")
        assert result == "default"
        func.assert_called_once()


class TestErrorDecorators:
    """Тесты для декораторов обработки ошибок"""

    def test_handle_document_load_errors_success(self):
        """Тест успешного выполнения с декоратором"""

        @handle_document_load_errors
        def test_func():
            return ["doc1", "doc2"]

        result = test_func()
        assert result == ["doc1", "doc2"]

    def test_handle_document_load_errors_failure(self):
        """Тест обработки ошибки с декоратором"""

        @handle_document_load_errors
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(DocumentLoadError):
            test_func()

    def test_handle_llm_errors_success(self):
        """Тест успешного выполнения LLM декоратора"""

        @handle_llm_errors
        def test_func():
            return {"answer": "test", "sources": []}

        result = test_func()
        assert result == {"answer": "test", "sources": []}

    def test_handle_llm_errors_failure(self):
        """Тест обработки ошибки LLM декоратора"""

        @handle_llm_errors
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(LLMError):
            test_func()

    def test_handle_webcrawler_errors_success(self):
        """Тест успешного выполнения веб-краулер декоратора"""

        @handle_webcrawler_errors
        def test_func():
            return [{"url": "test.com", "content": "test"}]

        result = test_func()
        assert result == [{"url": "test.com", "content": "test"}]

    def test_handle_webcrawler_errors_failure(self):
        """Тест обработки ошибки веб-краулер декоратора"""

        @handle_webcrawler_errors
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(WebCrawlerError):
            test_func()

    def test_custom_error_decorator(self):
        """Тест кастомного декоратора ошибок"""

        @handle_errors(ValueError, "default_value")
        def test_func():
            raise RuntimeError("Test error")

        with pytest.raises(ValueError):
            test_func()
