import pytest
import os
import tempfile
from unittest.mock import patch, Mock
from core.rag_chain import EoraRAGChain
from core.config import Config
from utils.file_loader import FileLoader
from utils.web_crawler import WebCrawler


class TestIntegration:
    """Интеграционные тесты для всей системы"""

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    @patch("core.rag_chain.FAISS")
    def test_full_rag_pipeline(self, mock_faiss, mock_llm_factory, mock_embeddings):
        """Тест полного RAG pipeline"""
        # Настройка моков
        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        mock_doc = Mock()
        mock_doc.page_content = "EORA делает AI решения для ритейла"
        mock_doc.metadata = {"source_file": "test.txt"}
        mock_vectorstore.similarity_search.return_value = [mock_doc]

        mock_response = Mock()
        mock_response.content = "EORA специализируется на AI решениях для ритейла"
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_provider = Mock()
        mock_provider.get_llm.return_value = mock_llm_instance
        mock_llm_factory.create_provider.return_value = mock_provider

        # Создание временного файла с данными
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("EORA - компания по разработке AI решений для ритейла")
            temp_file = f.name

        try:
            # Создание временной директории
            with tempfile.TemporaryDirectory() as temp_dir:
                # Перемещение файла в временную директорию
                import shutil

                test_file = os.path.join(temp_dir, "test.txt")
                shutil.move(temp_file, test_file)

                # Инициализация RAG chain
                chain = EoraRAGChain()

                # Мокаем file_loader для возврата тестовых данных
                chain.file_loader = Mock()
                chain.file_loader.load_directory.return_value = [mock_doc]

                # Мокаем web_crawler
                chain.web_crawler = Mock()
                chain.web_crawler.crawl_site.return_value = []

                # Загрузка документов
                doc_count = chain.load_documents(temp_dir, include_web=False)
                assert doc_count == 1

                # Генерация ответа
                result = chain.generate_answer("Что делает EORA?", "easy")

                # Проверки
                assert "answer" in result
                assert "sources" in result
                assert "complexity_level" in result
                assert result["complexity_level"] == "easy"
                assert len(result["sources"]) > 0

        finally:
            # Очистка
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_complexity_levels_integration(self, mock_llm_factory, mock_embeddings):
        """Тест интеграции всех уровней сложности"""
        # Настройка моков
        mock_response = Mock()
        mock_response.content = "Тестовый ответ"
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_provider = Mock()
        mock_provider.get_llm.return_value = mock_llm_instance
        mock_llm_factory.create_provider.return_value = mock_provider

        chain = EoraRAGChain()

        # Мокаем поиск документов
        mock_doc = Mock()
        mock_doc.page_content = "Тестовый контент"
        mock_doc.metadata = {"source_file": "test.txt"}
        chain.search_relevant_docs = Mock(return_value=[mock_doc])

        # Тестируем все уровни сложности
        levels = ["easy", "medium", "hard"]
        for level in levels:
            result = chain.generate_answer("Тестовый вопрос", level)
            assert result["complexity_level"] == level
            assert "answer" in result
            assert "sources" in result

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    def test_file_loader_integration(self):
        """Тест интеграции загрузчика файлов"""
        loader = FileLoader()

        # Создание тестового файла
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Тестовое содержимое файла для проверки загрузчика")
            temp_file = f.name

        try:
            # Загрузка файла
            chunks = loader.load_file(temp_file)

            # Проверки
            assert len(chunks) > 0
            assert all(hasattr(chunk, "page_content") for chunk in chunks)
            assert all(hasattr(chunk, "metadata") for chunk in chunks)
            assert all("source_file" in chunk.metadata for chunk in chunks)

        finally:
            os.unlink(temp_file)

    @patch("utils.web_crawler.requests.Session")
    def test_web_crawler_integration(self, mock_session):
        """Тест интеграции веб-краулера"""
        # Настройка мока
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><h1>Test Page</h1><p>Test content for web crawler integration testing with sufficient length</p></body></html>"

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        crawler = WebCrawler(base_url="http://test.com", delay=0.1)

        # Тест парсинга страницы
        page_data = crawler.crawl_page("http://test.com")

        assert page_data is not None
        assert "url" in page_data
        assert "content" in page_data
        assert "metadata" in page_data
        assert len(page_data["content"]) > 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_config_integration(self):
        """Тест интеграции конфигурации"""
        # Тест валидации конфигурации
        assert Config.validate() is True

        # Тест значений по умолчанию
        assert Config.MODEL_PROVIDER == "openai"
        assert Config.CHUNK_SIZE == 1000
        assert Config.SEARCH_K == 5

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_error_handling_integration(self, mock_llm_factory, mock_embeddings):
        """Тест интеграции обработки ошибок"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

        chain = EoraRAGChain()

        # Тест обработки пустого запроса
        with pytest.raises(Exception):  # Может быть LLMError или ValueError
            chain.generate_answer("", "easy")

        # Тест обработки неверного уровня сложности
        with pytest.raises(Exception):  # Может быть LLMError или ValueError
            chain.generate_answer("Тест", "invalid_level")

    def test_performance_monitoring_integration(self):
        """Тест интеграции мониторинга производительности"""
        from utils.performance import PerformanceMonitor, measure_time

        @measure_time
        def test_function():
            import time

            time.sleep(0.1)
            return "test"

        result = test_function()
        assert result == "test"

        # Тест мониторинга памяти (если psutil доступен)
        try:
            memory_usage = PerformanceMonitor.track_memory_usage()
            if memory_usage is not None:
                assert isinstance(memory_usage, (int, float))
                assert memory_usage > 0
        except ImportError:
            pass  # psutil не установлен

    def test_validation_integration(self):
        """Тест интеграции валидации"""
        from utils.validation import InputValidator, ResponseValidator

        # Тест валидации запроса
        valid_query = "Что вы можете сделать для ритейлеров?"
        assert InputValidator.validate_query(valid_query) is True

        # Тест очистки запроса
        dirty_query = "  Тест   запрос  "
        clean_query = InputValidator.sanitize_query(dirty_query)
        assert clean_query == "Тест запрос"

        # Тест валидации ответа
        valid_response = {
            "answer": "Тестовый ответ",
            "sources": [{"source_file": "test.txt"}],
            "complexity_level": "easy",
        }
        assert ResponseValidator.validate_response(valid_response) is True
