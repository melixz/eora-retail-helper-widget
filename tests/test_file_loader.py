import pytest
from unittest.mock import Mock, patch
from utils.file_loader import FileLoader
from langchain_core.documents import Document


class TestFileLoader:
    """Тесты для загрузчика файлов"""

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    def test_init(self):
        """Тест инициализации"""
        loader = FileLoader()
        assert loader.text_splitter is not None

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    def test_init_with_custom_params(self):
        """Тест инициализации с кастомными параметрами"""
        loader = FileLoader(chunk_size=500, chunk_overlap=100)
        assert loader.text_splitter._chunk_size == 500
        assert loader.text_splitter._chunk_overlap == 100

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("utils.file_loader.TextLoader")
    def test_load_txt_file(self, mock_text_loader):
        """Тест загрузки TXT файла"""
        mock_loader_instance = Mock()
        mock_text_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = [
            Document(page_content="Test content", metadata={})
        ]

        loader = FileLoader()
        loader.text_splitter = Mock()
        loader.text_splitter.split_documents.return_value = [
            Document(page_content="Test content", metadata={})
        ]

        result = loader.load_file("test.txt")

        assert len(result) == 1
        assert result[0].metadata["source_file"] == "test.txt"
        mock_text_loader.assert_called_once_with("test.txt", encoding="utf-8")

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("utils.file_loader.PyPDFLoader")
    def test_load_pdf_file(self, mock_pdf_loader):
        """Тест загрузки PDF файла"""
        mock_loader_instance = Mock()
        mock_pdf_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = [
            Document(page_content="PDF content", metadata={})
        ]

        loader = FileLoader()
        loader.text_splitter = Mock()
        loader.text_splitter.split_documents.return_value = [
            Document(page_content="PDF content", metadata={})
        ]

        result = loader.load_file("test.pdf")

        assert len(result) == 1
        assert result[0].metadata["source_file"] == "test.pdf"
        mock_pdf_loader.assert_called_once_with("test.pdf")

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    def test_unsupported_file_format(self):
        """Тест неподдерживаемого формата файла"""
        from core.exceptions import EoraRAGException

        loader = FileLoader()

        # Ожидаем, что будет поднято исключение EoraRAGException
        with pytest.raises(EoraRAGException):
            loader.load_file("test.xyz")

    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("utils.file_loader.os.walk")
    def test_load_directory(self, mock_walk):
        """Тест загрузки директории"""
        mock_walk.return_value = [
            ("/test", [], ["file1.txt", "file2.pdf", "file3.xyz"])
        ]

        loader = FileLoader()
        loader.load_file = Mock()
        loader.load_file.side_effect = [
            [Document(page_content="Content 1", metadata={"source_file": "file1.txt"})],
            [Document(page_content="Content 2", metadata={"source_file": "file2.pdf"})],
        ]

        result = loader.load_directory("/test")

        assert len(result) == 2
        assert loader.load_file.call_count == 2
