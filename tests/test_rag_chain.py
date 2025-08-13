import os
from unittest.mock import Mock, patch
from core.rag_chain import EoraRAGChain
from langchain_core.documents import Document


class TestEoraRAGChain:
    """Тесты для RAG pipeline"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.ChatOpenAI")
    def test_init(self, mock_chat, mock_embeddings):
        """Тест инициализации"""
        chain = EoraRAGChain()
        assert chain.documents == []
        assert chain.vectorstore is None
        mock_embeddings.assert_called_once()
        mock_chat.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.ChatOpenAI")
    def test_search_relevant_docs_empty_vectorstore(self, mock_chat, mock_embeddings):
        """Тест поиска при пустом векторном хранилище"""
        chain = EoraRAGChain()
        result = chain.search_relevant_docs("test query")
        assert result == []

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.ChatOpenAI")
    def test_generate_answer_no_docs(self, mock_chat, mock_embeddings):
        """Тест генерации ответа без документов"""
        chain = EoraRAGChain()
        result = chain.generate_answer("test query")

        assert "не нашел релевантной информации" in result["answer"]
        assert result["sources"] == []
        assert result["complexity_level"] == "easy"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.ChatOpenAI")
    def test_prepare_context_with_references(self, mock_chat, mock_embeddings):
        """Тест подготовки контекста с ссылками"""
        chain = EoraRAGChain()

        docs = [
            Document(
                page_content="Test content 1", metadata={"source_file": "test1.txt"}
            ),
            Document(
                page_content="Test content 2", metadata={"url": "http://test.com"}
            ),
        ]

        context = chain._prepare_context_with_references(docs)

        assert "[1] (test1.txt):" in context
        assert "[2] (http://test.com):" in context
        assert "Test content 1" in context
        assert "Test content 2" in context

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.ChatOpenAI")
    @patch("core.rag_chain.FAISS")
    def test_load_documents_with_files(self, mock_faiss, mock_chat, mock_embeddings):
        """Тест загрузки документов из файлов"""
        chain = EoraRAGChain()
        chain.file_loader = Mock()
        chain.file_loader.load_directory.return_value = [
            Document(page_content="Test", metadata={"source_file": "test.txt"})
        ]

        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        result = chain.load_documents("./test_data", include_web=False)

        assert result == 1
        assert chain.vectorstore == mock_vectorstore
        chain.file_loader.load_directory.assert_called_once_with("./test_data")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.ChatOpenAI")
    def test_complexity_levels(self, mock_chat, mock_embeddings):
        """Тест различных уровней сложности промптов"""
        chain = EoraRAGChain()

        easy_prompt = chain._get_easy_prompt()
        medium_prompt = chain._get_medium_prompt()
        hard_prompt = chain._get_hard_prompt()

        assert "Используя только предоставленную информацию" in easy_prompt.template
        assert "Источники: [1], [2]" in medium_prompt.template
        assert "inline" in hard_prompt.template or "[1], [2]" in hard_prompt.template
