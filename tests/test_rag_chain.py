from unittest.mock import Mock, patch
from core.rag_chain import EoraRAGChain
from langchain_core.documents import Document


class TestEoraRAGChain:
    """Тесты для RAG цепочки"""

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_init(self, mock_llm_factory, mock_embeddings):
        """Тест инициализации"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

        chain = EoraRAGChain()
        assert chain.vectorstore is None
        assert chain.llm_provider is not None

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_search_relevant_docs_empty_vectorstore(
        self, mock_llm_factory, mock_embeddings
    ):
        """Тест поиска при пустом векторном хранилище"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

        chain = EoraRAGChain()
        result = chain.search_relevant_docs("test query")
        assert result == []

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_generate_answer_no_docs(self, mock_llm_factory, mock_embeddings):
        """Тест генерации ответа без документов"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

        chain = EoraRAGChain()
        result = chain.generate_answer("test query")

        assert "не нашел релевантной информации" in result["answer"]
        assert result["sources"] == []
        assert result["complexity_level"] == "easy"

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_prepare_context_with_references(self, mock_llm_factory, mock_embeddings):
        """Тест подготовки контекста с ссылками"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

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

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    @patch("core.rag_chain.FAISS")
    @patch("os.path.exists")
    def test_load_documents_with_files(
        self, mock_exists, mock_faiss, mock_llm_factory, mock_embeddings
    ):
        """Тест загрузки документов из файлов"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

        # Мокаем что путь существует
        mock_exists.return_value = True

        chain = EoraRAGChain()
        chain.file_loader = Mock()
        chain.file_loader.load_directory.return_value = [
            Document(page_content="Test", metadata={"source_file": "test.txt"})
        ]

        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        result = chain.load_documents("./test_data", include_web=False)

        # Проверяем, что метод был вызван
        chain.file_loader.load_directory.assert_called_once_with("./test_data")
        assert result == 1  # Один документ загружен

    @patch("core.config.Config.OPENAI_API_KEY", "test-key")
    @patch("core.config.Config.CHUNK_SIZE", 1000)
    @patch("core.config.Config.CHUNK_OVERLAP", 200)
    @patch("core.rag_chain.OpenAIEmbeddings")
    @patch("core.rag_chain.LLMFactory")
    def test_complexity_levels(self, mock_llm_factory, mock_embeddings):
        """Тест различных уровней сложности промптов"""
        mock_provider = Mock()
        mock_llm = Mock()
        mock_provider.get_llm.return_value = mock_llm
        mock_llm_factory.create_provider.return_value = mock_provider

        chain = EoraRAGChain()

        easy_prompt = chain._get_easy_prompt()
        medium_prompt = chain._get_medium_prompt()
        hard_prompt = chain._get_hard_prompt()

        # Проверяем, что промпты созданы и являются ChatPromptTemplate
        assert easy_prompt is not None
        assert medium_prompt is not None
        assert hard_prompt is not None

        # Проверяем содержимое через format_messages или str()
        easy_str = str(easy_prompt)
        medium_str = str(medium_prompt)
        hard_str = str(hard_prompt)

        assert "Используя только предоставленную информацию" in easy_str
        assert "Источники:" in medium_str
        assert "[1], [2]" in hard_str
