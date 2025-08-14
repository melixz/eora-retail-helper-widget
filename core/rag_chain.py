import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from utils.file_loader import FileLoader
from utils.web_crawler import WebCrawler
from utils.error_handler import (
    ErrorHandler,
    handle_document_load_errors,
    handle_llm_errors,
)
from core.config import Config


class EoraRAGChain:
    """Основной класс для RAG pipeline"""

    def __init__(self):
        Config.validate()
        self.documents = []
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.1)
        self.file_loader = FileLoader()
        self.web_crawler = WebCrawler(delay=Config.CRAWL_DELAY)

    @handle_document_load_errors
    def load_documents(self, data_path: str, include_web: bool = True):
        """Загрузка и индексация документов"""
        all_documents = []

        if os.path.exists(data_path):
            ErrorHandler.log_info(f"Загрузка документов из {data_path}")
            file_chunks = self.file_loader.load_directory(data_path)
            all_documents.extend(file_chunks)
            ErrorHandler.log_info(f"Загружено {len(file_chunks)} чанков из файлов")

        if include_web:
            ErrorHandler.log_info("Начинаем парсинг веб-страниц")
            web_data = ErrorHandler.safe_execute(
                lambda: self.web_crawler.crawl_site(max_pages=Config.CRAWL_MAX_PAGES),
                default_return=[],
                context="парсинг веб-страниц",
            )
            for page in web_data:
                if page and page.get("content"):
                    doc = Document(
                        page_content=page["content"], metadata=page["metadata"]
                    )
                    all_documents.append(doc)
            ErrorHandler.log_info(f"Загружено {len(web_data)} веб-страниц")

        if all_documents:
            ErrorHandler.log_info(
                f"Создание векторного хранилища из {len(all_documents)} документов"
            )
            self.vectorstore = FAISS.from_documents(all_documents, self.embeddings)
            self.documents = all_documents
            return len(all_documents)
        else:
            ErrorHandler.log_warning("Документы не найдены")
            return 0

    def search_relevant_docs(self, query: str, k: int = None) -> List[Document]:
        """Поиск релевантных документов"""
        if not self.vectorstore:
            return []

        k = k or Config.SEARCH_K
        return self.vectorstore.similarity_search(query, k=k)

    @handle_llm_errors
    def generate_answer(
        self, query: str, complexity_level: str = "easy"
    ) -> Dict[str, Any]:
        """Генерация ответа с учетом уровня сложности"""
        if not query.strip():
            raise ValueError("Пустой запрос")

        ErrorHandler.log_info(f"Генерация ответа для запроса: {query[:50]}...")
        relevant_docs = self.search_relevant_docs(query)

        if not relevant_docs:
            ErrorHandler.log_warning("Релевантные документы не найдены")
            return {
                "answer": "Извините, я не нашел релевантной информации для ответа на ваш вопрос.",
                "sources": [],
                "complexity_level": complexity_level,
            }

        sources = [doc.metadata for doc in relevant_docs]
        ErrorHandler.log_info(f"Найдено {len(relevant_docs)} релевантных документов")

        if complexity_level == "easy":
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = self._get_easy_prompt()
        elif complexity_level == "medium":
            context = self._prepare_context_with_references(relevant_docs)
            prompt = self._get_medium_prompt()
        else:
            context = self._prepare_context_with_references(relevant_docs)
            prompt = self._get_hard_prompt()

        messages = prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)

        ErrorHandler.log_info("Ответ успешно сгенерирован")
        return {
            "answer": response.content,
            "sources": sources,
            "complexity_level": complexity_level,
        }

    def _prepare_context_with_references(self, docs: List[Document]) -> str:
        """Подготовка контекста с пронумерованными ссылками"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source_name = doc.metadata.get(
                "source_file", doc.metadata.get("url", f"Источник {i}")
            )
            context_parts.append(f"[{i}] ({source_name}):\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def _get_easy_prompt(self) -> ChatPromptTemplate:
        """Промпт для простого ответа"""
        return ChatPromptTemplate.from_template(
            "Используя только предоставленную информацию, ответьте на вопрос.\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос: {question}\n\n"
            "Ответ:"
        )

    def _get_medium_prompt(self) -> ChatPromptTemplate:
        """Промпт для ответа со списком источников"""
        return ChatPromptTemplate.from_template(
            "Используя только предоставленную информацию, ответьте на вопрос. "
            "В конце ответа укажите 'Источники: [1], [2], ...' для использованных материалов.\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос: {question}\n\n"
            "Ответ:"
        )

    def _get_hard_prompt(self) -> ChatPromptTemplate:
        """Промпт для ответа с inline ссылками"""
        return ChatPromptTemplate.from_template(
            "Используя только предоставленную информацию, ответьте на вопрос. "
            "Добавляйте ссылки на источники прямо в текст в формате [1], [2] и т.д.\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос: {question}\n\n"
            "Ответ:"
        )
