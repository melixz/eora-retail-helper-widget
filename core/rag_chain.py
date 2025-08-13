import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from utils.file_loader import FileLoader
from utils.web_crawler import WebCrawler
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

    def load_documents(self, data_path: str, include_web: bool = True):
        """Загрузка и индексация документов"""
        all_documents = []

        if os.path.exists(data_path):
            file_chunks = self.file_loader.load_directory(data_path)
            all_documents.extend(file_chunks)

        if include_web:
            web_data = self.web_crawler.crawl_site(max_pages=Config.CRAWL_MAX_PAGES)
            for page in web_data:
                if page and page.get("content"):
                    doc = Document(
                        page_content=page["content"], metadata=page["metadata"]
                    )
                    all_documents.append(doc)

        if all_documents:
            self.vectorstore = FAISS.from_documents(all_documents, self.embeddings)
            self.documents = all_documents
            return len(all_documents)
        return 0

    def search_relevant_docs(self, query: str, k: int = None) -> List[Document]:
        """Поиск релевантных документов"""
        if not self.vectorstore:
            return []

        k = k or Config.SEARCH_K
        return self.vectorstore.similarity_search(query, k=k)

    def generate_answer(
        self, query: str, complexity_level: str = "easy"
    ) -> Dict[str, Any]:
        """Генерация ответа с учетом уровня сложности"""
        relevant_docs = self.search_relevant_docs(query)

        if not relevant_docs:
            return {
                "answer": "Извините, я не нашел релевантной информации для ответа на ваш вопрос.",
                "sources": [],
                "complexity_level": complexity_level,
            }

        sources = [doc.metadata for doc in relevant_docs]

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
