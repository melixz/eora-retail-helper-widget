import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from utils.file_loader import FileLoader
from utils.web_crawler import WebCrawler

load_dotenv()


class EoraRAGChain:
    """Основной класс для RAG pipeline"""

    def __init__(self):
        self.documents = []
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"), temperature=0.1
        )
        self.file_loader = FileLoader()
        self.web_crawler = WebCrawler()

    def load_documents(self, data_path: str, include_web: bool = True):
        """Загрузка и индексация документов"""
        all_documents = []

        if os.path.exists(data_path):
            file_chunks = self.file_loader.load_directory(data_path)
            all_documents.extend(file_chunks)

        if include_web:
            web_data = self.web_crawler.crawl_site(max_pages=20)
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

    def search_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """Поиск релевантных документов"""
        if not self.vectorstore:
            return []

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

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = [doc.metadata for doc in relevant_docs]

        if complexity_level == "easy":
            prompt = self._get_easy_prompt()
        elif complexity_level == "medium":
            prompt = self._get_medium_prompt()
        else:
            prompt = self._get_hard_prompt()

        response = self.llm.invoke(prompt.format(context=context, question=query))

        return {
            "answer": response.content,
            "sources": sources,
            "complexity_level": complexity_level,
        }

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
