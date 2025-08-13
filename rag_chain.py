from typing import List, Dict, Any


class EoraRAGChain:
    """Основной класс для RAG pipeline"""

    def __init__(self):
        self.documents = []
        self.vectorstore = None
        self.llm = None

    def load_documents(self, data_path: str):
        """Загрузка и индексация документов"""
        pass

    def search_relevant_docs(self, query: str) -> List[Dict[str, Any]]:
        """Поиск релевантных документов"""
        pass

    def generate_answer(
        self, query: str, complexity_level: str = "easy"
    ) -> Dict[str, Any]:
        """Генерация ответа с учетом уровня сложности"""
        pass
