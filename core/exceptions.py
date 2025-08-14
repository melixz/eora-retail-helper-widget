class EoraRAGException(Exception):
    """Базовое исключение для EORA RAG системы"""

    pass


class ConfigurationError(EoraRAGException):
    """Ошибка конфигурации"""

    pass


class DocumentLoadError(EoraRAGException):
    """Ошибка загрузки документов"""

    pass


class VectorStoreError(EoraRAGException):
    """Ошибка векторного хранилища"""

    pass


class LLMError(EoraRAGException):
    """Ошибка LLM"""

    pass


class WebCrawlerError(EoraRAGException):
    """Ошибка веб-краулера"""

    pass
