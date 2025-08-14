from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from core.config import Config


class LLMProvider(ABC):
    """Абстрактный базовый класс для провайдеров LLM"""

    @abstractmethod
    def get_llm(self):
        """Получить экземпляр LLM"""
        pass

    @abstractmethod
    def invoke(self, messages):
        """Вызвать LLM с сообщениями"""
        pass


class OpenAIProvider(LLMProvider):
    """Провайдер для OpenAI"""

    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.1)

    def get_llm(self):
        return self.llm

    def invoke(self, messages):
        return self.llm.invoke(messages)


class GigaChatProvider(LLMProvider):
    """Провайдер для GigaChat (заглушка)"""

    def __init__(self):
        self.llm = None

    def get_llm(self):
        if self.llm is None:
            raise NotImplementedError("GigaChat провайдер еще не реализован")
        return self.llm

    def invoke(self, messages):
        if self.llm is None:
            raise NotImplementedError("GigaChat провайдер еще не реализован")
        return self.llm.invoke(messages)


class LLMFactory:
    """Фабрика для создания провайдеров LLM"""

    @staticmethod
    def create_provider(provider_name: str = None) -> LLMProvider:
        """Создать провайдер LLM"""
        if provider_name is None:
            provider_name = Config.MODEL_PROVIDER

        if provider_name.lower() == "openai":
            return OpenAIProvider()
        elif provider_name.lower() == "gigachat":
            return GigaChatProvider()
        else:
            raise ValueError(f"Неподдерживаемый провайдер LLM: {provider_name}")

    @staticmethod
    def get_available_providers():
        """Получить список доступных провайдеров"""
        return ["openai", "gigachat"]
