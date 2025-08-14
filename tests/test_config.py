import pytest
import os
from unittest.mock import patch
from core.config import Config
from core.exceptions import ConfigurationError


class TestConfig:
    """Тесты для конфигурации"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_validate_success(self):
        """Тест успешной валидации"""
        assert Config.validate() is True

    @patch("core.config.Config.OPENAI_API_KEY", None)
    @patch("core.config.Config.MODEL_PROVIDER", "openai")
    def test_validate_missing_openai_key(self):
        """Тест валидации без OpenAI ключа"""
        with pytest.raises(ConfigurationError) as exc_info:
            Config.validate()
        assert "OPENAI_API_KEY не установлен" in str(exc_info.value)

    @patch.dict(
        os.environ, {"GIGACHAT_API_KEY": "test-key", "MODEL_PROVIDER": "gigachat"}
    )
    def test_validate_gigachat_success(self):
        """Тест успешной валидации с GigaChat"""
        assert Config.validate() is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "CHUNK_SIZE": "0"})
    def test_validate_invalid_chunk_size(self):
        """Тест валидации с неверным размером чанка"""
        # Сбрасываем кэшированные значения конфигурации
        from importlib import reload
        import core.config

        reload(core.config)
        from core.config import Config

        with pytest.raises(ConfigurationError) as exc_info:
            Config.validate()
        assert "CHUNK_SIZE должен быть положительным числом" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "CHUNK_OVERLAP": "-1"})
    def test_validate_negative_chunk_overlap(self):
        """Тест валидации с отрицательным перекрытием чанков"""
        # Сбрасываем кэшированные значения конфигурации
        from importlib import reload
        import core.config

        reload(core.config)
        from core.config import Config

        with pytest.raises(ConfigurationError) as exc_info:
            Config.validate()
        assert "CHUNK_OVERLAP не может быть отрицательным" in str(exc_info.value)

    def test_default_values(self):
        """Тест значений по умолчанию"""
        assert Config.MODEL_PROVIDER == "openai"
        assert Config.MODEL_NAME == "gpt-3.5-turbo"
        assert Config.DATA_PATH == "./data"
        assert Config.CHUNK_SIZE == 1000
        assert Config.CHUNK_OVERLAP == 200
        assert Config.CRAWL_MAX_PAGES == 20
        assert Config.SEARCH_K == 5

    @patch("core.config.logging.basicConfig")
    def test_setup_logging(self, mock_basicConfig):
        """Тест настройки логирования"""
        Config.setup_logging()
        mock_basicConfig.assert_called_once()
