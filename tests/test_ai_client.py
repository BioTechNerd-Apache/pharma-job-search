"""Tests for the multi-provider AI client abstraction."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from src.ai_client import (
    AIResponse,
    DEFAULT_MODELS,
    MODEL_PRICING,
    get_model_pricing,
)
from src.config import EvaluationConfig


class TestAIResponse:
    def test_defaults(self):
        r = AIResponse(text="hello")
        assert r.text == "hello"
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_with_tokens(self):
        r = AIResponse(text="result", input_tokens=100, output_tokens=50)
        assert r.input_tokens == 100
        assert r.output_tokens == 50


class TestAIClientCreation:
    def test_anthropic_client(self):
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from src.ai_client import AIClient
            client = AIClient(provider="anthropic", api_key="sk-test")
            assert client.provider == "anthropic"
            assert client.model == DEFAULT_MODELS["anthropic"]
            mock_anthropic.Anthropic.assert_called_once()

    def test_openai_client(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import AIClient
            client = AIClient(provider="openai", api_key="sk-test")
            assert client.provider == "openai"
            assert client.model == DEFAULT_MODELS["openai"]
            mock_openai.OpenAI.assert_called_once()

    def test_ollama_client(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import AIClient
            client = AIClient(provider="ollama")
            assert client.provider == "ollama"
            assert client.model == DEFAULT_MODELS["ollama"]
            call_kwargs = mock_openai.OpenAI.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:11434/v1"
            assert call_kwargs["api_key"] == "ollama"

    def test_ollama_custom_base_url(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import AIClient
            client = AIClient(provider="ollama", base_url="http://myserver:11434/v1")
            call_kwargs = mock_openai.OpenAI.call_args[1]
            assert call_kwargs["base_url"] == "http://myserver:11434/v1"

    def test_unsupported_provider(self):
        from src.ai_client import AIClient
        with pytest.raises(ValueError, match="Unsupported AI provider"):
            AIClient(provider="gemini", api_key="test")

    def test_custom_model(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import AIClient
            client = AIClient(provider="openai", model="gpt-4o", api_key="sk-test")
            assert client.model == "gpt-4o"

    def test_provider_case_insensitive(self):
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from src.ai_client import AIClient
            client = AIClient(provider="Anthropic", api_key="sk-test")
            assert client.provider == "anthropic"


class TestModelPricing:
    def test_known_anthropic_model(self):
        pricing = get_model_pricing("anthropic", "claude-haiku-4-5-20251001")
        assert pricing == (0.80, 4.00)

    def test_known_openai_model(self):
        pricing = get_model_pricing("openai", "gpt-4o-mini")
        assert pricing == (0.15, 0.60)

    def test_ollama_always_none(self):
        pricing = get_model_pricing("ollama", "llama3.1:8b")
        assert pricing is None

    def test_unknown_model(self):
        pricing = get_model_pricing("openai", "some-future-model")
        assert pricing is None


class TestErrorDetection:
    def test_anthropic_rate_limit(self):
        mock_anthropic = MagicMock()
        mock_exc = MagicMock()
        mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        exc = mock_anthropic.RateLimitError()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from src.ai_client import AIClient
            client = AIClient(provider="anthropic", api_key="sk-test")
            assert client.is_rate_limit_error(exc) is True

    def test_openai_rate_limit(self):
        mock_openai = MagicMock()
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        exc = mock_openai.RateLimitError()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import AIClient
            client = AIClient(provider="openai", api_key="sk-test")
            assert client.is_rate_limit_error(exc) is True

    def test_not_rate_limit(self):
        mock_anthropic = MagicMock()
        mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from src.ai_client import AIClient
            client = AIClient(provider="anthropic", api_key="sk-test")
            assert client.is_rate_limit_error(ValueError("not rate limit")) is False

    def test_anthropic_overloaded(self):
        mock_anthropic = MagicMock()
        mock_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})
        exc = mock_anthropic.APIStatusError()
        exc.status_code = 529
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from src.ai_client import AIClient
            client = AIClient(provider="anthropic", api_key="sk-test")
            assert client.is_overloaded_error(exc) is True

    def test_openai_overloaded_503(self):
        mock_openai = MagicMock()
        mock_openai.APIStatusError = type("APIStatusError", (Exception,), {})
        exc = mock_openai.APIStatusError()
        exc.status_code = 503
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import AIClient
            client = AIClient(provider="openai", api_key="sk-test")
            assert client.is_overloaded_error(exc) is True


class TestCreateAIClient:
    def test_from_default_config(self):
        """Backward compat: config with only anthropic_api_key works."""
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from src.ai_client import create_ai_client
            config = EvaluationConfig(anthropic_api_key="sk-ant-test")
            client = create_ai_client(config)
            assert client.provider == "anthropic"
            assert client.api_key == "sk-ant-test"

    def test_from_openai_config(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import create_ai_client
            config = EvaluationConfig(provider="openai", api_key="sk-openai-test",
                                      model="gpt-4o-mini")
            client = create_ai_client(config)
            assert client.provider == "openai"
            assert client.model == "gpt-4o-mini"

    def test_from_ollama_config(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.ai_client import create_ai_client
            config = EvaluationConfig(provider="ollama", model="llama3.1:8b")
            client = create_ai_client(config)
            assert client.provider == "ollama"
            assert client.model == "llama3.1:8b"


class TestBackwardCompatConfig:
    def test_default_provider(self):
        config = EvaluationConfig()
        assert config.provider == "anthropic"

    def test_legacy_api_key_resolution(self):
        config = EvaluationConfig(anthropic_api_key="sk-legacy")
        assert config.get_api_key() == "sk-legacy"

    def test_new_api_key_preferred(self):
        config = EvaluationConfig(api_key="sk-new", anthropic_api_key="sk-legacy")
        assert config.get_api_key() == "sk-new"

    def test_ollama_enabled_without_key(self):
        config = EvaluationConfig(provider="ollama")
        assert config.enabled is True

    def test_anthropic_disabled_without_key(self):
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            config = EvaluationConfig(provider="anthropic")
            assert config.enabled is False
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_openai_env_var_resolution(self):
        import os
        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-env-openai"
        try:
            config = EvaluationConfig(provider="openai")
            assert config.get_api_key() == "sk-env-openai"
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old
            else:
                os.environ.pop("OPENAI_API_KEY", None)
