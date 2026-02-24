"""Multi-provider AI client abstraction for Anthropic, OpenAI, and Ollama."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cost per million tokens: (input, output). None = free/local/unknown.
MODEL_PRICING: dict[str, tuple[float, float] | None] = {
    # Anthropic
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5-20250514": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-nano": (0.10, 0.40),
}

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1:8b",
}

# Known models per provider (single source of truth for UI dropdowns)
PROVIDER_MODELS = {
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250514",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "ollama": [
        "llama3.1:8b",
        "llama3.1:70b",
        "gemma3:12b",
        "qwen2.5:7b",
        "mistral:7b",
    ],
}


@dataclass
class AIResponse:
    """Normalized response from any AI provider."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class AIClient:
    """Provider-agnostic AI client. Dispatches to Anthropic, OpenAI, or Ollama."""

    def __init__(
        self,
        provider: str,
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS.get(self.provider, "")
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self._client = self._create_client()

    def _create_client(self):
        if self.provider == "anthropic":
            import anthropic
            kwargs = {"max_retries": self.max_retries}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return anthropic.Anthropic(**kwargs)

        elif self.provider in ("openai", "ollama"):
            import openai
            kwargs = {"max_retries": self.max_retries}
            if self.provider == "ollama":
                kwargs["base_url"] = self.base_url or "http://localhost:11434/v1"
                kwargs["api_key"] = self.api_key or "ollama"
            else:
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
            return openai.OpenAI(**kwargs)

        else:
            raise ValueError(f"Unsupported AI provider: {self.provider!r}. "
                             f"Use 'anthropic', 'openai', or 'ollama'.")

    def create_message(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 1024,
    ) -> AIResponse:
        """Send a chat message and return a normalized AIResponse."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return AIResponse(
                text=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        else:  # openai / ollama
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user_content})

            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            choice = response.choices[0].message
            usage = response.usage
            return AIResponse(
                text=choice.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate limit error for this provider."""
        if self.provider == "anthropic":
            try:
                import anthropic
                return isinstance(exc, anthropic.RateLimitError)
            except ImportError:
                return False
        else:
            try:
                import openai
                return isinstance(exc, openai.RateLimitError)
            except ImportError:
                return False

    def is_overloaded_error(self, exc: Exception) -> bool:
        """Check if an exception is an overloaded/unavailable error."""
        if self.provider == "anthropic":
            try:
                import anthropic
                return (isinstance(exc, anthropic.APIStatusError)
                        and exc.status_code == 529)
            except ImportError:
                return False
        else:
            try:
                import openai
                return (isinstance(exc, openai.APIStatusError)
                        and exc.status_code in (502, 503))
            except ImportError:
                return False


def get_model_pricing(provider: str, model: str) -> tuple[float, float] | None:
    """Get (input_cost, output_cost) per million tokens, or None for free/unknown."""
    if provider == "ollama":
        return None
    return MODEL_PRICING.get(model)


def create_ai_client(config) -> AIClient:
    """Factory: build an AIClient from an EvaluationConfig."""
    provider = getattr(config, "provider", "anthropic")
    model = getattr(config, "model", "")
    api_key = config.get_api_key() if hasattr(config, "get_api_key") else ""
    base_url = getattr(config, "base_url", "")
    max_retries = getattr(config, "max_retries", 3)
    return AIClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
    )
