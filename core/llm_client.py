"""
Unified LLM Client - Multi-provider support for Gemini and Claude.

Provides a consistent interface for interacting with different LLM providers.
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, Optional

from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    CLAUDE = "claude"


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Provides a consistent interface for Gemini and Claude APIs.

    Example:
        client = LLMClient()
        response = client.generate("Explain RAG in simple terms")
        print(response.content)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider ("gemini" or "claude"). Defaults to env var.
            model: Specific model name. Defaults to env var.
        """
        self.provider = LLMProvider(
            provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
        )

        # Set default models
        if model:
            self.model = model
        elif self.provider == LLMProvider.GEMINI:
            self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        else:
            self.model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

        # Initialize provider client
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the provider-specific client."""
        if self.provider == LLMProvider.GEMINI:
            self._init_gemini()
        else:
            self._init_claude()

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Gemini client initialized with model: {self.model}")
        except ImportError:
            logger.error("google-generativeai not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    def _init_claude(self) -> None:
        """Initialize Anthropic Claude client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set")
            return

        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
            logger.info(f"Claude client initialized with model: {self.model}")
        except ImportError:
            logger.error("anthropic not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")

    def is_available(self) -> bool:
        """Check if the client is properly initialized."""
        return self._client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError(f"{self.provider.value} client not initialized")

        if self.provider == LLMProvider.GEMINI:
            return self._generate_gemini(prompt, system_prompt, max_tokens, temperature)
        else:
            return self._generate_claude(prompt, system_prompt, max_tokens, temperature)

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Generate response using Gemini."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = self._client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )

        return LLMResponse(
            content=response.text,
            provider=self.provider,
            model=self.model,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None
        )

    def _generate_claude(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Generate response using Claude."""
        messages = [{"role": "user", "content": prompt}]

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful AI assistant.",
            messages=messages
        )

        return LLMResponse(
            content=response.content[0].text,
            provider=self.provider,
            model=self.model,
            tokens_used=response.usage.output_tokens if hasattr(response, 'usage') else None,
            finish_reason=response.stop_reason
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response chunks from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions

        Yields:
            Text chunks as they're generated
        """
        if not self.is_available():
            raise RuntimeError(f"{self.provider.value} client not initialized")

        if self.provider == LLMProvider.GEMINI:
            yield from self._stream_gemini(prompt, system_prompt)
        else:
            yield from self._stream_claude(prompt, system_prompt)

    def _stream_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str]
    ) -> Generator[str, None, None]:
        """Stream response from Gemini."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self._client.generate_content(full_prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _stream_claude(
        self,
        prompt: str,
        system_prompt: Optional[str]
    ) -> Generator[str, None, None]:
        """Stream response from Claude."""
        with self._client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=system_prompt or "You are a helpful AI assistant.",
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text


def get_available_providers() -> Dict[str, bool]:
    """
    Check which LLM providers are available.

    Returns:
        Dict mapping provider name to availability status
    """
    providers = {}

    # Check Gemini
    gemini_key = os.getenv("GOOGLE_API_KEY")
    providers["gemini"] = bool(gemini_key and gemini_key != "your_gemini_api_key_here")

    # Check Claude
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    providers["claude"] = bool(claude_key and claude_key.startswith("sk-ant-"))

    return providers
