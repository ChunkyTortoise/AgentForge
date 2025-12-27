"""
Unified LLM Client - Multi-provider support for Gemini and Claude.

Provides a consistent interface for interacting with different LLM providers,
including direct SDK access and LangChain compatibility.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, Optional, Union, AsyncGenerator

from core.config import settings
from utils.logger import get_logger

# Import LangChain models only when needed
from langchain_core.language_models.chat_models import BaseChatModel

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
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider ("gemini" or "claude"). Defaults to settings.
            model: Specific model name. Defaults to settings.
        """
        provider_val = provider or settings.default_llm_provider
        self.provider = LLMProvider(provider_val)

        # Set default models
        if model:
            self.model = model
        elif self.provider == LLMProvider.GEMINI:
            self.model = settings.gemini_model
        else:
            self.model = settings.claude_model

        # Initialize provider client (Lazy initialization)
        self._client = None
        self._async_client = None

    def _init_client(self) -> None:
        """Initialize the provider-specific client."""
        if self._client is not None:
            return

        if self.provider == LLMProvider.GEMINI:
            self._init_gemini()
        else:
            self._init_claude()

    def _init_async_client(self) -> None:
        """Initialize the provider-specific async client."""
        if self._async_client is not None:
            return
            
        if self.provider == LLMProvider.CLAUDE:
            from anthropic import AsyncAnthropic
            self._async_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        elif self.provider == LLMProvider.GEMINI:
            # Gemini's main client handles async via generate_content_async
            self._init_gemini()
            self._async_client = self._client

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client."""
        api_key = settings.google_api_key
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
        api_key = settings.anthropic_api_key
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
        self._init_client()
        return self._client is not None

    def get_langchain_model(self, **kwargs) -> BaseChatModel:
        """
        Returns a LangChain-compatible chat model instance.
        """
        temperature = kwargs.get("temperature", 0.0)
        
        if self.provider == LLMProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not settings.google_api_key:
                 raise ValueError("GOOGLE_API_KEY not found in settings")
            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=settings.google_api_key,
                temperature=temperature,
                convert_system_message_to_human=True,
                **{k: v for k, v in kwargs.items() if k != "temperature"}
            )
        elif self.provider == LLMProvider.CLAUDE:
            from langchain_anthropic import ChatAnthropic
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in settings")
            return ChatAnthropic(
                model=self.model,
                anthropic_api_key=settings.anthropic_api_key,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k != "temperature"}
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM (Synchronous)."""
        if not self.is_available():
            raise RuntimeError(f"{self.provider.value} client not initialized")

        if self.provider == LLMProvider.GEMINI:
            return self._generate_gemini(prompt, system_prompt, max_tokens, temperature)
        else:
            return self._generate_claude(prompt, system_prompt, max_tokens, temperature)

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM (Asynchronous)."""
        self._init_async_client()
        if not self._async_client:
            raise RuntimeError(f"{self.provider.value} async client not initialized")

        if self.provider == LLMProvider.GEMINI:
            response = await self._async_client.generate_content_async(
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
            )
            return LLMResponse(
                content=response.text,
                provider=self.provider,
                model=self.model,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None
            )
        else:
            response = await self._async_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return LLMResponse(
                content=response.content[0].text,
                provider=self.provider,
                model=self.model,
                tokens_used=response.usage.output_tokens if hasattr(response, 'usage') else None,
                finish_reason=response.stop_reason
            )

    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the LLM (Asynchronous).
        """
        self._init_async_client()
        if not self._async_client:
            raise RuntimeError(f"{self.provider.value} async client not initialized")
            
        if self.provider == LLMProvider.GEMINI:
             full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
             response = await self._async_client.generate_content_async(full_prompt, stream=True)
             async for chunk in response:
                 if chunk.text:
                     yield chunk.text
        else:
            async with self._async_client.messages.stream(
                model=self.model,
                max_tokens=2048,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text

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
        """Stream response chunks from the LLM."""
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
    """Check which LLM providers are available."""
    providers = {}
    providers["gemini"] = bool(settings.google_api_key)
    providers["claude"] = bool(settings.anthropic_api_key and settings.anthropic_api_key.startswith("sk-ant-"))
    return providers