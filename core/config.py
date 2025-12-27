from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Centralized application settings using Pydantic.
    Handles environment variables with automatic .env loading.
    """
    # API Keys
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # LLM Configuration
    default_llm_provider: str = "gemini"
    gemini_model: str = "gemini-1.5-flash"
    claude_model: str = "claude-3-5-sonnet-20241022"
    
    # Infrastructure
    backend_url: str = "http://localhost:8000"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    
    # App Settings
    app_name: str = "AgentForge"
    version: str = "1.0.0"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global settings instance
settings = Settings()
