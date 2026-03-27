import os
import asyncio
import httpx
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class EmbeddingProvider(str, Enum):
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"

class ProviderConfig:
    def __init__(self, name: str, url: str, model: str, priority: int, api_key: Optional[str] = None):
        self.name = name
        self.url = url
        self.model = model
        self.priority = priority
        self.api_key = api_key
        self.is_healthy = True
        self.consecutive_failures = 0
        self.last_failure_time = None

class EmbedServiceManager:
    def __init__(self):
        self.providers: List[ProviderConfig] = []
        self.client = httpx.AsyncClient(timeout=30.0)
        self._init_providers()
        self.health_check_interval = 60
        self._health_check_task = None

    def _init_providers(self):
        """Initialize embedding providers based on environment variables"""

        if os.getenv("OLLAMA_ENABLED", "true").lower() == "true":
            ollama_provider = ProviderConfig(
                name="ollama",
                url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
                priority=int(os.getenv("OLLAMA_PRIORITY", "10")),
                api_key=None
            )
            self.providers.append(ollama_provider)

        if os.getenv("MISTRAL_ENABLED", "false").lower() == "true":
            mistral_provider = ProviderConfig(
                name="mistral",
                url="https://api.mistral.ai/v1/embeddings",
                model=os.getenv("MISTRAL_MODEL", "mistral-embed"),
                priority=int(os.getenv("MISTRAL_PRIORITY", "5")),
                api_key=os.getenv("MISTRAL_API_KEY")
            )
            self.providers.append(mistral_provider)

        if os.getenv("DEEPSEEK_ENABLED", "false").lower() == "true":
            deepseek_provider = ProviderConfig(
                name="deepseek",
                url="https://api.deepseek.com/v1/embeddings",
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-coder"),
                priority=int(os.getenv("DEEPSEEK_PRIORITY", "3")),
                api_key=os.getenv("DEEPSEEK_API_KEY")
            )
            self.providers.append(deepseek_provider)

        self.providers.sort(key=lambda p: p.priority, reverse=True)
        logger.info(f"Initialized providers: {[p.name for p in self.providers]}")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding with automatic fallback"""

        for provider in self.providers:
            if provider.consecutive_failures >= 5:
                if datetime.now() - provider.last_failure_time < timedelta(minutes=5):
                    logger.warning(f"Skipping {provider.name} (too many failures)")
                    continue

            try:
                embedding = await self._embed_with_provider(provider, text)
                provider.consecutive_failures = 0
                provider.is_healthy = True
                logger.debug(f"Successfully embedded with {provider.name}")
                return embedding
            except Exception as e:
                provider.consecutive_failures += 1
                provider.last_failure_time = datetime.now()
                logger.warning(f"Failed to embed with {provider.name}: {str(e)}")
                continue

        raise Exception("All embedding providers failed")

    async def _embed_with_provider(self, provider: ProviderConfig, text: str) -> List[float]:
        """Embed text using specific provider"""

        if provider.name == "ollama":
            return await self._embed_ollama(provider, text)
        elif provider.name == "mistral":
            return await self._embed_mistral(provider, text)
        elif provider.name == "deepseek":
            return await self._embed_deepseek(provider, text)

    async def _embed_ollama(self, provider: ProviderConfig, text: str) -> List[float]:
        """Embed using Ollama"""
        url = f"{provider.url}/api/embed"
        payload = {
            "model": provider.model,
            "input": text
        }
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]

    async def _embed_mistral(self, provider: ProviderConfig, text: str) -> List[float]:
        """Embed using Mistral API"""
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": provider.model,
            "input": [text]
        }
        response = await self.client.post(provider.url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def _embed_deepseek(self, provider: ProviderConfig, text: str) -> List[float]:
        """Embed using DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": provider.model,
            "input": text
        }
        response = await self.client.post(provider.url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def health_check(self) -> dict:
        """Check health of all providers"""
        health_status = {}

        for provider in self.providers:
            try:
                if provider.name == "ollama":
                    response = await self.client.get(f"{provider.url}/api/tags", timeout=5.0)
                    provider.is_healthy = response.status_code == 200
                else:
                    provider.is_healthy = provider.consecutive_failures < 3

                health_status[provider.name] = {
                    "enabled": True,
                    "is_healthy": provider.is_healthy,
                    "consecutive_failures": provider.consecutive_failures,
                    "priority": provider.priority
                }
            except Exception as e:
                provider.is_healthy = False
                health_status[provider.name] = {
                    "enabled": True,
                    "is_healthy": False,
                    "error": str(e),
                    "priority": provider.priority
                }

        return health_status

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
