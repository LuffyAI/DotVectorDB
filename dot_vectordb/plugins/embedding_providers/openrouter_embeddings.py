"""
Author: Larnell Moore
Date: 2026-07-03
Description:
Wrapper for the OpenRouter embedding provider. This calls the OpenRouter's API's embedding endpoint to generate embeddings for text. It returns a numpy array of floats.
"""
import aiohttp
import numpy as np
from dotenv import load_dotenv
import os
from collections.abc import Awaitable, Callable
from dot_vectordb.interfaces.EmbeddingModel import EmbeddingModel, Embedding
load_dotenv()

class OpenRouterEmbedding(EmbeddingModel):
    def __init__(
        self,
        model_name: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "baai/bge-large-en-v1.5"),
        api_key: str = os.getenv("OPENROUTER_API_KEY"),
        embedding_dim: int = os.getenv("OPENROUTER_EMBEDDING_DIM", 1536),
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            embedding_dim=embedding_dim,
        )

    async def _get_session(self):
        """Opens an aiohttp session if one does not exist or is closed, and returns it."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _call_embedding_model(self, text: str) -> Embedding:
        """Generate an embedding for the given text using the OpenRouter API. Returns a numpy array of floats."""
        session = await self._get_session()
        async with session.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": text,
            },
        ) as res:
            data = await res.json()
            embedding = data["data"][0]["embedding"]
            arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
            return arr

    async def embed(self, text: str) -> Embedding:
        return await super().embed(text)