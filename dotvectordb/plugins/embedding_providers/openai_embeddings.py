"""
Author: Larnell Moore
Date: 2026-07-03
Description:
Wrapper for the OpenAI embedding provider. This calls the OpenAI API's embedding endpoint to generate embeddings for text. It returns a numpy array of floats.
"""
from openai import AsyncOpenAI
import numpy as np
import os
from collections.abc import Awaitable, Callable
from dotvectordb.interfaces.EmbeddingModel import EmbeddingModel, Embedding

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key: str = os.getenv("OPENAI_API_KEY"),
        embedding_dim: int = os.getenv("OPENAI_EMBEDDING_DIM", 3072),
        Tokenizer: str = os.getenv("OPENAI_TOKENIZER", "gpt-4o"),
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            embedding_dim=embedding_dim,
            Tokenizer=Tokenizer
        )
        self._client = None

    async def __get_session(self):
        """Get the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client 
    
    async def _call_embedding_model(self, text: str) -> Embedding:
        """Generate an embedding for the given text using the OpenAI API. Returns a numpy array of floats."""
        tokens = self.tokenize(text)
        client = await self.__get_session()
        text = text.replace("\n", " ")
        response = await client.embeddings.create(input=[text], model=self.model_name)
        embedding = response.data[0].embedding
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return arr, tokens
