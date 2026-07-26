from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
import numpy as np
from numpy.typing import NDArray
import tiktoken
import traceback


# All Embeddings will be represented as a 1D numpy array of floats.
Embedding = NDArray[np.floating]

class EmbeddingModel(ABC):
    def __init__(self, model_name: str, api_key: str, embedding_dim: int, Tokenizer: str):
        self.model_name = model_name
        self.api_key = api_key
        self._tokenizer = tiktoken.encoding_for_model(Tokenizer)
        self.embedding_dim = embedding_dim

    def tokenize(self, text: str) -> list[int]:
        """Tokenizes the input text and returns a list of token IDs."""
        return self._tokenizer.encode(text)
    
    @abstractmethod
    async def _call_embedding_model(self, text: str) -> Embedding:
        """Abstract method to call the embedding model. Must be implemented by subclasses."""
        pass  

    async def embed(self, text: str) -> Embedding:
        """Given a text string, return its embedding as a 1D numpy array of floats."""
        try:
            embedding, tokens = await self._call_embedding_model(text)

            if not isinstance(embedding, np.ndarray):
                raise TypeError(
                    "call_embedding_model must return a numpy.ndarray."
                )

            if not np.issubdtype(embedding.dtype, np.floating):
                raise TypeError(
                    "Embedding numpy array must contain floats."
                )
            return embedding, len(tokens)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the Embedding Model's interface in EmbeddingModel.py: {e}")