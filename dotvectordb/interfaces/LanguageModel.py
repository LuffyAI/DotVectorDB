from typing import Any, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dotvectordb.interfaces.EmbeddingModel import EmbeddingModel
from pydantic import BaseModel, Field
import tiktoken
import traceback

class FinalResponse(BaseModel):
    answer: str = Field(description="The final answer to the user's query based on the knowledge graph and vector data.")

class LanguageModel(ABC):
    def __init__(self, model_name: str, api_key: str, max_tokens: int, Tokenizer: str):
        self.model_name = model_name
        self.api_key = api_key
        self._tokenizer = tiktoken.encoding_for_model(Tokenizer)
        self._max_tokens = max_tokens

    def tokenize(self, text: str) -> list[int]:
        """Tokenizes the input text and returns a list of token IDs."""
        return self._tokenizer.encode(text)
    
    @abstractmethod
    async def call_llm(self,prompt, response_format=None, verbose=False) -> Dict[Any, Any]:
        """Abstract method to call the language model. Must be implemented by subclasses."""
        pass
    
    async def generate(self, prompt: str, structuredInput=None) -> Dict[Any, Any]:
        """Return response, total, input, and output token counts."""
        try:
            if structuredInput is not None:
                response, total_tokens, input_tokens, output_tokens = (
                    await self.call_llm(prompt, structuredInput)
                )
            else:
                response, total_tokens, input_tokens, output_tokens = (
                    await self.call_llm(prompt)
                )
            return response, total_tokens, input_tokens, output_tokens
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the Language Model's interface in LanguageModel.py: {e}")
