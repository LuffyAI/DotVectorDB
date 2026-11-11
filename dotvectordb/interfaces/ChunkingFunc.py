from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import tiktoken
import traceback

class ChunkingFunc(ABC):
    """Abstract base class for chunking functions.
    
    Ensures that implementations return a list of strings.
    """
    def __init__(self, Tokenizer: str = os.getenv("OPENAI_TOKENIZER", "gpt-4o")):
        self._tokenizer = tiktoken.encoding_for_model(Tokenizer)

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into a list of dictionaries.
        
        Args:
            text: The input text to chunk.
            
        Returns:
            A list of string chunks.
        """
        pass

    def tokenize(self, text: str) -> list[int]:
        """Tokenizes the input text and returns a list of token IDs."""
        return self._tokenizer.encode(text)

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Make the chunking function callable.
        
        Args:
            text: The input text to chunk.
            
        Returns:
            A list of string chunks.
        """
        try:
            result = self.chunk(text)
            
            # Ensure result is a list of dictionaries
            if not isinstance(result, list):
                raise TypeError(f"chunk() must return a list, got {type(result)}")
            
            if not all(isinstance(chunk, dict) for chunk in result):
                raise TypeError("chunk() must return a list of dictionaries")
            
            return result
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the Chunking Function's interface in ChunkingFunc.py: {e}")
