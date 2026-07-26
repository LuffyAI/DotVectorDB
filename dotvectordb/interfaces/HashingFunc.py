from abc import ABC, abstractmethod
from typing import List
import traceback


class HashingFunc(ABC):
    """Abstract base class for hashing functions.
    
    Ensures that implementations return a list of strings.
    """
    @abstractmethod
    def compute_hash(self, text: str) -> int:
        """Hash the input data and return an integer.
        
        Args:
            text: The text to hash.
        
        Returns:
            A signed or unsigned integer hash value.
        """
        pass

    def content_hash(self, text: str) -> int:
        """Make the hashing function callable.
        
        Args:
            text: The input text to hash.
            
        Returns:
            A signed or unsigned integer hash value.
        """
        try: 
            result = self.compute_hash(text)
            
            # Ensure result is a signed or unsigned integer
            if not isinstance(result, int):
                raise TypeError(f"hash() must return an int, got {type(result)}")
            
            return result
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the Hashing Function's interface in HashingFunc.py: {e}")