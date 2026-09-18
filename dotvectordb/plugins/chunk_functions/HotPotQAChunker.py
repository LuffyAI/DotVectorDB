import tiktoken
from dotvectordb.interfaces.ChunkingFunc import ChunkingFunc
from dotvectordb.interfaces.HashingFunc import HashingFunc
import os
import traceback

class HotPotQAChunker(ChunkingFunc):
    def __init__(self, hash_func: HashingFunc, tokenizer=os.getenv("OPENAI_TOKENIZER", "gpt-4o")):
        self._hash_func = hash_func
        self._tokenizer = tiktoken.encoding_for_model(tokenizer)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into a list of integers."""
        return self._tokenizer.encode(text)

    def chunk(self, articles: list[dict]) -> list[dict]:
        """Turn each pre-chunked HotPotQA article into one chunk."""
        try:
            chunks = []
            for index, article in enumerate(articles):
                title = article["title"].strip()
                content = article["content"].strip()
                chunk_context = f"The title of this article is {title}. {content}"
                tokens = self.tokenize(chunk_context)
                chunk_hash = self._hash_func.content_hash(chunk_context)
                chunks.append(
                    {
                        "chunk_index": index,
                        "tokens": len(tokens),
                        "content": chunk_context,
                        "chunk_hash": chunk_hash,
                    }
                )
            return chunks
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the HotPotQAChunker chunking function: {e}")
