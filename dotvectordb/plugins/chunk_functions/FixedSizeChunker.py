import tiktoken
from dotvectordb.interfaces.ChunkingFunc import ChunkingFunc
from dotvectordb.interfaces.HashingFunc import HashingFunc
import os
import traceback
from tqdm.auto import tqdm

class FixedSizeChunker(ChunkingFunc):
    def __init__(self, hash_func: HashingFunc, tokenizer=os.getenv("OPENAI_TOKENIZER", "gpt-4o"), max_token_size=1024, overlap_token_size=128):
        self.max_token_size = max_token_size
        self.overlap_token_size = overlap_token_size
        self._hash_func = hash_func
        self._tokenizer = tiktoken.encoding_for_model(tokenizer)

    def break_into_numbered_lines(self, text: str) -> dict[int, str]:
        return dict(enumerate(
            (line for line in text.splitlines() if line.strip()),
            start=1,
        ))

    def chunk(self, text: str) -> list[str]:
        """Chunk text into a list of strings based on max_token_size and overlap_token_size."""
        try:
            tokens = self.tokenize(text)
            chunks = []
            for index, start in enumerate(tqdm(
                range(0, len(tokens), self.max_token_size - self.overlap_token_size),
                desc="Chunking document", unit="chunk",
            )):
                chunk_context = self._tokenizer.decode(tokens[start:start + self.max_token_size]).strip()
                hash = self._hash_func.content_hash(chunk_context)
                chunks.append(
                            {"chunk_index": index,
                            "tokens": min(self.max_token_size, len(tokens) - start),
                            "content": chunk_context,
                            "chunk_hash": hash,
                            "numbered_lines": self.break_into_numbered_lines(chunk_context)}
                            )
            return chunks
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the FixedSizeChunker chunking function: {e}")
