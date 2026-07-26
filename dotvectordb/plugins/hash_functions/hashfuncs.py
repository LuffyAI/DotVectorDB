import hashlib
from dotvectordb.interfaces.HashingFunc import HashingFunc

class SHA256HashingFunc(HashingFunc):
    def compute_hash(self, content: str) -> str:
        """Generate a SHA-256 hash for the given content string."""
        # Create a new sha256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the bytes of the input text
        sha256_hash.update(content.encode('utf-8'))

        # int.from_bytes converts the first 8 bytes of the hash digest to an integer
        # It takes the first 64 bits of the SHA-256 digest and turns that into a SQLite-compatible integer.
        return int.from_bytes(sha256_hash.digest()[:8], byteorder="big", signed=True)
