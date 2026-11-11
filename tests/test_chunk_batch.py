"""Batch chunk insertion keeps numbered lines and trigger updates transactional."""
import asyncio
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from dotvectordb.DotVDB import DotVectorDB
from dotvectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc
from test_relationship_supporting_lines import AsyncConnection


class BatchConnection(AsyncConnection):
    async def executemany(self, *args):
        return self.connection.executemany(*args)


class ChunkBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sql = sqlite3.connect(':memory:')
        self.addCleanup(self.sql.close)
        schema = Path(__file__).resolve().parents[1] / 'dotvectordb/schema'
        for name in ('settings.sql', 'chunk.sql', 'entity.sql', 'relationship.sql'):
            self.sql.executescript((schema / name).read_text())
        self.owner = object.__new__(DotVectorDB)
        self.owner.conn = BatchConnection(self.sql)
        self.owner.conn.commit = AsyncMock(wraps=self.owner.conn.commit)
        self.owner.conn.rollback = AsyncMock(wraps=self.owner.conn.rollback)
        self.owner.lock = asyncio.Lock()
        self.owner.hashing_func = SHA256HashingFunc()
        self.owner.chunking_func = SimpleNamespace(tokenize=lambda text: text.split())

    @staticmethod
    def chunk(index, tokens=1):
        return {'content': f'Passage {index}', 'tokens': tokens,
                'numbered_lines': {1: f'Passage {index}'}}

    async def test_default_batch_size_commits_once_per_10000_chunks(self):
        await self.owner.add_chunk_batch((self.chunk(i) for i in range(10001)), 'source', 'model')
        self.assertEqual(self.owner.conn.commit.await_count, 2)
        self.assertEqual(self.sql.execute('SELECT COUNT(*) FROM chunk').fetchone()[0], 10001)
        self.assertEqual(self.sql.execute('SELECT COUNT(*) FROM chunk_to_sentence_mapping').fetchone()[0], 10001)
        self.assertEqual(self.sql.execute('SELECT num_chunks FROM source').fetchone()[0], 10001)
        self.assertEqual(self.sql.execute('SELECT num_chunks FROM global_config').fetchone()[0], 10001)

    async def test_repeated_chunk_updates_tokens_and_lines_without_duplicate_count(self):
        await self.owner.add_chunk_batch([self.chunk(1)], 'source', 'model')
        updated = self.chunk(1, tokens=None)
        updated['numbered_lines'] = {1: 'Updated line'}
        await self.owner.add_chunk_batch([updated], 'source', 'model')
        self.assertEqual(self.sql.execute('SELECT tokens FROM chunk').fetchone()[0], 2)
        self.assertEqual(self.sql.execute('SELECT sentence_text FROM chunk_to_sentence_mapping').fetchone()[0], 'Updated line')
        self.assertEqual(self.sql.execute('SELECT num_chunks FROM source').fetchone()[0], 1)

    async def test_failed_batch_rolls_back_its_chunks_lines_and_metadata_changes(self):
        self.sql.executescript('''
            CREATE TRIGGER reject_line BEFORE INSERT ON chunk_to_sentence_mapping
            WHEN NEW.sentence_text = 'reject'
            BEGIN SELECT RAISE(ABORT, 'Rejected line'); END;
        ''')
        rows = [self.chunk(i) for i in range(4)]
        rows[3]['numbered_lines'] = {1: 'reject'}
        with self.assertRaises(sqlite3.IntegrityError):
            await self.owner.add_chunk_batch(rows, 'source', 'model', batch_size=2)
        self.assertEqual(self.owner.conn.commit.await_count, 1)
        self.owner.conn.rollback.assert_awaited_once()
        self.assertEqual(self.sql.execute('SELECT COUNT(*) FROM chunk').fetchone()[0], 2)
        self.assertEqual(self.sql.execute('SELECT COUNT(*) FROM chunk_to_sentence_mapping').fetchone()[0], 2)
        self.assertEqual(self.sql.execute('SELECT num_chunks FROM source').fetchone()[0], 2)
        self.assertEqual(self.sql.execute('SELECT num_chunks FROM global_config').fetchone()[0], 2)

    async def test_invalid_tokens_are_rejected_before_batch_writes(self):
        for tokens in (True, -1, '1'):
            with self.assertRaises((TypeError, ValueError)):
                await self.owner.add_chunk_batch([self.chunk(1), self.chunk(2, tokens)], 'source', 'model')
        self.assertEqual(self.sql.execute('SELECT COUNT(*) FROM chunk').fetchone()[0], 0)
        self.assertEqual(self.sql.execute('SELECT COUNT(*) FROM source').fetchone()[0], 0)
        self.owner.conn.commit.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
