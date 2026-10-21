import asyncio
from pathlib import Path
import uuid

import aiosqlite
import numpy as np

from dotvectordb.DotVDB import DotVectorDB
from dotvectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc


class CountingEmbeddingModel:
    model_name = "counting-test-model"
    embedding_dim = 3

    def __init__(self):
        self.calls = []

    async def embed(self, text):
        self.calls.append(text)
        await asyncio.sleep(0)
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32), len(text)


class UnusedLanguageModel:
    pass


def make_database(path, embedding_model):
    database = DotVectorDB.__new__(DotVectorDB)
    database.path = str(path)
    database.embedding_model = embedding_model
    database.language_model = UnusedLanguageModel()
    database.hashing_func = SHA256HashingFunc()
    database.verbose = False
    database.conn = None
    database.lock = asyncio.Lock()
    database._query_embedding_tasks = {}
    database._query_embedding_tasks_lock = asyncio.Lock()
    database._sql_browser_server = None
    return database


async def open_cache_database(database):
    database.conn = await aiosqlite.connect(database.path)
    schema = Path(
        "dotvectordb/schema/query_embeddings.sql"
    ).read_text(encoding="utf-8")
    await database.conn.executescript(schema)
    await database.conn.commit()
    database.DIM = 3


def test_query_embeddings_are_cached_persistently_and_concurrently():
    database_path = Path(
        "/tmp"
    ) / f"dotvectordb-query-cache-{uuid.uuid4().hex}.db"

    async def run_test():
        first_model = CountingEmbeddingModel()
        database = make_database(database_path, first_model)
        await open_cache_database(database)

        concurrent_results = await asyncio.gather(
            *(
                database._get_or_create_query_embedding("same query")
                for _ in range(5)
            )
        )
        assert len(first_model.calls) == 1
        assert sum(tokens for _, tokens in concurrent_results) == len(
            "same query"
        )

        cached_embedding, cached_tokens = (
            await database._get_or_create_query_embedding("same query")
        )
        np.testing.assert_array_equal(
            cached_embedding,
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        )
        assert cached_tokens == 0
        assert len(first_model.calls) == 1
        await database.close()

        second_model = CountingEmbeddingModel()
        reopened_database = make_database(database_path, second_model)
        await open_cache_database(reopened_database)
        _, reopened_tokens = (
            await reopened_database._get_or_create_query_embedding(
                "same query"
            )
        )
        assert reopened_tokens == 0
        assert second_model.calls == []

        row = await (
            await reopened_database.conn.execute(
                "SELECT query, dimensions, tokens FROM query_embeddings"
            )
        ).fetchone()
        assert row == ("same query", 3, len("same query"))
        await reopened_database.close()

    try:
        asyncio.run(run_test())
    finally:
        database_path.unlink(missing_ok=True)
