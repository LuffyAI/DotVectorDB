import asyncio
import sqlite3
import unittest
from pathlib import Path

from dotvectordb.DotVDB import DotVectorDB
from dotvectordb.graph.entity_indexer4 import EntityIndexer4
from dotvectordb.graph.relationship_indexer3 import RelationshipIndexer
from dotvectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "dotvectordb" / "schema"


class FakeEmbeddingModel:
    model_name = "test-embedding-model"

    @staticmethod
    def tokenize(text):
        return text.split()


class AsyncCursorAdapter:
    def __init__(self, cursor):
        self._cursor = cursor

    @property
    def description(self):
        return self._cursor.description

    async def fetchone(self):
        return self._cursor.fetchone()

    async def fetchall(self):
        return self._cursor.fetchall()

    async def close(self):
        self._cursor.close()


class AsyncSQLiteAdapter:
    """Minimal async facade used to exercise indexers without worker threads."""

    def __init__(self):
        self._connection = sqlite3.connect(":memory:")
        self._connection.execute("PRAGMA foreign_keys = ON")

    async def execute(self, sql, parameters=()):
        return AsyncCursorAdapter(self._connection.execute(sql, parameters))

    async def executemany(self, sql, parameters):
        return AsyncCursorAdapter(self._connection.executemany(sql, parameters))

    async def executescript(self, sql):
        self._connection.executescript(sql)

    async def commit(self):
        self._connection.commit()

    async def rollback(self):
        self._connection.rollback()

    async def close(self):
        self._connection.close()


class CustomGraphIngestionRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = AsyncSQLiteAdapter()
        self.addAsyncCleanup(self.conn.close)
        for schema_name in (
            "settings.sql",
            "chunk.sql",
            "entity.sql",
            "relationship.sql",
            "failure_log.sql",
        ):
            await self.conn.executescript(
                (SCHEMA / schema_name).read_text(encoding="utf-8")
            )
        self.hashing = SHA256HashingFunc()
        self.lock = asyncio.Lock()
        self.embedding_model = FakeEmbeddingModel()

    async def test_custom_records_use_integer_chunk_ids_and_are_mapped(self):
        content = "Alice works with Bob on the DotRAG project."
        chunks = [
            {
                "chunk_id": "local-chunk-1",
                "source_name": "test-source",
                "source_description": "Regression test source",
                "content": content,
            }
        ]
        entities = [
            {
                "entity_id": "alice-local",
                "chunk_id": "local-chunk-1",
                "entity_name": "Alice",
                "entity_description": "Alice works on DotRAG.",
                "entity_types": ["person"],
                "questions": ["Who works on DotRAG?"],
            },
            {
                "entity_id": "bob-local",
                "chunk_id": "local-chunk-1",
                "entity_name": "Bob",
                "entity_description": "Bob works with Alice.",
                "entity_types": ["person"],
                "questions": ["Who works with Alice?"],
            },
        ]
        relationships = [
            {
                "chunk_id": "local-chunk-1",
                "src_entity_id": "alice-local",
                "dst_entity_id": "bob-local",
                "relation_description": "Alice works with Bob.",
                "keywords": ["works with"],
                "weight": 1,
                "questions": ["Who does Alice work with?"],
            }
        ]

        database = object.__new__(DotVectorDB)
        database.hashing_func = self.hashing
        database.conn = self.conn
        database.lock = self.lock
        source_name_to_id = database._DotVectorDB__verify_source_json(chunks)
        chunk_mapping, prepared_chunks = (
            database._DotVectorDB__verify_prechunked_corpus(
                source_name_to_id,
                chunks,
            )
        )
        entity_mapping, prepared_entities = (
            database._DotVectorDB__verify_prebuilt_entity_json(
                chunk_mapping,
                prepared_chunks,
                entities,
            )
        )
        prepared_relationships = (
            database._DotVectorDB__verify_prebuilt_relationship_json(
                chunk_mapping,
                entity_mapping,
                prepared_chunks,
                prepared_entities,
                relationships,
            )
        )

        chunk_id = prepared_chunks[0]["chunk_id"]
        self.assertIsInstance(chunk_id, int)
        self.assertTrue(
            all(entity["chunk_id"] == chunk_id for entity in prepared_entities)
        )
        self.assertEqual(prepared_relationships[0]["chunk_id"], chunk_id)

        model_id = str(self.hashing.content_hash(self.embedding_model.model_name))
        source_id = prepared_chunks[0]["source_id"]
        await self.conn.execute(
            "INSERT INTO source (id, name) VALUES (?, ?)",
            (source_id, "test-source"),
        )
        await self.conn.execute(
            "INSERT INTO embedding_model (id, name) VALUES (?, ?)",
            (model_id, self.embedding_model.model_name),
        )
        await self.conn.execute(
            "INSERT INTO entity_types (entity_type, type_description) VALUES (?, ?)",
            ("person", "A person."),
        )
        await self.conn.execute(
            """
            INSERT INTO chunk (
                chunk_id, source_id, content, embedding_model_id, is_embedded
            ) VALUES (?, ?, ?, ?, 1)
            """,
            (chunk_id, source_id, content, model_id),
        )
        await self.conn.commit()

        entity_indexer = object.__new__(EntityIndexer4)
        entity_indexer.conn = self.conn
        entity_indexer.lock = self.lock
        entity_indexer.hash_func = self.hashing.content_hash
        entity_indexer.embedding_model = self.embedding_model
        entity_indexer.max_entity_relation_description_token_size = 1200
        await entity_indexer.ingest_custom_entities_from_ingested_chunks(
            prepared_entities,
            async_cooldown=0.001,
            verbose=False,
        )

        relationship_indexer = object.__new__(RelationshipIndexer)
        relationship_indexer.conn = self.conn
        relationship_indexer.lock = self.lock
        relationship_indexer.hash_func = self.hashing.content_hash
        relationship_indexer.embedding_model = self.embedding_model
        relationship_indexer.max_entity_relation_description_token_size = 1200
        await relationship_indexer.ingest_custom_relationships_from_ingested_chunks(
            prepared_relationships,
            async_cooldown=0.001,
            verbose=False,
        )

        await database._DotVectorDB__verify_custom_records_persisted(
            prepared_entities,
            prepared_relationships,
        )
        entity_count = (
            await (await self.conn.execute("SELECT COUNT(*) FROM entity")).fetchone()
        )[0]
        relationship_count = (
            await (
                await self.conn.execute("SELECT COUNT(*) FROM relationship")
            ).fetchone()
        )[0]
        entity_mappings = await (
            await self.conn.execute(
                "SELECT chunk_id, entity_id FROM chunk_to_entity_mapping"
            )
        ).fetchall()
        relationship_mappings = await (
            await self.conn.execute(
                "SELECT chunk_id, relation_id FROM chunk_to_relationship_mapping"
            )
        ).fetchall()

        self.assertEqual(entity_count, 2)
        self.assertEqual(relationship_count, 1)
        self.assertEqual(len(entity_mappings), 2)
        self.assertTrue(all(row[0] == chunk_id for row in entity_mappings))
        self.assertEqual(len(relationship_mappings), 1)
        self.assertEqual(relationship_mappings[0][0], chunk_id)


if __name__ == "__main__":
    unittest.main()
