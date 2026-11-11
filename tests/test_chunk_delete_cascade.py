"""Chunk deletion preserves shared graph data and cascades vector cleanup."""
import ast
import asyncio
import json
from pathlib import Path
import sqlite3
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from test_relationship_supporting_lines import AsyncConnection

ROOT = Path(__file__).resolve().parents[1] / "dotvectordb"
source = ast.parse((ROOT / "DotVDB.py").read_text())
source_class = next(node for node in source.body if isinstance(node, ast.ClassDef))
method = next(node for node in source_class.body
              if isinstance(node, ast.AsyncFunctionDef) and node.name == "add_embedding")
namespace = {"traceback": traceback}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             "embedding_under_test", "exec"), namespace)


fts_source = ast.parse((ROOT / "UnstructuredGraphConstructor.py").read_text())
fts_class = next(node for node in fts_source.body if isinstance(node, ast.ClassDef))
fts_method = next(node for node in fts_class.body
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "create_BM25_index_for_table")
exec(compile(ast.fix_missing_locations(ast.Module(body=[fts_method], type_ignores=[])),
             "bm25_under_test", "exec"), namespace)


class ChunkDeletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_entity_counts_follow_unique_support_and_cascades(self):
        def check_counts():
            for source_id, recorded in self.db.execute('SELECT id, num_entities FROM source').fetchall():
                actual = self.db.execute('''
                    SELECT COUNT(DISTINCT mapping.entity_id)
                    FROM chunk_to_entity_mapping AS mapping
                    JOIN chunk ON chunk.chunk_id = mapping.chunk_id
                    WHERE chunk.source_id = ?
                ''', (source_id,)).fetchone()[0]
                self.assertEqual(recorded, actual, source_id)

        check_counts()
        self.assertEqual(self.db.execute("SELECT num_entities FROM source WHERE id='s'").fetchone()[0], 4)
        self.db.execute('INSERT OR IGNORE INTO chunk_to_entity_mapping(chunk_id,entity_id) VALUES(101,2)')
        check_counts()
        self.db.execute("INSERT INTO source(id,name) VALUES('t','other source')")
        self.db.execute("INSERT INTO chunk(chunk_id,source_id,content,embedding_model_id,tokens) VALUES(404,'t','other','m',1)")
        self.db.execute('INSERT INTO chunk_to_entity_mapping(chunk_id,entity_id) VALUES(404,3)')
        check_counts()
        self.db.execute('DELETE FROM chunk_to_entity_mapping WHERE chunk_id=101 AND entity_id=2')
        check_counts()
        self.db.execute('DELETE FROM chunk WHERE chunk_id=101')
        check_counts()
        self.db.execute('DELETE FROM chunk WHERE chunk_id=202')
        check_counts()
        self.db.execute('DELETE FROM entity WHERE entity_id=3')
        check_counts()
        self.db.commit()
        before = self.db.execute('SELECT id,num_entities FROM source ORDER BY id').fetchall()
        self.db.execute('DELETE FROM chunk WHERE chunk_id=303')
        check_counts()
        self.db.rollback()
        check_counts()
        self.assertEqual(self.db.execute('SELECT id,num_entities FROM source ORDER BY id').fetchall(), before)
        self.db.execute('DELETE FROM source')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM source').fetchone()[0], 0)

    async def asyncSetUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.enable_load_extension(True)
        self.db.load_extension(str(ROOT / "compiled_vec1" / "vec1.so"))
        self.db.enable_load_extension(False)
        for filename in ("settings.sql", "chunk.sql", "entity.sql", "relationship.sql"):
            self.db.executescript((ROOT / "schema" / filename).read_text())
        self.db.execute("INSERT INTO embedding_model(id,name) VALUES('m','model')")
        self.db.execute("INSERT INTO source(id,name) VALUES('s','source')")
        for chunk in (101, 202, 303):
            self.db.execute("INSERT INTO chunk(chunk_id,source_id,content,embedding_model_id,tokens) VALUES(?,'s',?,'m',1)",
                            (chunk, f"chunk {chunk}"))
            self.db.execute("INSERT INTO chunk_to_sentence_mapping VALUES(?,1,'Evidence')", (chunk,))
        for entity in (1, 2, 3, 4):
            self.db.execute("INSERT INTO entity(entity_id,entity_name,entity_description,embedding_model_id,tokens) VALUES(?,?,?,'m',1)",
                            (entity, str(entity), "description"))
        self.db.executemany("INSERT INTO chunk_to_entity_mapping(chunk_id,entity_id) VALUES(?,?)",
                            [(101, 1), (101, 2), (202, 2), (202, 3), (303, 3), (303, 4)])
        for relation, src, dst, chunks in ((11, 1, 2, [101]), (12, 2, 3, [101, 202]), (13, 3, 4, [303])):
            self.db.execute("INSERT INTO relationship(relation_id,src_entity_id,dst_entity_id,relation_description,keywords,supporting_lines,embedding_model_id,tokens) VALUES(?,?,?,'description','[]',?,'m',1)",
                            (relation, src, dst, json.dumps({str(chunk): ["l1"] for chunk in chunks})))
            for chunk in chunks:
                self.db.execute("INSERT INTO chunk_to_relationship_mapping VALUES(?,?)", (chunk, relation))
                self.db.execute("INSERT INTO sentence_to_relationship_mapping VALUES(?,1,?)", (chunk, relation))
        self.db.commit()
        self.owner = SimpleNamespace(
            conn=AsyncConnection(self.db), lock=asyncio.Lock(), DIM=2,
            embedding_model=SimpleNamespace(embed=AsyncMock(
                return_value=(np.array([1, 2], dtype=np.float32), 1))),
        )
        setattr(self.owner, "__require_open_connection", lambda: None)
        self.owner._require_open_connection = lambda: None
        self.owner.global_config = {}

    async def embed_all(self):
        for table, ids in (("chunk", [101, 202, 303]), ("entity", [1, 2, 3, 4]),
                           ("relationship", [11, 12, 13])):
            for identifier in ids:
                await namespace["add_embedding"](self.owner, table, "content", identifier)
        self.db.execute("DELETE FROM new_and_updated_entity")
        self.db.execute("DELETE FROM new_and_updated_relationship")
        self.db.commit()

    def ids(self, table, column):
        return {row[0] for row in self.db.execute(f"SELECT {column} FROM {table}")}

    def assert_first_chunk_removed(self):
        self.assertEqual(self.ids("entity", "entity_id"), {2, 3, 4})
        self.assertEqual(self.ids("relationship", "relation_id"), {12, 13})
        self.assertEqual(self.ids("deleted_entity", "entity_id"), {1})
        self.assertEqual(self.ids("deleted_relationship", "relation_id"), {11})
        for table in ("chunk_to_sentence_mapping", "chunk_to_entity_mapping",
                      "chunk_to_relationship_mapping", "sentence_to_relationship_mapping"):
            self.assertEqual(self.db.execute(f"SELECT COUNT(*) FROM {table} WHERE chunk_id=101").fetchone()[0], 0)
        self.assertEqual(json.loads(self.db.execute("SELECT supporting_lines FROM relationship WHERE relation_id=12").fetchone()[0]),
                         {"202": ["l1"]})
        self.assertEqual(self.db.execute("SELECT num_chunks,num_entities,num_relations FROM global_config").fetchone(), (2, 3, 2))
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(), [])

    async def test_shared_support_and_embedding_cleanup(self):
        await self.embed_all()
        self.db.execute("DELETE FROM chunk WHERE chunk_id=101")
        self.db.commit()
        self.assert_first_chunk_removed()
        self.assertEqual(self.ids("chunk_embeddings", "rowid"), {202, 303})
        self.assertEqual(self.ids("entity_embeddings", "rowid"), {2, 3, 4})
        self.assertEqual(self.ids("relationship_embeddings", "rowid"), {12, 13})

    async def test_deletion_before_any_embedding_table_exists(self):
        self.db.execute("DELETE FROM chunk WHERE chunk_id=101")
        self.db.commit()
        self.assert_first_chunk_removed()

    async def test_last_support_and_multirow_delete(self):
        await self.embed_all()
        self.db.execute("DELETE FROM chunk WHERE chunk_id IN (101,202)")
        self.db.commit()
        self.assertEqual(self.ids("entity", "entity_id"), {3, 4})
        self.assertEqual(self.ids("relationship", "relation_id"), {13})
        self.db.execute("DELETE FROM chunk WHERE chunk_id=303")
        self.db.commit()
        for table in ("chunk", "entity", "relationship", "chunk_embeddings", "entity_embeddings", "relationship_embeddings"):
            self.assertEqual(self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        self.assertEqual(self.ids("deleted_entity", "entity_id"), {1, 2, 3, 4})
        self.assertEqual(self.ids("deleted_relationship", "relation_id"), {11, 12, 13})

    async def test_rollback_restores_graph_vectors_citations_and_tracking(self):
        await self.embed_all()
        tables = ("chunk", "entity", "relationship", "global_config", "deleted_entity", "deleted_relationship",
                  "new_and_updated_entity", "new_and_updated_relationship", "chunk_to_sentence_mapping",
                  "chunk_to_entity_mapping", "chunk_to_relationship_mapping", "sentence_to_relationship_mapping",
                  "chunk_embeddings", "entity_embeddings", "relationship_embeddings")
        before = {table: self.db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() for table in tables}
        self.db.execute("DELETE FROM chunk WHERE chunk_id=101")
        self.db.rollback()
        for table in tables:
            self.assertEqual(self.db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall(), before[table], table)

    async def build_bm25(self):
        for table in ("chunk", "entity", "relationship"):
            await namespace["create_BM25_index_for_table"](self.owner, table)

    def matches(self, table, query):
        return {row[0] for row in self.db.execute(
            f"SELECT rowid FROM {table}_fts WHERE {table}_fts MATCH ?", (query,)
        )}

    def assert_fts_integrity(self):
        for table in ("chunk", "entity", "relationship"):
            self.db.execute(f"INSERT INTO {table}_fts({table}_fts,rank) VALUES('integrity-check',1)")

    async def test_bm25_cleanup_follows_chunk_cascade(self):
        await self.embed_all()
        await self.build_bm25()
        # Rebuilding twice must not register duplicate maintenance triggers.
        await self.build_bm25()
        self.db.execute("DELETE FROM chunk WHERE chunk_id=101")
        self.db.commit()
        self.assertEqual(self.matches("chunk", "chunk"), {202, 303})
        self.assertEqual(self.matches("entity", "description"), {2, 3, 4})
        self.assertEqual(self.matches("relationship", "description"), {12, 13})
        self.assert_fts_integrity()
        self.db.execute("DELETE FROM source WHERE id='s'")
        self.db.commit()
        for table, query in (("chunk", "chunk"), ("entity", "description"), ("relationship", "description")):
            self.assertEqual(self.matches(table, query), set())
        self.assert_fts_integrity()

    async def test_bm25_tracks_insert_update_and_then_delete(self):
        await self.build_bm25()
        self.db.execute("INSERT INTO entity(entity_id,entity_name,entity_description,embedding_model_id,tokens) VALUES(9,'added','originalterm','m',1)")
        self.assertEqual(self.matches("entity", "originalterm"), {9})
        self.db.execute("UPDATE entity SET entity_description='replacementterm' WHERE entity_id=9")
        self.assertEqual(self.matches("entity", "originalterm"), set())
        self.assertEqual(self.matches("entity", "replacementterm"), {9})
        self.db.execute("DELETE FROM entity WHERE entity_id=9")
        self.assertEqual(self.matches("entity", "replacementterm"), set())
        self.assert_fts_integrity()

    async def test_bm25_deletion_rolls_back_with_source_rows(self):
        await self.build_bm25()
        before = {table: self.matches(table, query) for table, query in
                  (("chunk", "chunk"), ("entity", "description"), ("relationship", "description"))}
        self.db.execute("DELETE FROM chunk WHERE chunk_id=101")
        self.db.rollback()
        for table, query in (("chunk", "chunk"), ("entity", "description"), ("relationship", "description")):
            self.assertEqual(self.matches(table, query), before[table])
        self.assert_fts_integrity()

    async def test_source_cascade_uses_the_same_cleanup(self):
        await self.embed_all()
        self.db.execute("DELETE FROM source WHERE id='s'")
        self.db.commit()
        self.assertEqual(self.ids("entity", "entity_id"), set())
        self.assertEqual(self.ids("relationship", "relation_id"), set())
        self.assertEqual(self.ids("deleted_entity", "entity_id"), {1, 2, 3, 4})
        self.assertEqual(self.ids("deleted_relationship", "relation_id"), {11, 12, 13})


if __name__ == "__main__":
    unittest.main()
