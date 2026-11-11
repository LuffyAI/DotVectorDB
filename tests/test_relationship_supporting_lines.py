"""Exercise extraction SQL and rollback without importing optional model runtimes."""
import ast
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import AsyncMock


SOURCE = Path(__file__).resolve().parents[1] / "dotvectordb/graph/relationship_indexer.py"
tree = ast.parse(SOURCE.read_text())
indexer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RelationshipIndexer")
methods = [node for node in indexer.body if isinstance(node, ast.AsyncFunctionDef) and node.name in {
    "extract_relation", "get_sentence_highlighted_version_of_chunk", "log_failure",
    "get_chunks_waiting_for_relation_extraction",
}]
namespace = dict(json=json, re=re, traceback=traceback, RELATIONSHIP_PROMPT="{chunk}",
                 ENTITY_EXAMPLE="", INPUT_EXAMPLE="", OUTPUT_EXAMPLE="", Relationships=object)
cls = ast.ClassDef(name="ExtractionUnderTest", bases=[], keywords=[], body=methods, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), namespace)
ExtractionUnderTest = namespace["ExtractionUnderTest"]


class AsyncCursor:
    def __init__(self, cursor):
        self.cursor = cursor

    def __await__(self):
        async def result():
            return self
        return result().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def fetchone(self):
        return self.cursor.fetchone()

    async def fetchall(self):
        return self.cursor.fetchall()

    async def close(self):
        self.cursor.close()


class AsyncConnection:
    """Use real SQLite statements and transactions with an async cursor adapter."""
    def __init__(self, connection):
        self.connection = connection

    def execute(self, *args):
        return AsyncCursor(self.connection.execute(*args))

    async def commit(self):
        self.connection.commit()

    async def rollback(self):
        self.connection.rollback()


class SupportingLineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE chunk (chunk_id INTEGER PRIMARY KEY, content TEXT,
                is_embedded INTEGER DEFAULT 1, are_entities_extracted INTEGER DEFAULT 1,
                are_relations_extracted INTEGER DEFAULT 0);
            CREATE TABLE chunk_to_sentence_mapping (
                chunk_id INTEGER, sentence_number INTEGER, sentence_text TEXT);
            CREATE TABLE relationship (relation_id INTEGER PRIMARY KEY,
                src_entity_id INTEGER, dst_entity_id INTEGER, relation_description TEXT,
                embedding_model_id TEXT, keywords JSON, supporting_lines JSON NOT NULL,
                tokens INTEGER, is_summarized BOOLEAN, is_embedded BOOLEAN DEFAULT FALSE);
            CREATE TABLE chunk_to_relationship_mapping (chunk_id INTEGER, relation_id INTEGER,
                PRIMARY KEY (chunk_id, relation_id));
            CREATE TABLE failure_log (phase TEXT, item_type TEXT, item_id TEXT, error_message TEXT);
            INSERT INTO chunk (chunk_id, content) VALUES (101, 'chunk-a'), (202, 'chunk-b');
        """)
        self.db.executemany("INSERT INTO chunk_to_sentence_mapping VALUES (?, ?, ?)",
                            [(chunk, number, f"Sentence {number}.")
                             for chunk, count in ((101, 5), (202, 3)) for number in range(1, count + 1)])
        self.db.commit()
        self.owner = ExtractionUnderTest()
        self.owner.conn = AsyncConnection(self.db)
        self.owner.lock = asyncio.Lock()
        self.owner.hash_func = lambda value: {"chunk-a": 101, "chunk-b": 202, "model": 404, "11_22": 303}[value]
        self.owner.embedding_model = SimpleNamespace(model_name="model", tokenize=lambda text: text.split())
        self.owner.summary_language = "English"
        self.owner.relationship_extract_prompt = namespace['RELATIONSHIP_PROMPT']
        self.owner.relationship_input_example = namespace['INPUT_EXAMPLE']
        self.owner.relationship_output_example = namespace['OUTPUT_EXAMPLE']
        self.owner.relationship_entity_example = namespace['ENTITY_EXAMPLE']
        self.owner.max_entity_relation_description_token_size = 1000
        self.owner.get_entities_connected_to_chunk = AsyncMock(return_value=[11, 22])
        self.owner.prepare_local_entity_mapping_for_chunk = AsyncMock(return_value=({1: 11, 2: 22}, {1: "A", 2: "B"}))
        self.owner.save_input_and_output_tokens_for_llm = AsyncMock()
        self.owner.language_model = AsyncMock()

    def relation(self, lines):
        return dict(src_entity_id=1, dst_entity_id=2, description="A connects to B.",
                    keywords=["connection"], supporting_lines=lines)

    async def extract(self, relations, chunk="chunk-a"):
        self.owner.language_model.return_value = ({"relations": relations}, {})
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return await self.owner.extract_relation(chunk)

    def evidence(self):
        return json.loads(self.db.execute("SELECT supporting_lines FROM relationship").fetchone()[0])

    async def test_normalizes_and_deduplicates_valid_labels(self):
        self.assertEqual(await self.extract([self.relation(["L5", "l1", "L1", "l2"])]), 1)
        self.assertEqual(self.evidence(), {"101": ["l1", "l2", "l5"]})
        self.assertEqual(self.db.execute("SELECT are_relations_extracted FROM chunk WHERE chunk_id=101").fetchone()[0], 1)

    async def test_merges_chunks_and_repeated_chunk_evidence(self):
        await self.extract([self.relation(["L2", "L4"])])
        self.db.execute("UPDATE relationship SET is_embedded=TRUE")
        self.db.commit()
        self.assertEqual(await self.extract([self.relation(["L1"])], "chunk-b"), 0)
        self.assertEqual(self.db.execute("SELECT is_embedded FROM relationship").fetchone()[0], 0)
        await self.extract([self.relation(["L1", "l2"])])
        self.assertEqual(self.evidence(), {"101": ["l1", "l2", "l4"], "202": ["l1"]})

    async def test_invalid_labels_log_failure_and_leave_chunk_retryable(self):
        for invalid in (None, "L1", {}, ("L1",), [1], [True], ["L0"], ["L6"],
                        ["L01"], ["[L1]"], [" L1"], ["L1\n"], ["L-1"], ["L1.5"]):
            with self.subTest(lines=invalid):
                self.assertEqual(await self.extract([self.relation(invalid)]), 0)
                self.assertEqual(self.db.execute("SELECT COUNT(*) FROM relationship").fetchone()[0], 0)
                self.assertEqual(self.db.execute("SELECT are_relations_extracted FROM chunk WHERE chunk_id=101").fetchone()[0], 0)
                phase, kind, item = self.db.execute("SELECT phase,item_type,item_id FROM failure_log ORDER BY rowid DESC LIMIT 1").fetchone()
                self.assertEqual((phase, kind, item), ("relation", "chunk", "101"))
                self.assertIn(("chunk-a",), await self.owner.get_chunks_waiting_for_relation_extraction())

    async def test_missing_list_is_a_failure(self):
        relation = self.relation(["L1"])
        del relation["supporting_lines"]
        self.assertEqual(await self.extract([relation]), 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM failure_log").fetchone()[0], 1)

    async def test_invalid_later_relation_rolls_back_whole_chunk(self):
        await self.extract([self.relation(["L1"]), self.relation(["L6"])])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM relationship").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM chunk_to_relationship_mapping").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM failure_log").fetchone()[0], 1)
        self.owner.save_input_and_output_tokens_for_llm.assert_not_awaited()

    async def test_failed_update_preserves_preexisting_evidence(self):
        await self.extract([self.relation(["L1"])])
        before = self.db.execute("SELECT * FROM relationship").fetchall()
        await self.extract([self.relation(["L2"]), self.relation(["L6"])], "chunk-b")
        self.assertEqual(self.db.execute("SELECT * FROM relationship").fetchall(), before)
        self.assertEqual(self.db.execute("SELECT are_relations_extracted FROM chunk WHERE chunk_id=202").fetchone()[0], 0)

    async def test_empty_list_is_allowed_by_existing_prompt(self):
        self.assertEqual(await self.extract([self.relation([])]), 1)
        self.assertEqual(self.evidence(), {"101": []})


if __name__ == "__main__":
    unittest.main()
