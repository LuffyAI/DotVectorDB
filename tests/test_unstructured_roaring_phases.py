"""Exercise the constructor's four phases with real SQLite and Roaring stores."""
import ast
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import traceback
import unittest
from unittest.mock import AsyncMock, patch

from pyroaring import BitMap
from tqdm.auto import tqdm
from test_relationship_supporting_lines import AsyncConnection
from test_roaring_later_hops import MODULE, exact_hop


SOURCE = Path(__file__).resolve().parents[1] / "dotvectordb/UnstructuredGraphConstructor.py"
tree = ast.parse(SOURCE.read_text())
source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnstructuredGraphConstructor")
methods = [n for n in source_class.body if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
           and n.name in {"__init__", "construct_reachability_map", "construct_GraphRAG"}]
base_tree = ast.parse((SOURCE.parent / "DotVDB.py").read_text())
base_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == "DotVectorDB")
methods.append(next(n for n in base_class.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_require_size_positive"))
class ConstructorBase:
    def __init__(self, path):
        self.path = str(path)


namespace = dict(ConstructorBase=ConstructorBase, Path=Path, LmdbStore=MODULE.LmdbStore, json=json, BitMap=BitMap, tqdm=tqdm,
                 ExactHopBitmapBuilder=MODULE.ExactHopBitmapBuilder, traceback=traceback)
cls = ast.ClassDef(name="ConstructorUnderTest", bases=[ast.Name(id="ConstructorBase", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), namespace)
ConstructorUnderTest = namespace["ConstructorUnderTest"]


class RecordedConnection(AsyncConnection):
    def __init__(self, connection):
        super().__init__(connection)
        self.queries = []

    def execute(self, sql, parameters=()):
        self.queries.append((sql, parameters))
        return super().execute(sql, parameters)


class ConstructorPhaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE global_config (id INTEGER PRIMARY KEY, current_version_count INTEGER);
            INSERT INTO global_config VALUES (1, 1);
            CREATE TABLE entity (entity_id INTEGER PRIMARY KEY, degrees INTEGER);
            CREATE TABLE entity_to_entity_type_mapping (entity_id INTEGER, entity_type TEXT,
                PRIMARY KEY(entity_id, entity_type));
        """)
        self.owner = ConstructorUnderTest(Path(self.temp.name) / "graph.db")
        self.owner.conn = RecordedConnection(self.db)
        self.owner.verbose = False
        self.owner.global_config = {"current_version_count": 1}
        self.messages = []
        self.owner._graph_construction_print = self.messages.append
        self.owner._require_open_connection = lambda: None
        self.owner.lock = asyncio.Lock()
        self.graph = {101: {202}, 202: {101, 505}, 303: set(), 505: {202}}
        self.types = {101: ["Person", "Founder"], 202: ["Organization"], 303: [], 505: ["Organization"]}
        self.relationships = [dict(relation_id=1, src_entity_id=101, dst_entity_id=202),
                              dict(relation_id=2, src_entity_id=202, dst_entity_id=505)]
        self.save_graph()

    def save_graph(self):
        for node, neighbors in self.graph.items():
            self.db.execute("INSERT INTO entity VALUES (?, ?) ON CONFLICT(entity_id) DO UPDATE SET degrees=excluded.degrees",
                            (node, len(neighbors)))
            for kind in self.types[node]:
                self.db.execute("INSERT OR IGNORE INTO entity_to_entity_type_mapping VALUES (?, ?)", (node, kind))
        self.db.commit()

    def assert_index(self, metadata):
        builder = MODULE.ExactHopBitmapBuilder(metadata["path"], max_hop=metadata["max_hop"])
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
            for anchor in self.graph:
                row = int(mapping[f"id:{anchor}".encode()])
                for hop in range(1, metadata["max_hop"] + 1):
                    expected = set(exact_hop(self.graph, anchor, hop))
                    actual = {int(mapping[f"row:{n}".encode()]) for n in builder._read_hop_bitmap(row, hop)}
                    self.assertEqual(actual, expected, (anchor, hop))
        for anchor in self.graph:
            for hop in range(1, metadata["max_hop"] + 1):
                expected = set(exact_hop(self.graph, anchor, hop))
                for kind in ("Person", "Founder", "Organization", "Employer"):
                    self.assertEqual(set(builder.get_hop_neighbors_by_type(anchor, hop, kind)),
                                     {n for n in expected if kind in self.types[n]})
        self.assertFalse(list(Path(metadata["path"]).rglob("*.npy")))
        self.assertFalse(list(Path(metadata["path"]).glob(".roaring-workers-*")))

    async def test_initial_build_and_incremental_phases(self):
        first = await self.owner.construct_reachability_map(list(self.graph), self.relationships, [],
                                                            batch_size=2, workers=2)
        self.assertFalse(any("FROM global_config" in sql for sql, _ in self.owner.conn.queries))
        self.assertEqual(first["new_entity_rows"], 4)
        self.assertEqual(first["one_hop_additions"], 4)
        self.assertEqual(first["later_hop_rows"], {"hop_2_rows": 4, "hop_3_rows": 4})
        self.assert_index(first)
        self.graph[404] = {101}
        self.graph[101].add(404)
        self.types[404] = ["Person"]
        self.types[202].append("Employer")
        self.save_graph()
        self.db.execute("UPDATE global_config SET current_version_count=2")
        self.owner.global_config = {"current_version_count": 2}
        self.db.commit()
        self.owner.conn.queries.clear()
        second = await self.owner.construct_reachability_map(
            [404], [dict(relation_id=3, src_entity_id=404, dst_entity_id=101)], [202],
            batch_size=2, workers=2)
        self.assertEqual(second["path"], first["path"])
        self.assertEqual(second["indexed_version"], 2)
        self.assertEqual(second["entity_count"], 5)
        self.assertEqual(second["new_entity_rows"], 1)
        self.assertEqual(second["one_hop_additions"], 2)
        queried = {n for sql, params in self.owner.conn.queries if "SELECT entity.entity_id" in sql for n in params}
        self.assertEqual(queried, {404, 202, 101})
        self.assert_index(second)
        checkpoint = json.loads(Path(self.owner.path).with_suffix("").with_name("graph__graph.pkl.memory_map.current.json").read_text())
        self.assertEqual(checkpoint, second)
        reopened = ConstructorUnderTest(self.owner.path)
        self.assertEqual(reopened.previous_max_hop, 3)
        self.assertEqual(reopened.reachability_output_dir, Path(second["path"]))
        stats = json.loads((Path(second["path"]) / "entity_stats.json").read_text())
        self.assertEqual(stats, {"avg_entity_degree": 1.2, "max_entity_degree": 2, "least_entity_degree": 0})

    async def test_extra_hops_are_optional_and_can_run_later(self):
        one = await self.owner.construct_reachability_map(list(self.graph), self.relationships, [],
                                                          max_hop=1, batch_size=2, workers=1)
        self.assertEqual(one["later_hop_rows"], {})
        self.assertEqual(len(self.messages), 3)
        three = await self.owner.construct_reachability_map([], [], [], workers=2)
        self.assertEqual(three["new_entity_rows"], 0)
        self.assertEqual(three["later_hop_rows"], {"hop_2_rows": 4, "hop_3_rows": 4})
        self.assert_index(three)

    async def test_extending_completed_two_hops_to_three(self):
        await self.owner.construct_reachability_map(list(self.graph), self.relationships, [],
                                                     max_hop=2, batch_size=2, workers=1)
        result = await self.owner.construct_reachability_map([], [], [], workers=2)
        self.assertEqual(result["later_hop_rows"], {"hop_3_rows": 4})
        self.assert_index(result)

    async def test_retry_reuses_allocated_and_assigned_rows(self):
        with patch.object(MODULE.ExactHopBitmapBuilder, "add_entity_types", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                await self.owner.construct_reachability_map(list(self.graph), self.relationships, [], workers=1)
        result = await self.owner.construct_reachability_map(list(self.graph), self.relationships, [], workers=1)
        self.assertEqual(result["entity_count"], 4)
        self.assertEqual(result["new_entity_rows"], 0)
        again = await self.owner.construct_reachability_map(list(self.graph), self.relationships, [], workers=1)
        self.assertEqual(again["entity_count"], 4)
        self.assertEqual(again["new_entity_rows"], 0)
        self.assertEqual(again["one_hop_additions"], 0)
        self.assert_index(again)

    async def test_retry_seeds_degrees_for_already_assigned_rows(self):
        output = Path(self.owner.path).with_suffix("").with_name("graph__graph.pkl.memory_map")
        builder = MODULE.ExactHopBitmapBuilder(output)
        builder.add_new_entity_ids(len(self.graph))
        builder.add_entity_types(self.types.items())
        result = await self.owner.construct_reachability_map(
            list(self.graph), self.relationships, [], batch_size=1, workers=1)
        self.assertEqual(result["new_entity_rows"], 0)
        self.assert_index(result)

    async def test_empty_graph(self):
        self.db.execute("DELETE FROM entity")
        self.db.execute("DELETE FROM entity_to_entity_type_mapping")
        self.db.commit()
        result = await self.owner.construct_reachability_map([], [], [], workers=1)
        self.assertEqual(result["entity_count"], 0)
        self.assertEqual(result["one_hop_additions"], 0)
        self.assertEqual(result["later_hop_rows"], {})

    async def test_graph_rag_passes_the_tracked_changes(self):
        new, updated, relationships = [101], [202], self.relationships
        self.owner.ingest_documents = AsyncMock()
        self.owner.extract_and_embed_entities = AsyncMock(return_value=(new, updated))
        self.owner.extract_and_embed_relationships = AsyncMock(return_value=relationships)
        self.owner.generate_global_descriptions = AsyncMock()
        self.owner.create_BM25_index_for_table = AsyncMock()
        self.owner.construct_reachability_map = AsyncMock()
        await self.owner.construct_GraphRAG()
        self.owner.construct_reachability_map.assert_awaited_once_with(
            new_entities=new, new_relationships=relationships, updated_entities=updated)


if __name__ == "__main__":
    unittest.main()
