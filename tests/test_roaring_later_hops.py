"""Exact-hop and incremental checks against an independent breadth-first search."""
import importlib.util
import random
import json
import pickle
import sys
import tempfile
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path
from unittest.mock import patch

from pyroaring import BitMap


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "dotvectordb/data_structures/bitmap/RoaringBitMapConstruction.py"
)
SPEC = importlib.util.spec_from_file_location("roaring_builder_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def exact_hop(graph, anchor, hop):
    distances = {anchor: 0}
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        if distances[node] == hop:
            continue
        for neighbor in graph[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return BitMap(node for node, distance in distances.items() if distance == hop)


class LaterHopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create_builder(self, graph, name="index", max_hop=3):
        builder = MODULE.ExactHopBitmapBuilder(
            self.root / name, max_hop=max_hop
        )
        builder.add_new_entity_ids(len(graph))
        builder.add_entity_types((1000 + row, f"type{row % 2}") for row in graph)
        # Degrees are provided before one-hop construction, as in the caller's phase order.
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "w") as mapping:
            for row in graph:
                mapping[f"degree:{row}".encode()] = str(len(graph[row])).encode()
        for row, neighbors in graph.items():
            builder.add_new_1_hop_relationships([
                (1000 + row, [1000 + other for other in neighbors], len(neighbors))
            ])
        return builder

    def assert_index(self, builder, graph):
        for hop in range(2, builder.max_hop + 1):
            for row in graph:
                expected = exact_hop(graph, row, hop)
                self.assertEqual(builder._read_hop_bitmap(row, hop), expected)
                for kind in ("type0", "type1"):
                    type_rows = BitMap(n for n in graph if f"type{n % 2}" == kind)
                    actual = builder._read_hop_bitmap(row, hop, kind)
                    self.assertEqual(actual, expected & type_rows)
                    self.assertEqual(
                        builder.get_hop_neighbors_by_type(1000 + row, hop, kind),
                        [1000 + n for n in expected & type_rows],
                    )

    def test_fast_and_batched_paths_match_bfs(self):
        randomizer = random.Random(42)
        graph = {row: set() for row in range(28)}
        for left in graph:
            for right in range(left + 1, 27):
                if randomizer.random() < 0.15:
                    graph[left].add(right)
                    graph[right].add(left)
        for row in range(1, 15):
            graph[0].add(row)
            graph[row].add(0)
        for name, limit in (("fast", 1_000_000), ("batched", 1)):
            with self.subTest(name=name):
                builder = self.create_builder(graph, name)
                with patch.object(
                    MODULE.ExactHopBitmapBuilder, "_compute_later_hop_rows",
                    wraps=MODULE.ExactHopBitmapBuilder._compute_later_hop_rows,
                ) as expand:
                    result = builder.add_later_hops(limit, batch_size=2, verbose=False)
                self.assertEqual(result, {"hop_2_rows": 28, "hop_3_rows": 28})
                self.assertTrue(all(call.args[0][4] == limit for call in expand.call_args_list))
                self.assert_index(builder, graph)
                self.assertEqual(builder.add_later_hops(limit, verbose=False), {})

    def test_incremental_shortcuts_and_new_nodes(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3},
                 5: set(), 6: {7}, 7: {6}}
        builder = self.create_builder(graph)
        builder.add_later_hops(2, batch_size=1, verbose=False)
        before = builder._read_hop_bitmap(6, 3)
        builder = MODULE.ExactHopBitmapBuilder(builder.output_dir)
        builder.add_new_entity_ids(1)
        builder.add_entity_types([(1008, "type0")])
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "w") as mapping:
            mapping[b"degree:8"] = b"1"
        graph[8] = {4}
        graph[4].add(8)
        graph[0].add(3)
        graph[3].add(0)
        for row, additions in ((0, [3]), (3, [0]), (4, [8]), (8, [4])):
            builder.add_new_1_hop_relationships([
                (1000 + row, [1000 + other for other in additions], len(graph[row]))
            ])
        builder.add_later_hops(2, batch_size=1, verbose=False)
        self.assert_index(builder, graph)
        self.assertEqual(builder._read_hop_bitmap(6, 3), before)

    def test_projected_work_order(self):
        graph = {0: {1, 2, 3}, 1: {0, 4}, 2: {0}, 3: {0}, 4: {1}, 5: set()}
        builder = self.create_builder(graph)
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
            expected = sorted(graph, key=lambda row: (
                int(mapping[f"next_hop_estimate:{row}".encode()]),
                int(mapping[f"degree:{row}".encode()]), row,
            ))
        saved_rows = []
        original = builder._save_roaring_hop_row
        def save(row, hop, bitmap, store=None):
            saved_rows.append((hop, row))
            original(row, hop, bitmap, store=store)
        with patch.object(builder, "_save_roaring_hop_row", side_effect=save):
            builder.add_later_hops(2, verbose=False)
        self.assertEqual([row for hop, row in saved_rows if hop == 2], expected)

    def test_failure_keeps_pending_work_for_restart(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph)
        original = builder._save_roaring_hop_row
        def fail_on_three(row, hop, bitmap, store=None):
            if hop == 3:
                raise OSError("simulated interrupted save")
            original(row, hop, bitmap, store=store)
        with patch.object(builder, "_save_roaring_hop_row", side_effect=fail_on_three):
            with self.assertRaises(OSError):
                builder.add_later_hops(1, batch_size=1, verbose=False)
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
            self.assertTrue(BitMap.deserialize(mapping[b"later_hop_dirty"]))
        builder = MODULE.ExactHopBitmapBuilder(builder.output_dir)
        builder.add_later_hops(1, batch_size=1, verbose=False)
        self.assert_index(builder, graph)

    def test_supplied_modified_entities_run_before_affected_anchors(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}, 4: set()}
        builder = self.create_builder(graph)
        builder.add_later_hops(1, verbose=False)
        writes = []
        original = builder._save_roaring_hop_row
        def save(row, hop, bitmap, store=None):
            writes.append((hop, row))
            original(row, hop, bitmap, store=store)
        with patch.object(builder, "_save_roaring_hop_row", side_effect=save):
            builder.add_later_hops(
                1, modified_entity_ids=[1002], batch_size=1, verbose=False
            )
        self.assertEqual([row for hop, row in writes if hop == 2][0], 2)
        self.assertEqual([row for hop, row in writes if hop == 3][0], 2)
        self.assertNotIn(4, [row for _, row in writes])
        self.assert_index(builder, graph)

    def test_later_type_changes_update_existing_buckets(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph)
        builder.add_later_hops(1, batch_size=1, verbose=False)
        builder.add_entity_types([(1003, "new_type")])
        builder.add_later_hops(1, batch_size=1, verbose=False)
        self.assertEqual(builder._read_hop_bitmap(0, 3, "new_type"), BitMap([3]))
        self.assertEqual(builder.get_hop_neighbors_by_type(1002, 1, "new_type"), [1003])

    def test_empty_buckets_are_replaced_after_a_shortcut(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1}}
        builder = self.create_builder(graph)
        builder.add_later_hops(1, batch_size=1, verbose=False)
        graph[0].add(2)
        graph[2].add(0)
        for row, neighbor in ((0, 2), (2, 0)):
            builder.add_new_1_hop_relationships([(1000 + row, [1000 + neighbor], len(graph[row]))])
        builder.add_later_hops(1, batch_size=1, verbose=False)
        self.assert_index(builder, graph)

    def test_indexed_storage_has_no_entity_folders_and_survives_restart(self):
        graph = {row: {other for other in (row - 1, row + 1) if 0 <= other < 200}
                 for row in range(200)}
        builder = self.create_builder(graph)
        builder.add_later_hops(10, verbose=False)
        self.assertEqual([p for p in builder.output_dir.iterdir() if p.is_dir()], [])
        self.assertLess(len(list(builder.output_dir.iterdir())), 10)
        restarted = MODULE.ExactHopBitmapBuilder(builder.output_dir)
        self.assertEqual(restarted.last_entity_row, 199)
        self.assert_index(restarted, graph)
        self.assertEqual(restarted.get_hop_neighbors_by_type(1000, 1, "missing"), [])


    def test_extend_saved_two_hops_without_recomputing_them(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph, max_hop=2)
        builder.add_later_hops(10, verbose=False)
        builder.max_hop = 3
        with patch.object(builder, "_save_roaring_hop_row", wraps=builder._save_roaring_hop_row) as save:
            counts = builder.add_later_hops(10, modified_entity_ids=[1000 + row for row in graph],
                                            start_hop=3, verbose=False)
        self.assertEqual(counts, {"hop_3_rows": 4})
        self.assertTrue(all(call.args[1] == 3 for call in save.call_args_list))
        self.assert_index(builder, graph)

    def test_default_one_hop_batches_are_bounded_and_keep_stores_open(self):
        builder = MODULE.ExactHopBitmapBuilder(self.root / "batch")
        builder.add_new_entity_ids(2001)
        builder.add_entity_types((1000 + row, "type0") for row in range(2001))
        sizes = []
        original = builder._write_one_hop_worker_batch
        def save(batch, mapping, store):
            sizes.append(len(batch))
            return original(batch, mapping, store)
        with patch.object(builder, "_write_one_hop_worker_batch", side_effect=save), \
             patch.object(builder, "_open_bitmap_store", wraps=builder._open_bitmap_store) as opened:
            added = builder.add_new_1_hop_relationships(
                (1000 + row, [1000 + (row ^ 1)] if row < 2000 else [],
                 1 if row < 2000 else 0) for row in range(2001)
            )
        self.assertEqual(sizes, [1000, 1000, 1])
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(added, 2000)
        self.assertEqual(builder.get_hop_neighbors_by_type(2998, 1, "type0"), [2999])
        self.assertEqual(builder.get_entity_degree(3000), 0)

    def test_batched_incremental_rows_preserve_neighbors_and_update_estimates(self):
        for batch_size in (1, 1000):
            with self.subTest(batch_size=batch_size):
                graph = {0: {1}, 1: {0}, 2: set(), 3: set()}
                builder = self.create_builder(graph, name=f"increment_{batch_size}")
                builder.add_later_hops(10, verbose=False)
                updates = [(1000, [1002], 2), (1001, [1003], 2),
                           (1002, [1000], 1), (1003, [1001], 1)]
                self.assertEqual(builder.add_new_1_hop_relationships(iter(updates), batch_size), 4)
                graph = {0: {1, 2}, 1: {0, 3}, 2: {0}, 3: {1}}
                for row, neighbors in graph.items():
                    self.assertEqual(builder._read_hop_bitmap(row, 1), BitMap(neighbors))
                    self.assertEqual(builder.compute_next_hop_estimates(1000 + row),
                                     sum(len(graph[n]) for n in neighbors))
                builder.add_later_hops(10, verbose=False)
                self.assert_index(builder, graph)
                self.assertEqual(builder.add_new_1_hop_relationships(iter(updates), batch_size), 0)
                for row, neighbors in graph.items():
                    self.assertEqual(builder.compute_next_hop_estimates(1000 + row),
                                     sum(len(graph[n]) for n in neighbors))

    def test_invalid_batch_sizes_are_rejected(self):
        builder = MODULE.ExactHopBitmapBuilder(self.root / "invalid")
        for value in (0, -1):
            with self.assertRaises(ValueError):
                builder.add_new_1_hop_relationships([], value)
        for value in (True, 1.5):
            with self.assertRaises(TypeError):
                builder.add_new_1_hop_relationships([], value)
        self.assertEqual(builder.add_new_1_hop_relationships([]), 0)

    def test_later_hop_store_stays_open_across_checkpointed_batches(self):
        graph = {row: {other for other in (row - 1, row + 1) if 0 <= other < 11}
                 for row in range(11)}
        for size in (1, 4, 10000):
            with self.subTest(size=size):
                builder = self.create_builder(graph, name=f"later_batch_{size}")
                with patch.object(builder, "_open_bitmap_store", wraps=builder._open_bitmap_store) as opened:
                    builder.add_later_hops(2, batch_size=size, verbose=False)
                self.assertEqual(opened.call_count, 2)
                self.assert_index(builder, graph)

    def test_roaring_exposes_no_numpy_build_path(self):
        builder = MODULE.ExactHopBitmapBuilder(self.root / "roaring_only")
        self.assertEqual(builder.last_entity_row, -1)
        for name in ("build", "_save_entity_ids", "_save_adjacency", "_save_entity_types",
                     "_save_later_hops", "entity_ids_path", "hop_offsets_paths",
                     "get_1_hop_neighbors_by_type", "build_workers",
                     "entity_ids", "entity_id_to_dense_id"):
            self.assertFalse(hasattr(builder, name), name)
        graph = {0: {1}, 1: {0, 2}, 2: {1}}
        builder = self.create_builder(graph, name="roaring_only")
        builder.add_later_hops(10, verbose=False)
        self.assert_index(builder, graph)
        self.assertEqual({p.name for p in builder.output_dir.iterdir()},
                         {"bitmaps.lmdb", "bitmaps.lmdb-lock", "entity_ids.lmdb", "entity_ids.lmdb-lock", "entity_row_state.json"})
        self.assertEqual(list(builder.output_dir.rglob("*.npy")), [])

    def test_process_workers_match_serial_and_incremental_updates(self):
        graph = {row: {other for other in (row - 1, row + 1) if 0 <= other < 16}
                 for row in range(16)}
        for workers in (1, 2):
            with self.subTest(workers=workers):
                builder = MODULE.ExactHopBitmapBuilder(self.root / f"workers_{workers}", workers=workers)
                builder.add_new_entity_ids(len(graph))
                builder.add_entity_types((1000 + row, f"type{row % 2}") for row in graph)
                with MODULE.LmdbStore(str(builder.entity_mapping_path), "w") as mapping:
                    for row in graph:
                        mapping[f"degree:{row}".encode()] = str(len(graph[row])).encode()
                count = builder.add_new_1_hop_relationships(
                    ((1000 + row, (1000 + n for n in neighbors), len(neighbors))
                     for row, neighbors in graph.items()), batch_size=5)
                self.assertEqual(count, sum(map(len, graph.values())))
                builder.add_later_hops(2, batch_size=5, verbose=False)
                self.assert_index(builder, graph)
                updated = {row: set(neighbors) for row, neighbors in graph.items()}
                updated[0].add(8); updated[8].add(0)
                builder.add_new_1_hop_relationships([
                    (1000, [1008], len(updated[0])), (1008, [1000], len(updated[8]))], batch_size=5)
                builder.add_later_hops(2, batch_size=5, verbose=False)
                self.assert_index(builder, updated)
                for row in updated:
                    self.assertEqual(builder.compute_next_hop_estimates(1000 + row),
                                     sum(len(updated[n]) for n in updated[row]))
                self.assertEqual(builder.add_later_hops(2, verbose=False), {})

    def test_worker_failure_preserves_dirty_checkpoint(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph)
        builder.workers = 2
        with builder._open_bitmap_store("w") as store:
            key = builder._hop_bitmap_key(1, 1)
            saved = store[key]
            del store[key]
        with self.assertRaises(KeyError):
            builder.add_later_hops(2, batch_size=4, verbose=False)
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
            self.assertTrue(BitMap.deserialize(mapping[b"later_hop_dirty"]))
        with builder._open_bitmap_store("w") as store:
            store[key] = saved
        builder.add_later_hops(2, batch_size=4, verbose=False)
        self.assert_index(builder, graph)

    def test_parallel_one_hop_combines_duplicate_source_rows(self):
        graph = {0: set(), 1: set(), 2: set()}
        builder = self.create_builder(graph)
        builder.workers = 2
        added = builder.add_new_1_hop_relationships([
            (1000, [1001], 2), (1000, [1002], 2),
            (1001, [1000], 1), (1002, [1000], 1)], batch_size=10)
        self.assertEqual(added, 4)
        self.assertEqual(builder._read_hop_bitmap(0, 1), BitMap([1, 2]))
        self.assertEqual(builder.compute_next_hop_estimates(1000), 2)
        builder.add_later_hops(2, batch_size=10, verbose=False)
        self.assert_index(builder, {0: {1, 2}, 1: {0}, 2: {0}})

    def test_empty_parallel_input_needs_no_index(self):
        builder = MODULE.ExactHopBitmapBuilder(self.root / "empty", workers=2)
        self.assertEqual(builder.add_new_1_hop_relationships([]), 0)
        self.assertFalse(builder.output_dir.exists())

    def test_worker_count_validation(self):
        for workers in (0, -1):
            with self.assertRaises(ValueError):
                MODULE.ExactHopBitmapBuilder(self.root, workers=workers)
        for workers in (True, 1.5):
            with self.assertRaises(TypeError):
                MODULE.ExactHopBitmapBuilder(self.root, workers=workers)

    def test_free_worker_starts_next_job_before_slow_worker_finishes(self):
        builder = MODULE.ExactHopBitmapBuilder(self.root, workers=2)
        next_started = Event()
        def work(job):
            if job == "slow":
                self.assertTrue(next_started.wait(3), "next task waited for the slow worker")
            elif job == "next":
                next_started.set()
            return 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            with patch.object(MODULE.ExactHopBitmapBuilder, "_stage_worker_result", side_effect=work):
                builder._run_worker_queue(pool, iter(["slow", "fast", "next"]))
        self.assertTrue(next_started.is_set())

    def test_parallel_queue_tasks_each_receive_full_batch_size(self):
        graph = {row: {other for other in (row - 1, row + 1) if 0 <= other < 7}
                 for row in range(7)}
        builder = self.create_builder(graph)
        builder.workers = 2
        sizes = []
        original = builder._run_worker_queue
        def run(pool, jobs, progress=None):
            jobs = list(jobs)
            task_sizes = []
            for job in jobs:
                with open(job[1], "rb") as source:
                    task_sizes.append(len(pickle.load(source)[2]))
            sizes.append(task_sizes)
            return original(pool, iter(jobs), progress)
        with patch.object(builder, "_run_worker_queue", side_effect=run):
            builder.add_later_hops(2, batch_size=3, verbose=False)
        self.assertEqual(sizes, [[3, 3, 1], [3, 3, 1]])
        self.assert_index(builder, graph)
        self.assertEqual(list(builder.output_dir.glob(".roaring-workers-*")), [])

    def test_repeated_source_across_parallel_batches_preserves_additions(self):
        builder = self.create_builder({0: set(), 1: set(), 2: set()})
        builder.workers = 2
        self.assertEqual(builder.add_new_1_hop_relationships([
            (1000, [1001], 1), (1001, [1000], 1),
            (1000, [1002], 2), (1002, [1000], 1)], batch_size=2), 4)
        graph = {0: {1, 2}, 1: {0}, 2: {0}}
        builder.add_later_hops(2, batch_size=2, verbose=False)
        self.assert_index(builder, graph)
        for row in graph:
            self.assertEqual(builder.compute_next_hop_estimates(1000 + row),
                             sum(len(graph[n]) for n in graph[row]))
        self.assertEqual(list(builder.output_dir.glob(".roaring-workers-*")), [])

    def test_one_hop_range_checkpoint_survives_interrupted_write(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1}, 3: set()}
        builder = self.create_builder({row: set() for row in graph}, max_hop=2)
        builder.workers = 2
        builder._checkpoint_interval_seconds = 0
        original = builder._write_one_hop_worker_batch
        calls = []
        def fail_second(group, mapping, store):
            calls.append(group[0][0])
            if len(calls) == 2:
                raise OSError("interrupted one-hop batch")
            return original(group, mapping, store)
        inputs = [(1000 + row, [1000 + other for other in neighbors], len(neighbors))
                  for row, neighbors in graph.items()]
        with patch.object(builder, "_write_one_hop_worker_batch", side_effect=fail_second):
            with self.assertRaises(OSError):
                builder.add_new_1_hop_relationships(inputs, batch_size=1)
        checkpoint = builder.output_dir / ".batch-checkpoint-one_hop" / "ranges.json"
        self.assertEqual(list(json.loads(checkpoint.read_text()).values()),
                         [True, False, False, False])
        restarted = MODULE.ExactHopBitmapBuilder(builder.output_dir, max_hop=2, workers=2)
        with patch.object(restarted, "_write_one_hop_worker_batch",
                          wraps=restarted._write_one_hop_worker_batch) as writes:
            restarted.add_new_1_hop_relationships(inputs, batch_size=3)
        self.assertEqual(writes.call_count, 3)
        self.assertFalse(checkpoint.exists())
        for row, neighbors in graph.items():
            self.assertEqual(restarted._read_hop_bitmap(row, 1), BitMap(neighbors))
            self.assertEqual(restarted.compute_next_hop_estimates(1000 + row),
                             sum(len(graph[other]) for other in neighbors))
        restarted.add_later_hops(2, verbose=False)
        self.assert_index(restarted, graph)

    def test_local_ranges_resume_only_unwritten_batches(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph, max_hop=2)
        builder.workers = 2
        builder._checkpoint_interval_seconds = 0
        original = builder._write_later_hop_worker_batch
        calls = []
        def fail_second(group, mapping, store, hop):
            calls.append(group[0][0])
            if len(calls) == 2:
                raise OSError("interrupted batch write")
            return original(group, mapping, store, hop)
        with patch.object(builder, "_write_later_hop_worker_batch", side_effect=fail_second):
            with self.assertRaises(OSError):
                builder.add_later_hops(2, batch_size=1, verbose=False)
        path = builder.output_dir / ".batch-checkpoint-hop_2" / "ranges.json"
        state = json.loads(path.read_text())
        self.assertEqual(list(state.values()), [True, False, False, False])
        restarted = MODULE.ExactHopBitmapBuilder(builder.output_dir, max_hop=2, workers=2)
        # A changed caller batch size must not change saved range boundaries.
        with patch.object(restarted, "_write_later_hop_worker_batch",
                          wraps=restarted._write_later_hop_worker_batch) as write, \
             patch.object(restarted, "_run_worker_queue", wraps=restarted._run_worker_queue) as queue:
            restarted.add_later_hops(2, batch_size=3, verbose=False)
        self.assertEqual(write.call_count, 3)
        self.assertNotIn(calls[0], [call.args[0][0][0] for call in write.call_args_list])
        self.assertFalse(path.exists())
        self.assert_index(restarted, graph)

    def test_fast_phase_has_no_explicit_completion_flush(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph, max_hop=2)
        with patch.object(builder, "_flush_batch_checkpoint",
                          wraps=builder._flush_batch_checkpoint) as flush:
            builder.add_later_hops(2, batch_size=1, verbose=False)
        self.assertEqual(flush.call_count, 0)
        self.assert_index(builder, graph)

    def test_five_minute_timer_flushes_only_while_work_remains(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph, max_hop=2)
        saved_states = []
        original = builder._write_batch_checkpoint
        def record(path, state):
            saved_states.append(state.copy())
            return original(path, state)
        with patch.object(MODULE, "monotonic", side_effect=[0, 301, 301, 302, 303, 304]), \
             patch.object(builder, "_write_batch_checkpoint", side_effect=record), \
             patch.object(builder, "_flush_batch_checkpoint",
                          wraps=builder._flush_batch_checkpoint) as flush:
            builder.add_later_hops(2, batch_size=1, verbose=False)
        self.assertEqual(flush.call_count, 1)
        self.assertEqual([sum(state.values()) for state in saved_states], [0, 1])
        self.assert_index(builder, graph)

    def test_unflushed_batches_are_replayed_from_retained_outputs(self):
        graph = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        builder = self.create_builder(graph, max_hop=2)
        original = builder._write_later_hop_worker_batch
        calls = []
        def fail_second(group, mapping, store, hop):
            calls.append(group[0][0])
            if len(calls) == 2:
                raise OSError("before five-minute checkpoint")
            return original(group, mapping, store, hop)
        with patch.object(builder, "_write_later_hop_worker_batch", side_effect=fail_second):
            with self.assertRaises(OSError):
                builder.add_later_hops(2, batch_size=1, verbose=False)
        directory = builder.output_dir / ".batch-checkpoint-hop_2"
        self.assertEqual(list(json.loads((directory / "ranges.json").read_text()).values()),
                         [False, False, False, False])
        self.assertEqual(len(list(directory.glob("output_*.pkl"))), 4)
        restarted = MODULE.ExactHopBitmapBuilder(builder.output_dir, max_hop=2)
        with patch.object(restarted, "_write_later_hop_worker_batch",
                          wraps=restarted._write_later_hop_worker_batch) as write, \
             patch.object(restarted, "_stage_worker_result", side_effect=AssertionError("already staged")):
            restarted.add_later_hops(2, batch_size=1, verbose=False)
        self.assertEqual(write.call_count, 4)
        self.assertFalse(directory.exists())
        self.assert_index(restarted, graph)

    def test_worker_failure_cleans_staged_batches(self):
        builder = self.create_builder({0: {1}, 1: {0}})
        builder.workers = 2
        with builder._open_bitmap_store("w") as store:
            del store[builder._hop_bitmap_key(1, 1)]
        with self.assertRaises(KeyError):
            builder.add_later_hops(2, batch_size=1, verbose=False)
        self.assertEqual(list(builder.output_dir.glob(".roaring-workers-*")), [])
        with MODULE.LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
            self.assertTrue(BitMap.deserialize(mapping[b"later_hop_dirty"]))


if __name__ == "__main__":
    unittest.main()
