"""Read a constructed index, including updates committed after opening retrieval."""
import tempfile
import asyncio
import threading
import unittest
from unittest.mock import patch
from collections import deque
from pyroaring import BitMap64
from dotvectordb.data_structures.bitmap.RoaringBitMapConstruction import ExactHopBitmapBuilder, LmdbStore
from dotvectordb.data_structures.bitmap.RoaringBitMapRetriever import ExactHopBitmapRetriever, OneHopBitmapRetriever

class RoaringRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_refresh_waits_for_active_query_then_publishes_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=1)
            builder.add_new_entity_ids(2)
            builder.add_entity_types([(10, ['Person']), (20, ['Person'])])
            builder.add_new_1_hop_relationships([(10, [20], 1), (20, [10], 1)], relationships=[
                {'relation_id': 123, 'src_entity_id': 10, 'dst_entity_id': 20}])
            memory = ExactHopBitmapRetriever(directory, max_hop=1, load_all_into_memory=True)
            key = ExactHopBitmapBuilder._relationship_bitmap_key(0, 1, 'Person')
            with LmdbStore(builder.bitmap_store_path, 'w') as store:
                store[key] = BitMap64([123, 456]).serialize()
            entered, release = threading.Event(), threading.Event()
            original = memory._normalise_types

            async def hold_first_query(types):
                if not entered.is_set():
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError('Query was not released')
                return await original(types)

            async def wait_until_refresh_is_pending():
                while True:
                    with memory._memory_condition:
                        if memory._memory_refreshing:
                            return
                    await asyncio.sleep(0.001)

            with patch.object(memory, '_normalise_types', hold_first_query):
                old_query = asyncio.create_task(memory.constrained_relationship_search(10, 1, 'Person'))
                refresh = new_query = None
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    refresh = asyncio.create_task(memory.refresh())
                    await asyncio.wait_for(wait_until_refresh_is_pending(), 3)
                    new_query = asyncio.create_task(memory.constrained_relationship_search(10, 1, 'Person'))
                    await asyncio.sleep(0.01)
                    self.assertFalse(refresh.done())
                    self.assertFalse(new_query.done())
                finally:
                    release.set()
                    old_ids = await old_query
                    if refresh is not None:
                        await refresh
                    if new_query is not None:
                        new_ids = await new_query
                self.assertEqual(old_ids, [123])
                self.assertEqual(new_ids, [123, 456])

    async def test_memory_refresh_reads_only_changes_and_removes_deleted_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=2)
            builder.add_new_entity_ids(3)
            builder.add_entity_types([(10, ['Person']), (20, ['Person']), (30, ['Person'])])
            builder.add_new_1_hop_relationships([(10, [20], 1), (20, [10, 30], 2), (30, [20], 1)], relationships=[
                {'relation_id': 1, 'src_entity_id': 10, 'dst_entity_id': 20},
                {'relation_id': 2, 'src_entity_id': 20, 'dst_entity_id': 30}])
            builder.add_later_hops(100, verbose=False)
            memory = ExactHopBitmapRetriever(directory, max_hop=2, load_all_into_memory=True)
            unchanged_key = ExactHopBitmapBuilder._relationship_bitmap_key(0, 1, 'Person')
            unchanged_value = memory._memory_stores[builder.bitmap_store_path][unchanged_key]
            self.assertEqual(await memory.refresh(), 0)
            builder.add_new_entity_ids(1)
            builder.add_entity_types([(40, ['Person'])])
            builder.add_new_1_hop_relationships([(20, [40], 3), (40, [20], 1)], relationships=[
                {'relation_id': 3, 'src_entity_id': 20, 'dst_entity_id': 40}])
            builder.add_later_hops(100, modified_entity_ids=[20, 40], verbose=False)
            self.assertEqual(await memory.constrained_relationship_search(10, 2, 'Person'), [2])
            original = LmdbStore.get
            read_keys = []
            def tracked_get(store, key, default=None):
                read_keys.append(key)
                return original(store, key, default)
            with patch.object(LmdbStore, 'get', tracked_get):
                refreshed = await memory.refresh()
            self.assertGreater(refreshed, 0)
            self.assertEqual(refreshed, len(read_keys))
            self.assertNotIn(unchanged_key, read_keys)
            self.assertIs(memory._memory_stores[builder.bitmap_store_path][unchanged_key], unchanged_value)
            self.assertEqual(await memory.constrained_relationship_search(10, 2, 'Person'), [2, 3])
            self.assertEqual(await memory.refresh(), 0)
            deleted_key = ExactHopBitmapBuilder._relationship_bitmap_key(0, 2, 'Person')
            with LmdbStore(builder.bitmap_store_path, 'w') as store:
                del store[deleted_key]
            self.assertEqual(await memory.refresh(), 1)
            self.assertEqual(await memory.constrained_relationship_search(10, 2, 'Person'), [])
            self.assertEqual(await ExactHopBitmapRetriever(directory).refresh(), 0)

    async def test_full_memory_mode_matches_on_demand_without_query_disk_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=2)
            builder.add_new_entity_ids(4)
            builder.add_entity_types([(10, ['Person']), (20, ['Organization']),
                                      (30, ['Person', 'Actor']), (40, ['Person'])])
            builder.add_new_1_hop_relationships([(10, [20], 1), (20, [10, 30], 2),
                                                (30, [20, 40], 2), (40, [30], 1)], relationships=[
                {'relation_id': -1, 'src_entity_id': 10, 'dst_entity_id': 20},
                {'relation_id': 2, 'src_entity_id': 20, 'dst_entity_id': 30},
                {'relation_id': 3, 'src_entity_id': 30, 'dst_entity_id': 40}])
            builder.add_later_hops(100, verbose=False)
            disk = ExactHopBitmapRetriever(directory, max_hop=2)
            memory = ExactHopBitmapRetriever(directory, max_hop=2, load_all_into_memory=True)
            one_hop = OneHopBitmapRetriever(directory, load_all_into_memory=True)
            expected = [await disk.constrained_relationship_search(10, hop, 'Person') for hop in (1, 2, 3)]
            with patch('dotvectordb.data_structures.bitmap.RoaringBitMapRetriever.LmdbStore',
                       side_effect=AssertionError('Memory queries must not open LMDB')):
                for hop, relationship_ids in zip((1, 2, 3), expected):
                    self.assertEqual(await memory.constrained_relationship_search(10, hop, 'Person'), relationship_ids)
                self.assertEqual(list(await memory.constrained_entity_search(10, 2, 'Actor')), [30])
                self.assertEqual(await memory.constrained_relationship_search(10, 2, ['Person', 'Actor'], True), [2])
                self.assertEqual(await one_hop.relationships_from_frontier([30], 'Person'), {3: (30, 40)})
                results = await asyncio.gather(*[memory.constrained_relationship_search(10, 2, 'Person') for _ in range(50)])
                self.assertEqual(results, [[2]] * 50)

    async def test_concurrent_readers_share_environment_without_blocking_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=1)
            builder.add_new_entity_ids(2)
            builder.add_entity_types([(10, ['Person']), (20, ['Person'])])
            builder.add_new_1_hop_relationships([(10, [20], 1), (20, [10], 1)], relationships=[
                {'relation_id': -123, 'src_entity_id': 10, 'dst_entity_id': 20}])
            retrievers = [ExactHopBitmapRetriever(directory, max_hop=1) for _ in range(4)]
            entered = threading.Event()
            release = threading.Event()
            lock = threading.Lock()
            records = []
            original = LmdbStore.__getitem__
            loop_thread = threading.get_ident()

            def blocked_read(store, key):
                if key == b'id:10':
                    with lock:
                        records.append((threading.get_ident(), id(store.environment), id(store.transaction)))
                        if len(records) == 4:
                            entered.set()
                    if not release.wait(5):
                        raise TimeoutError('Concurrent readers did not get released')
                return original(store, key)

            with patch.object(LmdbStore, '__getitem__', blocked_read):
                tasks = [asyncio.create_task(r.constrained_relationship_search(10, 1, 'Person')) for r in retrievers]
                try:
                    self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4))
                    # This coroutine continues while all four query threads are blocked.
                    await asyncio.sleep(0.01)
                    self.assertTrue(all(not task.done() for task in tasks))
                    self.assertEqual(len({record[0] for record in records}), 4)
                    self.assertNotIn(loop_thread, {record[0] for record in records})
                    self.assertEqual(len({record[1] for record in records}), 1)
                    self.assertEqual(len({record[2] for record in records}), 4)
                finally:
                    release.set()
                    results = await asyncio.gather(*tasks)
                self.assertEqual(results, [[-123]] * 4)
            results = await asyncio.gather(*[
                retrievers[i % 4].constrained_relationship_search(10, 1, 'Person') for i in range(80)])
            self.assertEqual(results, [[-123]] * 80)
            self.assertNotIn(str(builder.entity_mapping_path), LmdbStore._environments)
            self.assertNotIn(str(builder.bitmap_store_path), LmdbStore._environments)

    async def test_precomputed_relationship_search_never_traverses(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=3)
            builder.add_new_entity_ids(4)
            builder.add_entity_types([(10, ['Person']), (20, ['Organization']),
                                      (30, ['Person', 'Actor']), (40, ['Person'])])
            relationships = [{'relation_id': relation_id, 'src_entity_id': source, 'dst_entity_id': target}
                             for relation_id, source, target in [(-1, 10, 20), (2, 20, 30), (3, 30, 40)]]
            builder.add_new_1_hop_relationships([(10, [20], 1), (20, [10, 30], 2),
                                                (30, [20, 40], 2), (40, [30], 1)], relationships=relationships)
            builder.add_later_hops(100, verbose=False)
            retriever = ExactHopBitmapRetriever(directory, max_hop=3)
            with patch.object(retriever, '_hop_bitmap', side_effect=AssertionError('Endpoint traversal is forbidden')), \
                 patch.object(OneHopBitmapRetriever, '_expand_frontier', side_effect=AssertionError('Expansion is forbidden')), \
                 patch.object(OneHopBitmapRetriever, '_relationships_from_frontier', side_effect=AssertionError('Frontier scan is forbidden')):
                self.assertEqual(await retriever.constrained_relationship_search(10, 1, 'Organization'), [-1])
                self.assertEqual(await retriever.constrained_relationship_search(10, 2, 'Person'), [2])
                self.assertEqual(await retriever.constrained_relationship_search(10, 3, 'Person'), [3])
                self.assertEqual(await retriever.constrained_relationship_search(10, 2, ['Person', 'Actor'], True), [2])
                self.assertEqual(await retriever.constrained_relationship_search(10, 3, 'Actor'), [])
            builder.add_new_entity_ids(1)
            builder.add_entity_types([(50, ['Person'])])
            builder.add_new_1_hop_relationships([(20, [50], 3), (50, [20], 1)], relationships=[
                {'relation_id': 4, 'src_entity_id': 20, 'dst_entity_id': 50}])
            builder.add_later_hops(100, modified_entity_ids=[20, 50], verbose=False)
            self.assertEqual(await retriever.constrained_relationship_search(10, 2, 'Person'), [2, 4])


    async def test_expand_beyond_precomputed_hops(self):
        graph = {10: [20, 30], 20: [10, 30], 30: [10, 20, 40], 40: [30, 50], 50: [40, 60], 60: [50], 70: []}
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=2)
            builder.add_new_entity_ids(len(graph))
            builder.add_entity_types(((node, ['Target'] if node in (50, 60) else ['Other']) for node in graph))
            edges = [{'relation_id': source * 100 + destination, 'src_entity_id': source, 'dst_entity_id': destination} for (source, neighbors) in graph.items() for destination in neighbors if source < destination]
            builder.add_new_1_hop_relationships(((node, neighbors, len(neighbors)) for (node, neighbors) in graph.items()), relationships=edges)
            builder.add_later_hops(100, verbose=False)
            retriever = ExactHopBitmapRetriever(directory, max_hop=2)
            for anchor in graph:
                distances = {anchor: 0}
                queue = deque([anchor])
                while queue:
                    source = queue.popleft()
                    for destination in graph[source]:
                        if destination not in distances:
                            distances[destination] = distances[source] + 1
                            queue.append(destination)
                for hop in range(1, 8):
                    expected = {node for (node, distance) in distances.items() if distance == hop}
                    self.assertEqual(await retriever.entity_id_set_at_hop(anchor, hop), expected)
            self.assertEqual(await retriever.entity_id_set_at_hop(10, 3, 'Target'), {50})
            self.assertEqual(await retriever.entity_id_set_at_hop(10, 3, ['Target', 'Other'], True), set())
            self.assertEqual(await retriever.constrained_relationship_search(10, 3), [4050])
            self.assertEqual(await OneHopBitmapRetriever(directory).entity_id_set_at_hop(10, 4), {60})

    async def test_typed_exact_hops_and_incremental_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ExactHopBitmapBuilder(directory, max_hop=3)
            builder.add_new_entity_ids(4)
            builder.add_entity_types([(100, ['Person']), (200, ['Organization', 'Company']), (300, ['Organization']), (400, ['Person'])])
            builder.add_new_1_hop_relationships([(100, [200, 300], 2), (200, [100, 400], 2), (300, [100], 1), (400, [200], 1)], relationships=[{'relation_id': -9001, 'src_entity_id': 100, 'dst_entity_id': 200}, {'relation_id': 9002, 'src_entity_id': 100, 'dst_entity_id': 300}, {'relation_id': 9003, 'src_entity_id': 200, 'dst_entity_id': 400}])
            builder.add_later_hops(100, verbose=False)
            retriever = ExactHopBitmapRetriever(directory)
            self.assertEqual(await retriever.entity_id_set_at_hop(100, 1), {200, 300})
            self.assertEqual(await retriever.entity_id_set_at_hop(100, 2), {400})
            self.assertEqual(await retriever.entity_id_set_at_hop(100, 3), set())
            one_hop = OneHopBitmapRetriever(directory)
            self.assertEqual(await one_hop.relationships_from_frontier([100], 'Company'), {-9001: (100, 200)})
            self.assertEqual(await one_hop.relationships_from_frontier([400]), {9003: (400, 200)})
            self.assertEqual(await retriever.constrained_relationship_search(100, 2), [9003])
            self.assertEqual(len(await one_hop.expansions_from_paths([{'entity_ids': [200, 100]}])), 1)
            self.assertEqual(await retriever.entity_id_set_at_hop(100, 1, ['Organization', 'Company'], True), {200})
            self.assertEqual(await retriever.entity_id_set_at_hop(100, 1, 'missing'), set())
            self.assertEqual(set(await retriever.entity_ids_for_types(['Organization', 'Company'], True)), {200})
            self.assertEqual(await retriever.reachable_types(100, 1), {'Company': 1, 'Organization': 2})
            self.assertEqual(await retriever.entity_types_for_id(200), ('Company', 'Organization'))
            self.assertEqual(await retriever.adjacent_type_groups([100], [200]), [('Organization',)])
            with self.assertRaises(KeyError):
                await retriever.constrained_entity_search(999, 1)
            with self.assertRaises(ValueError):
                await retriever.constrained_entity_search(100, 0)
            builder.add_new_entity_ids(1)
            builder.add_entity_types([(500, ['Person'])])
            builder.add_new_1_hop_relationships([(100, [500], 3), (500, [100], 1)])
            self.assertEqual(await retriever.entity_id_set_at_hop(100, 1, 'Person'), {500})
            self.assertEqual(set(await retriever.entity_ids_for_types(None)), {100, 200, 300, 400, 500})
if __name__ == '__main__':
    unittest.main()
