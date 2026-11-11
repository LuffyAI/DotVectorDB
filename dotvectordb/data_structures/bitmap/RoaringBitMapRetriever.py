from __future__ import annotations
import asyncio
from contextvars import ContextVar
from contextlib import nullcontext
from functools import wraps
import json
import time
from pathlib import Path
from threading import Condition, Lock
import numpy as np
from pyroaring import BitMap, BitMap64
from .RoaringBitMapConstruction import ExactHopBitmapBuilder, LmdbStore


class ExactHopBitmapRetriever:
    """Read exact-hop Roaring buckets without loading the graph into memory.

    Queries run in worker threads. Concurrent readers share an LMDB environment
    per file with separate read transactions. Subsequent calls see committed
    incremental updates. Relationship IDs come from the saved database IDs.
    Requests beyond max_hop extend the last precomputed frontier using one-hop
    adjacency, without writing additional hop buckets.
    """

    _in_retrieval_worker = ContextVar("in_roaring_retrieval_worker", default=False)


    @staticmethod
    def _execute_retrieval(method, args, kwargs):
        # Create and complete the coroutine in the worker, including all LMDB reads.
        retriever = args[0]
        memory_reader = retriever.load_all_into_memory and method.__name__ != 'refresh'
        if memory_reader:
            with retriever._memory_condition:
                while retriever._memory_refreshing:
                    retriever._memory_condition.wait()
                retriever._memory_readers += 1
        try:
            return asyncio.run(method(*args, **kwargs))
        finally:
            if memory_reader:
                with retriever._memory_condition:
                    retriever._memory_readers -= 1
                    retriever._memory_condition.notify_all()


    @staticmethod
    def _worker_thread(method):
        @wraps(method)
        async def wrapped(*args, **kwargs):
            retriever = args[0]
            context = retriever._in_retrieval_worker
            if context.get():
                return await method(*args, **kwargs)
            token = context.set(True)
            try:
                return await asyncio.to_thread(retriever._execute_retrieval, method, args, kwargs)
            finally:
                context.reset(token)
        return wrapped

    def __init__(self, index_path: str | Path, max_hop: int | None=None,
                 load_all_into_memory: bool = False):
        """Optionally copy both LMDB stores into a read-only in-memory snapshot.

        The default reads only requested keys. Call await refresh() after
        construction updates to patch a full snapshot with changed entries.
        """
        started = time.perf_counter()
        if not isinstance(load_all_into_memory, bool):
            raise TypeError('load_all_into_memory must be a boolean')
        self.load_all_into_memory = load_all_into_memory
        self.index_path = Path(index_path).resolve()
        pointer = Path(f'{self.index_path}.current.json')
        self.metadata = json.loads(pointer.read_text()) if pointer.exists() else {}
        self.max_hop = int(max_hop if max_hop is not None else self.metadata.get('max_hop', 3))
        if self.max_hop < 1:
            raise ValueError('max_hop must be at least 1')
        self.mapping_path = self.index_path / 'entity_ids.lmdb'
        self.bitmap_path = self.index_path / 'bitmaps.lmdb'
        self._memory_stores = {}
        self._memory_versions = {}
        self._memory_condition = Condition()
        self._memory_readers = 0
        self._memory_refreshing = False
        self._refresh_lock = Lock()
        with LmdbStore(self.mapping_path) as mapping, LmdbStore(self.bitmap_path) as store:
            if load_all_into_memory:
                for path, opened in ((self.mapping_path, mapping), (self.bitmap_path, store)):
                    self._memory_stores[path] = {
                        key: value for key, value in opened.transaction.cursor()
                        if not key.startswith(b'\x00memory_')
                    }
                    self._memory_versions[path] = opened.transaction.id()
        self.load_seconds = time.perf_counter() - started

    @_worker_thread
    async def refresh(self) -> int:
        """Patch committed changes into memory after bitmap construction finishes.

        Only keys changed since this snapshot are read. Concurrent queries
        finish against the old snapshot before the patch is applied. Return
        the number of refreshed entries; on-demand mode needs no refresh.
        """
        if not self.load_all_into_memory:
            return 0
        with self._refresh_lock:
            changes = {}
            versions = {}
            for path in (self.mapping_path, self.bitmap_path):
                with LmdbStore(path) as store:
                    versions[path] = store.transaction.id()
                    changed = {}
                    prefix = LmdbStore._change_prefix
                    start = prefix + (self._memory_versions[path] + 1).to_bytes(8, 'big')
                    with store.transaction.cursor() as cursor:
                        if cursor.set_range(start):
                            for key, _ in cursor:
                                if not key.startswith(prefix):
                                    break
                                changed_key = key[len(prefix) + 8:]
                                changed[changed_key] = store.get(changed_key)
                    changes[path] = changed
            with self._memory_condition:
                self._memory_refreshing = True
                try:
                    while self._memory_readers:
                        self._memory_condition.wait()
                    for path, changed in changes.items():
                        cache = self._memory_stores[path]
                        for key, value in changed.items():
                            if value is None:
                                cache.pop(key, None)
                            else:
                                cache[key] = value
                    self._memory_versions.update(versions)
                finally:
                    self._memory_refreshing = False
                    self._memory_condition.notify_all()
            return sum(map(len, changes.values()))

    async def _open_store(self, path):
        if self.load_all_into_memory:
            return nullcontext(self._memory_stores[path])
        return LmdbStore(path)

    @staticmethod
    async def _normalise_types(entity_types):
        if entity_types is None:
            return ()
        if isinstance(entity_types, str):
            return (entity_types,)
        return tuple(dict.fromkeys((str(kind) for kind in entity_types)))

    @staticmethod
    async def _read_bitmap(store, key):
        saved = store.get(key)
        return BitMap.deserialize(saved) if saved is not None else BitMap()

    @_worker_thread
    async def entity_types(self) -> tuple[str, ...]:
        with await self._open_store(self.bitmap_path) as store:
            return tuple(json.loads(store.get(b'type_names', b'[]')))

    async def _hop_bitmap(self, anchor_entity_id, hop, entity_types=None, match_all_types=False):
        if isinstance(hop, bool) or not isinstance(hop, int) or hop < 1:
            raise ValueError('hop must be a positive integer')
        kinds = await self._normalise_types(entity_types)
        with await self._open_store(self.mapping_path) as mapping, await self._open_store(self.bitmap_path) as store:
            row = int(mapping[f'id:{int(anchor_entity_id)}'.encode()])
            if hop > self.max_hop:
                return await OneHopBitmapRetriever._remaining_hop_bitmap(
                    row, hop, self.max_hop, kinds, match_all_types, store)
            key = ExactHopBitmapBuilder._hop_bitmap_key(row, hop)
            if key not in store:
                raise ValueError(f'Hop {hop} has not been indexed for entity {anchor_entity_id}')
            if not kinds:
                return BitMap.deserialize(store[key])
            result = None
            for kind in kinds:
                bucket = await self._read_bitmap(store, ExactHopBitmapBuilder._hop_bitmap_key(row, hop, kind))
                if result is None:
                    result = bucket
                elif match_all_types:
                    result &= bucket
                else:
                    result |= bucket
            return result

    @_worker_thread
    async def dense_ids_at_hop(self, anchor_entity_id, hop) -> np.ndarray:
        return np.fromiter(await self._hop_bitmap(anchor_entity_id, hop), dtype=np.int64)

    @_worker_thread
    async def constrained_entity_search(self, anchor_entity_id, hop, entity_types=None, match_all_types=False) -> np.ndarray:
        rows = await self._hop_bitmap(anchor_entity_id, hop, entity_types, match_all_types)
        with await self._open_store(self.mapping_path) as mapping:
            return np.fromiter((int(mapping[f'row:{row}'.encode()]) for row in rows), dtype=np.int64)

    @_worker_thread
    async def entity_id_set_at_hop(self, anchor_entity_id, hop, entity_types=None, match_all_types=False) -> set[int]:
        return set(map(int, await self.constrained_entity_search(anchor_entity_id, hop, entity_types, match_all_types)))

    @_worker_thread
    async def entity_types_for_id(self, entity_id) -> tuple[str, ...]:
        with await self._open_store(self.mapping_path) as mapping:
            row = int(mapping[f'id:{int(entity_id)}'.encode()])
            return tuple(json.loads(mapping[f'types:{row}'.encode()]))

    @_worker_thread
    async def entity_type_groups_by_id(self, entity_ids) -> dict[int, tuple[str, ...]]:
        groups = {}
        with await self._open_store(self.mapping_path) as mapping:
            for entity_id in dict.fromkeys(map(int, entity_ids)):
                row = mapping.get(f'id:{entity_id}'.encode())
                if row is not None:
                    groups[entity_id] = tuple(json.loads(mapping[f'types:{int(row)}'.encode()]))
        return groups

    @_worker_thread
    async def type_group_counts_for_ids(self, entity_ids) -> dict[tuple[str, ...], int]:
        entity_ids = list(map(int, entity_ids))
        groups = await self.entity_type_groups_by_id(entity_ids)
        counts = {}
        for entity_id in entity_ids:
            group = groups.get(entity_id, ())
            if group:
                counts[group] = counts.get(group, 0) + 1
        return counts

    @_worker_thread
    async def entity_ids_for_types(self, entity_types, match_all_types=False) -> np.ndarray:
        kinds = await self._normalise_types(entity_types)
        with await self._open_store(self.mapping_path) as mapping, await self._open_store(self.bitmap_path) as store:
            if not kinds:
                rows = range(int(mapping.get(b'next_row', b'0')))
            else:
                rows = None
                for kind in kinds:
                    bucket = await self._read_bitmap(store, ExactHopBitmapBuilder._type_bitmap_key(kind))
                    if rows is None:
                        rows = bucket
                    elif match_all_types:
                        rows &= bucket
                    else:
                        rows |= bucket
            return np.fromiter((int(mapping[f'row:{row}'.encode()]) for row in rows), dtype=np.int64)

    @_worker_thread
    async def type_counts(self, anchor_entity_id, hop) -> dict[str, int]:
        return {kind: len(await self._hop_bitmap(anchor_entity_id, hop, kind)) for kind in await self.entity_types()}

    @_worker_thread
    async def reachable_types(self, anchor_entity_id, hop) -> dict[str, int]:
        return {kind: count for (kind, count) in (await self.type_counts(anchor_entity_id, hop)).items() if count}

    @_worker_thread
    async def hop_overview(self, anchor_entity_id) -> dict[int, dict[str, int]]:
        return {hop: await self.reachable_types(anchor_entity_id, hop) for hop in range(1, self.max_hop + 1)}

    @_worker_thread
    async def adjacent_type_groups(self, frontier_entity_ids, exclude_entity_ids=None) -> list[tuple[str, ...]]:
        rows = BitMap()
        for entity_id in frontier_entity_ids:
            rows |= await self._hop_bitmap(entity_id, 1)
        with await self._open_store(self.mapping_path) as mapping:
            for entity_id in () if exclude_entity_ids is None else exclude_entity_ids:
                row = mapping.get(f'id:{int(entity_id)}'.encode())
                if row is not None:
                    rows.discard(int(row))
            groups = {tuple(json.loads(mapping[f'types:{row}'.encode()])) for row in rows}
        return sorted((group for group in groups if group))

    @_worker_thread
    async def constrained_relationship_search(self, anchor_entity_id, exact_hop,
                                              entity_types=None, match_all_types=False) -> list[int]:
        """Read the precomputed relationship-ID bucket for this anchor/hop/type."""
        if isinstance(exact_hop, bool) or not isinstance(exact_hop, int) or exact_hop < 1:
            raise ValueError('exact_hop must be a positive integer')
        kinds = await self._normalise_types(entity_types)
        if exact_hop > self.max_hop:
            return await OneHopBitmapRetriever._remaining_relationship_search(
                self, anchor_entity_id, exact_hop, kinds, match_all_types)
        with await self._open_store(self.mapping_path) as mapping, await self._open_store(self.bitmap_path) as store:
            row = int(mapping[f'id:{int(anchor_entity_id)}'.encode()])
            key = ExactHopBitmapBuilder._relationship_bitmap_key(row, exact_hop)
            if key not in store:
                raise ValueError(f'Relationship hop {exact_hop} has not been indexed for entity {anchor_entity_id}')
            if not kinds:
                result = BitMap64.deserialize(store[key])
            else:
                result = None
                for kind in kinds:
                    saved = store.get(ExactHopBitmapBuilder._relationship_bitmap_key(row, exact_hop, kind))
                    bucket = BitMap64.deserialize(saved) if saved is not None else BitMap64()
                    if result is None:
                        result = bucket
                    elif match_all_types:
                        result &= bucket
                    else:
                        result |= bucket
            return [value if value < (1 << 63) else value - (1 << 64) for value in result]

class OneHopBitmapRetriever(ExactHopBitmapRetriever):
    """Read one-hop Roaring adjacency and entity type buckets."""

    def __init__(self, index_path: str | Path, load_all_into_memory: bool = False):
        super().__init__(index_path, max_hop=1, load_all_into_memory=load_all_into_memory)

    @staticmethod
    async def _expand_frontier(store, frontier):
        """Union the saved one-hop neighbors of a frontier."""
        neighbors = BitMap()
        for row in frontier:
            neighbors |= BitMap.deserialize(store[ExactHopBitmapBuilder._hop_bitmap_key(row, 1)])
        return neighbors

    @ExactHopBitmapRetriever._worker_thread
    async def relationships_from_frontier(self, frontier_entity_ids, entity_types=None, match_all_types=False, exclude_entity_ids=None):
        return await OneHopBitmapRetriever._relationships_from_frontier(self, frontier_entity_ids, entity_types, match_all_types, exclude_entity_ids)

    @ExactHopBitmapRetriever._worker_thread
    async def expansions_from_paths(self, paths, entity_types=None, match_all_types=False):
        expansions = []
        for path in paths:
            entity_ids = tuple(map(int, path.get('entity_ids', ())))
            if not entity_ids:
                continue
            relationships = await self.relationships_from_frontier([entity_ids[-1]], entity_types, match_all_types, entity_ids)
            for (relation_id, (source, destination)) in relationships.items():
                expansions.append({'relationship_id': relation_id, 'frontier_entity_id': source, 'endpoint_entity_id': destination, 'prefix': path})
        return expansions

    async def _relationships_from_frontier(self, frontier_entity_ids, entity_types=None, match_all_types=False, exclude_entity_ids=None, allowed_rows=None):
        excluded = set(map(int, () if exclude_entity_ids is None else exclude_entity_ids))
        relationships = {}
        for source in map(int, frontier_entity_ids):
            neighbors = await self._hop_bitmap(source, 1, entity_types, match_all_types)
            if allowed_rows is not None:
                neighbors &= allowed_rows
            with await self._open_store(self.mapping_path) as mapping:
                source_row = int(mapping[f'id:{source}'.encode()])
                for row in neighbors:
                    destination = int(mapping[f'row:{row}'.encode()])
                    if destination in excluded:
                        continue
                    relation_id = int(mapping[ExactHopBitmapBuilder._relationship_key(source_row, row)])
                    relationships[relation_id] = (source, destination)
        return relationships


    @staticmethod
    async def _remaining_hop_bitmap(row, hop, max_hop, kinds, match_all_types, store):
        reached = BitMap([row])
        for saved_hop in range(1, max_hop + 1):
            frontier = BitMap.deserialize(store[ExactHopBitmapBuilder._hop_bitmap_key(row, saved_hop)])
            reached |= frontier
        for _ in range(max_hop, hop):
            frontier = await OneHopBitmapRetriever._expand_frontier(store, frontier) - reached
            reached |= frontier
            if not frontier:
                break
        if kinds:
            allowed = None
            for kind in kinds:
                saved = store.get(ExactHopBitmapBuilder._type_bitmap_key(kind))
                bucket = BitMap.deserialize(saved) if saved is not None else BitMap()
                if allowed is None:
                    allowed = bucket
                elif match_all_types:
                    allowed &= bucket
                else:
                    allowed |= bucket
            frontier &= allowed
        return frontier

    @staticmethod
    async def _remaining_relationship_search(retriever, anchor_entity_id, hop, kinds, match_all_types):
        # Only uncomputed hops need a frontier expansion.
        allowed = await retriever._hop_bitmap(anchor_entity_id, hop, kinds, match_all_types)
        frontier = await retriever.constrained_entity_search(anchor_entity_id, hop - 1)
        relationships = await OneHopBitmapRetriever._relationships_from_frontier(
            retriever, frontier, allowed_rows=allowed)
        return list(relationships)
