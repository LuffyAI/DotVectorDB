from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
import multiprocessing
import lmdb
import json
from itertools import chain, islice
import os
import pickle
import shutil
from time import monotonic
from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Condition

from pyroaring import BitMap, BitMap64
from tqdm import tqdm


class LmdbStore:
    """Share each open environment while giving every caller its own transaction."""

    _environments = {}
    _environment_condition = Condition()
    _revision_prefix = b"\x00memory_revision:"
    _change_prefix = b"\x00memory_change:"

    def __init__(self, path, flag="r"):
        self.readonly = flag == "r"
        self._path = str(Path(path).resolve())
        with self._environment_condition:
            # A read-only environment cannot create write transactions. Wait
            # for its readers to finish before reopening it for a writer.
            while (entry := self._environments.get(self._path)) is not None and entry[2] and not self.readonly:
                self._environment_condition.wait()
            if entry is None:
                environment = lmdb.open(
                    self._path, subdir=False, readonly=self.readonly, create=flag == "c",
                    map_size=1 << 40, max_spare_txns=0,
                )
                entry = [environment, 0, self.readonly]
                self._environments[self._path] = entry
            entry[1] += 1
            self.environment = entry[0]
        try:
            self.transaction = self.environment.begin(write=not self.readonly)
        except BaseException:
            self._release_environment()
            raise

    def _release_environment(self):
        with self._environment_condition:
            entry = self._environments[self._path]
            entry[1] -= 1
            if entry[1] == 0:
                self.environment.close()
                del self._environments[self._path]
                self._environment_condition.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None and not self.readonly:
                self.transaction.commit()
            else:
                self.transaction.abort()
        finally:
            self._release_environment()

    def get(self, key, default=None):
        value = self.transaction.get(key)
        return default if value is None else value

    def __getitem__(self, key):
        value = self.transaction.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        if self.readonly:
            raise lmdb.ReadonlyError("Cannot write to a read-only transaction")
        if self.transaction.get(key) == value:
            return
        self.transaction.put(key, value)
        self._record_change(key)

    def __delitem__(self, key):
        if not self.transaction.delete(key):
            raise KeyError(key)
        self._record_change(key)

    def _record_change(self, key):
        """Keep one latest revision per key, committed with the changed value.

        The revision-ordered index lets memory snapshots seek directly to new
        changes. Rewriting a key removes its old index entry, retaining no
        history of its intermediate values.
        """
        revision = self.transaction.id().to_bytes(8, "big")
        revision_key = self._revision_prefix + key
        previous = self.transaction.get(revision_key)
        if previous == revision:
            return
        if previous is not None:
            self.transaction.delete(self._change_prefix + previous + key)
        self.transaction.put(revision_key, revision)
        self.transaction.put(self._change_prefix + revision + key, b"")

    def __contains__(self, key):
        return self.transaction.get(key) is not None

    def sync(self):
        """Commit durably before publishing a batch checkpoint."""
        self.transaction.commit()
        self.transaction = self.environment.begin(write=True)


class ExactHopBitmapBuilder:
    """Build incremental exact-hop Roaring indexes with direct type buckets.

    ``bitmaps.lmdb`` stores hop rows and type membership. ``entity_ids.lmdb``
    stores ID mappings, degrees, estimates, reverse connections, and pending
    updates. Small JSON files persist the last allocated row and entity stats.
    Process workers take full batch_size tasks from a shared queue. Completed
    results are staged before the parent commits them, avoiding conflicting
    LMDB readers and writers. Local range checkpoints retain unfinished batches
    across restarts and are removed after successful completion.
    This builder does not create NumPy arrays or dense bitmap sidecars.
    """

    _checkpoint_interval_seconds = 300

    @staticmethod
    def _compute_one_hop_rows(task):
        """Compute rows using read-only stores; return before the parent writes."""
        mapping_path, bitmap_path, rows = task
        results = []
        with LmdbStore(mapping_path, "r") as mapping, LmdbStore(bitmap_path, "r") as store:
            for row, neighbor_ids, _ in rows:
                neighbors = BitMap(int(mapping[f"id:{entity_id}".encode()]) for entity_id in neighbor_ids)
                neighbors.discard(row)
                key = ExactHopBitmapBuilder._hop_bitmap_key(row, 1)
                saved = store.get(key)
                previous = BitMap.deserialize(saved) if saved is not None else BitMap()
                updated = previous | neighbors
                added = neighbors - previous
                estimate_key = f"next_hop_estimate:{row}".encode()
                saved_estimate = mapping.get(estimate_key)
                estimate = int(saved_estimate) if saved_estimate is not None else 0
                estimate += sum(int(mapping[f"degree:{other}".encode()])
                                for other in (added if saved_estimate is not None else updated))
                typed = {}
                for other in neighbors:
                    for kind in json.loads(mapping.get(f"types:{other}".encode(), b"[]")):
                        typed.setdefault(kind, BitMap()).add(other)
                buckets = {}
                for kind, members in typed.items():
                    saved = store.get(ExactHopBitmapBuilder._hop_bitmap_key(row, 1, kind))
                    bucket = BitMap.deserialize(saved) if saved is not None else BitMap()
                    buckets[kind] = (bucket | members).serialize()
                relationships = ExactHopBitmapBuilder._compute_relationship_buckets(
                    (row,), updated, mapping, store, direct=True)
                results.append((row, updated.serialize(), buckets, estimate, added.serialize(), relationships))
        return results


    @staticmethod
    def _compute_later_hop_rows(task):
        """Expand independent anchors without holding a writer lock or full graph."""
        mapping_path, bitmap_path, rows, hop, high_degree_limit, batch_size = task
        results = []
        with LmdbStore(mapping_path, "r") as mapping, LmdbStore(bitmap_path, "r") as store:
            for row, projected_work, degree in rows:
                previous = BitMap.deserialize(store[ExactHopBitmapBuilder._hop_bitmap_key(row, hop - 1)])
                excluded = previous | BitMap([row])
                for shorter in range(1, hop - 1):
                    excluded |= BitMap.deserialize(store[ExactHopBitmapBuilder._hop_bitmap_key(row, shorter)])
                result = BitMap()
                batched = degree > high_degree_limit or projected_work > high_degree_limit
                for other in previous:
                    neighbors = BitMap.deserialize(store[ExactHopBitmapBuilder._hop_bitmap_key(other, 1)])
                    if batched or int(mapping[f"degree:{other}".encode()]) > high_degree_limit:
                        for start in range(0, len(neighbors), batch_size):
                            result |= neighbors[start:start + batch_size] - excluded
                    else:
                        result |= neighbors - excluded
                estimate = sum(int(mapping[f"degree:{other}".encode()]) for other in result)
                relationships = ExactHopBitmapBuilder._compute_relationship_buckets(
                    previous, result, mapping, store)
                results.append((row, result.serialize(), estimate, relationships))
        return results

    @staticmethod
    def _compute_relationship_buckets(frontier, endpoints, mapping, store, direct=False):
        """Resolve database relationship IDs during construction, never at search."""
        buckets = {None: BitMap64()}
        for source in frontier:
            neighbors = endpoints if direct else (
                BitMap.deserialize(store[ExactHopBitmapBuilder._hop_bitmap_key(source, 1)]) & endpoints)
            for destination in neighbors:
                saved = mapping.get(ExactHopBitmapBuilder._relationship_key(source, destination))
                if saved is None:
                    # Adjacency-only construction has no supplied relationship IDs.
                    continue
                relation_id = int(saved) & ((1 << 64) - 1)
                buckets[None].add(relation_id)
                for kind in json.loads(mapping[f"types:{destination}".encode()]):
                    buckets.setdefault(kind, BitMap64()).add(relation_id)
        return {kind: bitmap.serialize() for kind, bitmap in buckets.items()}

    def _save_relationship_buckets(self, row, hop, buckets, store):
        """Replace the saved relation buckets, including emptied type buckets."""
        for kind in [None, *json.loads(store.get(b"type_names", b"[]"))]:
            key = self._relationship_bitmap_key(row, hop, kind)
            store[key] = buckets.get(kind, BitMap64().serialize())


    @staticmethod
    def _stage_worker_result(job):
        """Keep completed payloads off the parent heap while readers keep working."""
        operation, task, destination = job
        if isinstance(task, str):
            with open(task, "rb") as source:
                task = pickle.load(source)
        rows = (ExactHopBitmapBuilder._compute_one_hop_rows(task) if operation == "one"
                else ExactHopBitmapBuilder._compute_later_hop_rows(task))
        temporary = destination + ".tmp"
        with open(temporary, "wb") as output:
            pickle.dump(rows, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        return len(rows)

    def __init__(
        self,
        output_dir: str | Path,
        max_hop: int = 3,
        workers: int = 1,
    ):
        if isinstance(workers, bool) or not isinstance(workers, int):
            raise TypeError("workers must be an integer")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self.output_dir = Path(output_dir).resolve()
        self.max_hop = int(max_hop)
        if self.max_hop < 1:
            raise ValueError("max_hop must be at least 1")

        self.last_entity_row = -1
        self.entity_row_state_path = self.output_dir / "entity_row_state.json"
        if self.entity_row_state_path.exists():
            row_state = json.loads(self.entity_row_state_path.read_text(encoding="utf-8"))
            self.last_entity_row = int(row_state["last_entity_row"])
        self.entity_mapping_path = self.output_dir / "entity_ids.lmdb"
        self.bitmap_store_path = self.output_dir / "bitmaps.lmdb"

    def add_new_entity_ids(self, new_entity_count: int) -> range:
        """Append empty entity rows and return their zero-based row indices."""
        if isinstance(new_entity_count, bool) or not isinstance(new_entity_count, int):
            raise TypeError("new_entity_count must be an integer")
        if new_entity_count < 0:
            raise ValueError("new_entity_count cannot be negative")

        first_new_row = self.last_entity_row + 1
        last_entity_row = self.last_entity_row + new_entity_count
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.entity_row_state_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as state_file:
            json.dump(
                {
                    "last_entity_row": last_entity_row,
                },
                state_file,
            )
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temporary_path, self.entity_row_state_path)
        self.last_entity_row = last_entity_row
        return range(first_new_row, self.last_entity_row + 1)

    def add_entity_types(self, entity_type_batch) -> dict:
        """Assign supplied types to reserved rows without loading the full index."""
        assignments = [
            (int(entity_id), self._normalise_types(types))
            for entity_id, types in entity_type_batch
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pending_types = {}
        pending_neighbor_types = []
        changed_rows = BitMap()
        with LmdbStore(str(self.entity_mapping_path), "c") as mapping:
            next_row = int(mapping.get(b"next_row", b"0"))
            new_ids = list(dict.fromkeys(
                entity_id for entity_id, _ in assignments
                if f"id:{entity_id}".encode() not in mapping
            ))
            if next_row + len(new_ids) > self.last_entity_row + 1:
                raise ValueError("Allocate enough rows with add_new_entity_ids first")
            for entity_id in new_ids:
                changed_rows.add(next_row)
                mapping[f"id:{entity_id}".encode()] = str(next_row).encode()
                mapping[f"row:{next_row}".encode()] = str(entity_id).encode()
                next_row += 1
            mapping[b"next_row"] = str(next_row).encode()
            for entity_id, entity_types in assignments:
                row = int(mapping[f"id:{entity_id}".encode()])
                types_key = f"types:{row}".encode()
                saved_types = set(json.loads(mapping.get(types_key, b"[]")))
                added_types = set(entity_types) - saved_types
                saved_types.update(entity_types)
                mapping[types_key] = json.dumps(sorted(saved_types)).encode()
                incoming = mapping.get(f"incoming:{row}".encode())
                if added_types:
                    changed_rows.add(row)
                    if incoming is not None:
                        changed_rows |= BitMap.deserialize(incoming)
                if incoming is not None:
                    pending_neighbor_types.append((row, entity_types, incoming))
                for entity_type in entity_types:
                    pending_types.setdefault(entity_type, BitMap()).add(row)
            self._mark_later_hop_dirty(mapping, changed_rows)

        type_assignments_added = 0
        # Only the type names and one touched bitmap are read at a time.
        with self._open_bitmap_store("c") as store:
            type_names = set(json.loads(store.get(b"type_names", b"[]")))
            for entity_type, added_rows in pending_types.items():
                key = self._type_bitmap_key(entity_type)
                saved = store.get(key)
                bitmap = BitMap.deserialize(saved) if saved is not None else BitMap()
                updated = bitmap | added_rows
                type_assignments_added += len(updated) - len(bitmap)
                store[key] = updated.serialize()
                type_names.add(entity_type)
            store[b"type_names"] = json.dumps(sorted(type_names)).encode()
        for neighbor_row, entity_types, incoming in pending_neighbor_types:
            for anchor_row in BitMap.deserialize(incoming):
                for entity_type in entity_types:
                    self._add_1_hop_type_bucket(
                        anchor_row, entity_type, BitMap([neighbor_row])
                    )
        return {
            "entities_added": len(new_ids),
            "type_assignments_added": type_assignments_added,
            "entity_count": next_row,
        }

    def add_new_1_hop_relationships(self, relationship_batch, batch_size=1000, *, relationships=()) -> int:
        """Add (entity_id, neighbor_ids, total_degree) rows in bounded batches.

        With multiple workers, batch_size is the number of rows per task.
        Completed tasks are staged on disk until readers close. Existing neighbors are
        preserved; the return value counts newly added adjacency memberships.
        IDs and types must already be assigned. Degrees for neighbors outside
        the batch must already be saved, as in the single-row construction API.
        relationships supplies the database relation_id, src_entity_id and
        dst_entity_id dictionaries. Each entity pair stores its single database
        relation_id, accessible from both directions of adjacency.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        rows = iter(relationship_batch)
        first_batch = list(islice(rows, batch_size))
        if not first_batch:
            return 0
        with LmdbStore(self.entity_mapping_path, "w") as mapping:
            for relationship in relationships:
                source = int(mapping[f"id:{int(relationship['src_entity_id'])}".encode()])
                destination = int(mapping[f"id:{int(relationship['dst_entity_id'])}".encode()])
                key = self._relationship_key(source, destination)
                mapping[key] = str(int(relationship["relation_id"])).encode()
        return self._add_parallel_1_hops(chain(first_batch, rows), batch_size)

    def _worker_pool(self):
        if self.workers == 1:
            return nullcontext(None)
        # Fork keeps notebook-loaded modules available without importing the
        # application or copying the graph into every process.
        return ProcessPoolExecutor(max_workers=self.workers,
                                   mp_context=multiprocessing.get_context("fork"))

    def _run_worker_queue(self, pool, jobs, progress=None):
        """Refill a bounded pending queue as any worker completes its task."""
        jobs = iter(jobs)
        pending = set()
        limit = self.workers * 2
        while True:
            while len(pending) < limit:
                try:
                    job = next(jobs)
                except StopIteration:
                    break
                pending.add(pool.submit(self._stage_worker_result, job))
            if not pending:
                return
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                count = future.result()
                if progress is not None:
                    progress.update(count)

    @staticmethod
    def _write_batch_checkpoint(path, state):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as output:
            json.dump(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    def _flush_batch_checkpoint(self, store, mapping, checkpoint, state, pending_outputs):
        """Publish completion only after both stores are flushed; then retire outputs."""
        store.sync()
        mapping.sync()
        self._write_batch_checkpoint(checkpoint, state)
        for path in pending_outputs:
            path.unlink()
        pending_outputs.clear()

    def _checkpointed_worker_batches(self, operation, jobs, name, verbose=False):
        """Freeze batch ranges; checkpoint every five minutes while work remains."""
        last_checkpoint = monotonic()
        directory = self.output_dir / f".batch-checkpoint-{name}"
        directory.mkdir(exist_ok=True)
        checkpoint = directory / "ranges.json"
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
        else:
            state = {}
            start = 0
            for index, (_, task, _) in enumerate(jobs):
                if isinstance(task, str):
                    with open(task, "rb") as source:
                        task = pickle.load(source)
                end = start + len(task[2])
                with (directory / f"input_{index}.pkl").open("wb") as output:
                    pickle.dump(task, output, protocol=pickle.HIGHEST_PROTOCOL)
                    output.flush()
                    os.fsync(output.fileno())
                state[f"{start}:{end}"] = False
                start = end
            self._write_batch_checkpoint(checkpoint, state)
        pending_jobs = (
            (operation, str(directory / f"input_{index}.pkl"),
             str(directory / f"output_{index}.pkl"))
            for index, done in enumerate(state.values())
            if not done and not (directory / f"output_{index}.pkl").exists()
        )
        total = sum(int(key.split(":")[1]) - int(key.split(":")[0]) for key in state)
        ready = sum(
            int(key.split(":")[1]) - int(key.split(":")[0])
            for index, (key, done) in enumerate(state.items())
            if done or (directory / f"output_{index}.pkl").exists()
        )
        with tqdm(total=total, initial=ready, desc=f"Computing Roaring {name}",
                  unit="entity", disable=not verbose) as progress:
            if self.workers > 1:
                with self._worker_pool() as pool:
                    self._run_worker_queue(pool, pending_jobs, progress)
            else:
                for job in pending_jobs:
                    progress.update(self._stage_worker_result(job))
        added = 0
        pending_outputs = []
        remaining_batches = sum(not done for done in state.values())
        # Readers are closed before the parent takes the LMDB write transactions.
        with LmdbStore(str(self.entity_mapping_path), "w") as mapping:
            with self._open_bitmap_store("w") as store:
                for index, (batch_range, done) in enumerate(state.items()):
                    if done:
                        continue
                    output_path = directory / f"output_{index}.pkl"
                    with output_path.open("rb") as source:
                        group = pickle.load(source)
                    if operation == "one":
                        added += self._write_one_hop_worker_batch(group, mapping, store)
                    else:
                        self._write_later_hop_worker_batch(group, mapping, store, int(name.split("_")[-1]))
                    state[batch_range] = True
                    # Retain replayable results until their completion is durable.
                    pending_outputs.append(output_path)
                    remaining_batches -= 1
                    if remaining_batches and monotonic() - last_checkpoint >= self._checkpoint_interval_seconds:
                        self._flush_batch_checkpoint(store, mapping, checkpoint, state, pending_outputs)
                        last_checkpoint = monotonic()
        count = sum(int(key.split(":")[1]) - int(key.split(":")[0]) for key in state)
        shutil.rmtree(directory)
        return added if operation == "one" else count

    def _write_one_hop_worker_batch(self, group, mapping, store):
        added_count = 0
        for row, bitmap, buckets, estimate, added, relationships in group:
            store[self._hop_bitmap_key(row, 1)] = bitmap
            for kind, bucket in buckets.items():
                store[self._hop_bitmap_key(row, 1, kind)] = bucket
            self._save_relationship_buckets(row, 1, relationships, store)
            mapping[f"next_hop_estimate:{row}".encode()] = str(estimate).encode()
            neighbors = BitMap.deserialize(added)
            for other in neighbors:
                key = f"incoming:{other}".encode()
                saved = mapping.get(key)
                incoming = BitMap.deserialize(saved) if saved is not None else BitMap()
                incoming.add(row)
                mapping[key] = incoming.serialize()
            added_count += len(neighbors)
        return added_count

    def _write_later_hop_worker_batch(self, group, mapping, store, hop):
        for row, serialized, estimate, relationships in group:
            self._save_roaring_hop_row(row, hop, BitMap.deserialize(serialized), store=store)
            self._save_relationship_buckets(row, hop, relationships, store)
            mapping[f"next_hop_estimate:{hop}:{row}".encode()] = str(estimate).encode()

    def _prepare_1_hop_batch(self, batch, mapping):
        # Seed total degrees before computing any row in the batch.
        prepared = []
        for entity_id, neighbor_ids, degree in batch:
            row = int(mapping[f"id:{int(entity_id)}".encode()])
            prepared.append((row, neighbor_ids, int(degree)))
        for row, _, degree in prepared:
            degree_key = f"degree:{row}".encode()
            previous_degree = mapping.get(degree_key)
            mapping[degree_key] = str(degree).encode()
            if previous_degree is not None and int(previous_degree) != degree:
                delta = degree - int(previous_degree)
                saved_incoming = mapping.get(f"incoming:{row}".encode())
                if saved_incoming is not None:
                    for anchor_row in BitMap.deserialize(saved_incoming):
                        key = f"next_hop_estimate:{anchor_row}".encode()
                        estimate = mapping.get(key)
                        if estimate is not None:
                            mapping[key] = str(int(estimate) + delta).encode()

        # Persist one pending marker for the supplied batch before bitmap writes.
        self._mark_later_hop_dirty(mapping, BitMap(row for row, _, _ in prepared))
        return prepared

    def _add_parallel_1_hops(self, rows, batch_size):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".roaring-workers-", dir=self.output_dir) as scratch:
            scratch = Path(scratch)
            task_count = 0
            seen = BitMap()
            repeated_source = False
            # Stage bounded inputs and degree changes before read-only workers
            # start. No complete ID mapping or adjacency is retained in memory.
            with LmdbStore(str(self.entity_mapping_path), "w") as mapping:
                while batch := list(islice(rows, batch_size)):
                    combined = {}
                    for entity_id, neighbor_ids, degree in batch:
                        entity_id = int(entity_id)
                        neighbors, _ = combined.setdefault(entity_id, (set(), int(degree)))
                        neighbors.update(int(other) for other in neighbor_ids)
                        combined[entity_id] = (neighbors, int(degree))
                    prepared = self._prepare_1_hop_batch(
                        ((entity_id, neighbors, degree)
                         for entity_id, (neighbors, degree) in combined.items()), mapping)
                    source_rows = BitMap(row for row, _, _ in prepared)
                    repeated_source |= bool(seen & source_rows)
                    seen |= source_rows
                    task = (str(self.entity_mapping_path), str(self.bitmap_store_path), prepared)
                    with (scratch / f"input_{task_count}.pkl").open("wb") as output:
                        pickle.dump(task, output, protocol=pickle.HIGHEST_PROTOCOL)
                    task_count += 1
            if not task_count:
                return 0
            with self._open_bitmap_store("c"):
                pass
            if repeated_source:
                # Dependent updates to the same row retain input order.
                added = 0
                with LmdbStore(str(self.entity_mapping_path), "w") as mapping:
                    with self._open_bitmap_store("w") as store:
                        for index in range(task_count):
                            with (scratch / f"input_{index}.pkl").open("rb") as source:
                                prepared = pickle.load(source)[2]
                            batch = ((int(mapping[f"row:{row}".encode()]), neighbors, degree)
                                     for row, neighbors, degree in prepared)
                            added += self._add_1_hop_relationship_batch(batch, mapping, store)
                return added
            jobs = (("one", str(scratch / f"input_{index}.pkl"), str(scratch / f"output_{index}.pkl"))
                    for index in range(task_count))
            return self._checkpointed_worker_batches("one", jobs, "one_hop")

    def _add_1_hop_relationship_batch(self, batch, mapping, store) -> int:
        prepared = self._prepare_1_hop_batch(batch, mapping)
        added = 0
        for row, neighbor_ids, _ in prepared:
            neighbor_rows = BitMap(
                int(mapping[f"id:{int(neighbor_id)}".encode()])
                for neighbor_id in neighbor_ids
            )
            neighbor_rows.discard(row)
            key = self._hop_bitmap_key(row, 1)
            saved = store.get(key)
            bitmap = BitMap.deserialize(saved) if saved is not None else BitMap()
            updated = bitmap | neighbor_rows
            new_neighbors = neighbor_rows - bitmap
            estimate_key = f"next_hop_estimate:{row}".encode()
            saved_estimate = mapping.get(estimate_key)
            estimate = int(saved_estimate) if saved_estimate is not None else 0
            frontier = new_neighbors if saved_estimate is not None else updated
            estimate += sum(int(mapping[f"degree:{other}".encode()]) for other in frontier)
            mapping[estimate_key] = str(estimate).encode()
            store[key] = updated.serialize()

            relationships = self._compute_relationship_buckets((row,), updated, mapping, store, direct=True)
            self._save_relationship_buckets(row, 1, relationships, store)

            typed_neighbors = {}
            for neighbor_row in neighbor_rows:
                for entity_type in json.loads(mapping.get(f"types:{neighbor_row}".encode(), b"[]")):
                    typed_neighbors.setdefault(entity_type, BitMap()).add(neighbor_row)
            for entity_type, neighbors in typed_neighbors.items():
                key = self._hop_bitmap_key(row, 1, entity_type)
                saved = store.get(key)
                bucket = BitMap.deserialize(saved) if saved is not None else BitMap()
                store[key] = (bucket | neighbors).serialize()
            for neighbor_row in new_neighbors:
                key = f"incoming:{neighbor_row}".encode()
                saved = mapping.get(key)
                incoming = BitMap.deserialize(saved) if saved is not None else BitMap()
                incoming.add(row)
                mapping[key] = incoming.serialize()
            added += len(new_neighbors)
        return added

    def get_entity_degree(self, entity_id) -> int:
        """Read the saved degree without loading the entity's neighbor bitmap."""
        with LmdbStore(str(self.entity_mapping_path), "r") as mapping:
            row = int(mapping[f"id:{int(entity_id)}".encode()])
            return int(mapping[f"degree:{row}".encode()])

    def compute_next_hop_estimates(self, entity_id) -> int:
        """Read the next-hop estimate maintained during one-hop updates.

        Counts adjacency entries to examine, not distinct reachable entities.
        """
        with LmdbStore(str(self.entity_mapping_path), "r") as mapping:
            row = int(mapping[f"id:{int(entity_id)}".encode()])
            return int(mapping[f"next_hop_estimate:{row}".encode()])

    @staticmethod
    def _mark_later_hop_dirty(mapping, rows) -> None:
        if not rows:
            return
        saved = mapping.get(b"later_hop_dirty")
        dirty = BitMap.deserialize(saved) if saved is not None else BitMap()
        mapping[b"later_hop_dirty"] = (dirty | rows).serialize()

    def _expand_roaring_frontier(
        self, frontier, excluded, mapping, high_degree_limit, batch_size, batched,
        *, store,
    ) -> BitMap:
        """Union frontier rows using the current batch's open bitmap store."""
        result = BitMap()
        for frontier_row in frontier:
            neighbors = self._read_hop_bitmap(frontier_row, 1, store=store)
            degree = int(mapping[f"degree:{frontier_row}".encode()])
            if batched or degree > high_degree_limit:
                for start in range(0, len(neighbors), batch_size):
                    result |= neighbors[start:start + batch_size] - excluded
            else:
                result |= neighbors - excluded
        return result

    def _save_roaring_hop_row(self, row, hop, bitmap, *, store) -> None:
        """Save a row and its type buckets in the current batch's store."""
        store[self._hop_bitmap_key(row, hop)] = bitmap.serialize()
        for entity_type in json.loads(store.get(b"type_names", b"[]")):
            type_bitmap = BitMap.deserialize(store[self._type_bitmap_key(entity_type)])
            bucket = bitmap & type_bitmap
            key = self._hop_bitmap_key(row, hop, entity_type)
            if bucket:
                store[key] = bucket.serialize()
            elif key in store:
                # A shortcut can remove all members of an exact-hop bucket.
                del store[key]

    def add_later_hops(
        self, high_degree_limit, batch_size=4096, modified_entity_ids=None, verbose=True,
        start_hop=2,
    ) -> dict:
        """Recompute affected exact-hop rows, using the saved previous frontier.

        Recently modified rows run first, ordered by projected expansion work
        and then degree. Affected existing anchors follow in the same order.
        Expensive expansions union bounded slices instead of gathering all
        neighbor IDs into a temporary list. The accumulated result can grow.
        batch_size also limits anchor rows processed per bitmap-store open.
        The mapping store stays open for the operation; results are streamed.
        Set start_hop to extend already-saved frontiers without rebuilding
        earlier hops; supply modified_entity_ids to select the source rows.
        """
        if isinstance(start_hop, bool) or not isinstance(start_hop, int) or start_hop < 2:
            raise ValueError("start_hop must be an integer of at least 2")
        if high_degree_limit < 0:
            raise ValueError("high_degree_limit cannot be negative")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.max_hop < 2:
            return {}

        return self._add_parallel_later_hops(
            high_degree_limit, batch_size, modified_entity_ids, verbose, start_hop)

    def _affected_later_hop_rows(self, modified_entity_ids):
        with LmdbStore(str(self.entity_mapping_path), "w") as mapping:
            if modified_entity_ids is not None:
                supplied_rows = BitMap(
                    int(mapping[f"id:{int(entity_id)}".encode()])
                    for entity_id in modified_entity_ids
                )
                self._mark_later_hop_dirty(mapping, supplied_rows)
            saved_dirty = mapping.get(b"later_hop_dirty")
            recent = BitMap.deserialize(saved_dirty) if saved_dirty is not None else BitMap()
            if not recent:
                return recent, recent.copy()

            # A changed one-hop row also affects anchors that can reach it.
            # Follow the saved reverse index, not every entity in the database.
            affected = recent.copy()
            frontier = recent.copy()
            for _ in range(self.max_hop - 1):
                preceding = BitMap()
                for row in frontier:
                    incoming = mapping.get(f"incoming:{row}".encode())
                    if incoming is not None:
                        preceding |= BitMap.deserialize(incoming)
                frontier = preceding - affected
                affected |= frontier
                if not frontier:
                    break

            return recent, affected

    def _add_parallel_later_hops(self, high_degree_limit, batch_size,
                                 modified_entity_ids, verbose, start_hop):
        recent, affected = self._affected_later_hop_rows(modified_entity_ids)
        if not recent:
            return {}
        completed = {}
        for hop in range(start_hop, self.max_hop + 1):
            with LmdbStore(str(self.entity_mapping_path), "r") as mapping:
                def scheduling_key(row):
                    key = (f"next_hop_estimate:{row}" if hop == 2
                           else f"next_hop_estimate:{hop - 1}:{row}").encode()
                    return int(mapping[key]), int(mapping[f"degree:{row}".encode()]), row
                ordered = sorted(recent, key=scheduling_key)
                ordered.extend(sorted(affected - recent, key=scheduling_key))
                scheduled = [(row, *scheduling_key(row)[:2]) for row in ordered]
            jobs = (
                ("later", (str(self.entity_mapping_path), str(self.bitmap_store_path),
                           scheduled[start:start + batch_size], hop, high_degree_limit, batch_size), "")
                for start in range(0, len(scheduled), batch_size)
            )
            completed[f"hop_{hop}_rows"] = self._checkpointed_worker_batches(
                "later", jobs, f"hop_{hop}", verbose=verbose)
        with LmdbStore(str(self.entity_mapping_path), "w") as mapping:
            mapping[b"later_hop_dirty"] = BitMap().serialize()
        return completed

    def save_entity_stats(self, avg_entity_degree, max_entity_degree, least_entity_degree) -> None:
        """Save supplied degree statistics without computing them from the index."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "entity_stats.json"
        stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        stats.update({
            "avg_entity_degree": float(avg_entity_degree),
            "max_entity_degree": int(max_entity_degree),
            "least_entity_degree": int(least_entity_degree),
        })
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)

    def _open_bitmap_store(self, flag="r"):
        """Open an indexed disk store; never load a Python index of every key."""
        return LmdbStore(str(self.bitmap_store_path), flag)

    @staticmethod
    def _relationship_bitmap_key(row, hop, entity_type=None):
        prefix = f"relationship_hop:{hop}:{row}".encode()
        return prefix + b":all" if entity_type is None else prefix + b":type:" + entity_type.encode("utf-8")

    @staticmethod
    def _relationship_key(source_row, destination_row):
        first, second = sorted((source_row, destination_row))
        return f"relationships:{first}:{second}".encode()

    @staticmethod
    def _type_bitmap_key(entity_type):
        return b"type:" + entity_type.encode("utf-8")

    @staticmethod
    def _hop_bitmap_key(row, hop, entity_type=None):
        prefix = f"hop:{hop}:{row}".encode()
        return prefix + b":all" if entity_type is None else prefix + b":type:" + entity_type.encode("utf-8")

    def _read_hop_bitmap(self, row, hop, entity_type=None, store=None):
        """Read only the requested row or type bucket from disk."""
        if store is None:
            with self._open_bitmap_store("r") as opened:
                return self._read_hop_bitmap(row, hop, entity_type, opened)
        key = self._hop_bitmap_key(row, hop, entity_type)
        if entity_type is None:
            return BitMap.deserialize(store[key])
        saved = store.get(key)
        return BitMap.deserialize(saved) if saved is not None else BitMap()

    def _add_1_hop_type_bucket(self, row, entity_type, neighbor_rows) -> None:
        with LmdbStore(self.entity_mapping_path, "r") as mapping, self._open_bitmap_store("w") as store:
            bitmap = self._read_hop_bitmap(row, 1, entity_type, store)
            store[self._hop_bitmap_key(row, 1, entity_type)] = (bitmap | neighbor_rows).serialize()
            key = self._relationship_bitmap_key(row, 1, entity_type)
            saved = store.get(key)
            relationships = BitMap64.deserialize(saved) if saved is not None else BitMap64()
            for neighbor in neighbor_rows:
                relation_id = mapping.get(self._relationship_key(row, neighbor))
                if relation_id is not None:
                    relationships.add(int(relation_id) & ((1 << 64) - 1))
            store[key] = relationships.serialize()

    def get_hop_neighbors_by_type(self, entity_id, hop, entity_type) -> list[int]:
        """Read one indexed type bucket without scanning the full neighborhood."""
        if not 1 <= hop <= self.max_hop:
            raise ValueError("hop must be between 1 and max_hop")
        with LmdbStore(str(self.entity_mapping_path), "r") as mapping:
            row = int(mapping[f"id:{int(entity_id)}".encode()])
            return [
                int(mapping[f"row:{neighbor_row}".encode()])
                for neighbor_row in self._read_hop_bitmap(row, hop, entity_type)
            ]

    @staticmethod
    def _normalise_types(value) -> list[str]:
        """Return an entity's type attribute as a list."""
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        return [str(item) for item in value]

