
from __future__ import annotations
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
import json
import multiprocessing
import os
import time
from collections import deque
from pathlib import Path
import igraph as ig
import numpy as np
from tqdm import tqdm


# On POSIX systems, later-hop workers inherit this read-only graph through
# fork. Its memory pages remain shared because workers never mutate the graph.
_IGRAPH_TRAVERSAL_GRAPH: ig.Graph | None = None


def _encode_igraph_exact_hop(
    anchor: int,
    target_hop: int,
    node_count: int,
    bytes_per_row: int,
) -> tuple[int, int, bytes, int]:
    """Discover and adaptively encode one exact-hop row with iGraph."""
    if _IGRAPH_TRAVERSAL_GRAPH is None:
        raise RuntimeError("The shared iGraph has not been constructed")
    neighborhood = np.asarray(
        _IGRAPH_TRAVERSAL_GRAPH.neighborhood(
            vertices=anchor,
            order=target_hop,
            mindist=target_hop,
            mode="all",
        ),
        dtype=np.uint32,
    )
    if neighborhood.nbytes < bytes_per_row:
        return anchor, 0, neighborhood.tobytes(), len(neighborhood)

    membership = np.zeros(node_count, dtype=np.uint8)
    membership[neighborhood] = 1
    payload = np.packbits(
        membership, bitorder="little"
    ).tobytes()
    return anchor, 1, payload, len(neighborhood)


def _build_igraph_hop_anchor_range(
    target_hop: int,
    target_hop_path: str,
    node_count: int,
    anchor_start: int,
    anchor_end: int,
) -> int:
    """Build one exact-hop bitmap range using the inherited iGraph."""
    if _IGRAPH_TRAVERSAL_GRAPH is None:
        raise RuntimeError(
            "The temporary iGraph was not inherited by the build worker"
        )

    anchors = list(range(anchor_start, anchor_end))
    neighborhoods = _IGRAPH_TRAVERSAL_GRAPH.neighborhood(
        vertices=anchors,
        order=target_hop,
        mindist=target_hop,
        mode="all",
    )

    # Convert each exact-hop neighborhood into one packed bitmap row.
    membership = np.zeros(
        (anchor_end - anchor_start, node_count),
        dtype=bool,
    )
    for row_index, neighborhood in enumerate(neighborhoods):
        membership[row_index, neighborhood] = True

    packed_rows = np.packbits(
        membership,
        axis=1,
        bitorder="little",
    )

    # Each task owns a disjoint contiguous range, so workers can write the
    # shared memory-mapped output without locks.
    output_rows = np.lib.format.open_memmap(
        target_hop_path,
        mode="r+",
    )
    output_rows[anchor_start:anchor_end] = packed_rows
    del output_rows

    return anchor_end - anchor_start


def _available_build_workers() -> int:
    """Return the CPU count available to this process or Slurm task."""
    limits = []

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            limits.append(int(slurm_cpus))
        except ValueError:
            pass

    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:
        pass

    if limits:
        return max(1, min(limits))

    return os.cpu_count() or 1


def _build_exact_hop_anchor_range(
    adjacency_offsets_path: str,
    adjacency_neighbors_path: str,
    hop_1_offsets_path: str,
    hop_1_nodes_path: str,
    prior_hop_paths: dict[int, str],
    target_hop: int,
    target_hop_path: str,
    anchor_start: int,
    anchor_end: int,
) -> tuple[int, float]:
    """Construct one exact-hop phase for a disjoint range of anchors."""
    adjacency_offsets = np.load(
        adjacency_offsets_path,
        mmap_mode="r",
    )
    adjacency_neighbors = np.load(
        adjacency_neighbors_path,
        mmap_mode="r",
    )
    hop_1_offsets = np.load(
        hop_1_offsets_path,
        mmap_mode="r",
    )
    hop_1_nodes = np.load(
        hop_1_nodes_path,
        mmap_mode="r",
    )
    prior_hop_rows = {
        hop: np.load(path, mmap_mode="r")
        for hop, path in prior_hop_paths.items()
    }
    node_count = len(adjacency_offsets) - 1
    batch_anchor_count = anchor_end - anchor_start

    # Prepare the complete target-hop range in process-local memory. Workers
    # do not mutate the shared output file while expanding paths.
    shape_reader = np.load(
        target_hop_path,
        mmap_mode="r",
    )
    bytes_per_row = int(shape_reader.shape[1])
    del shape_reader

    batch_rows = np.zeros(
        (batch_anchor_count, bytes_per_row),
        dtype=np.uint8,
    )
    hop_seconds = 0.0

    # Every worker writes only to its assigned anchor rows. The ranges never
    # overlap, so workers can safely populate the same memory-mapped files
    # without locks or inter-process synchronization.
    for anchor_index in range(anchor_start, anchor_end):
        hop_1_start = int(
            hop_1_offsets[anchor_index]
        )
        hop_1_end = int(
            hop_1_offsets[anchor_index + 1]
        )
        hop_1_frontier = np.asarray(
            hop_1_nodes[
                hop_1_start:hop_1_end
            ],
            dtype=np.int64,
        )

        # Track previously reached entities in a dense Boolean workspace.
        # Direct Boolean assignment handles repeated neighbor IDs without the
        # sorting cost of unique() or the indexed-update cost of bitwise_or.at.
        visited = np.zeros(
            node_count,
            dtype=bool,
        )
        visited[anchor_index] = True
        visited[hop_1_frontier] = True

        # Hop two expands the one-hop frontier. Later phases read the exact
        # frontier completed by the immediately preceding global hop phase.
        frontier = hop_1_frontier
        for prior_hop in range(2, target_hop):
            packed_row = prior_hop_rows[prior_hop][
                anchor_index
            ]
            prior_frontier = np.flatnonzero(
                np.unpackbits(
                    packed_row,
                    count=node_count,
                    bitorder="little",
                )
            )
            visited[prior_frontier] = True
            frontier = prior_frontier

        batch_row_index = anchor_index - anchor_start
        hop_started = time.perf_counter()

        # Mark neighbors directly in a Boolean membership workspace. Repeated
        # neighbors simply write True again, so no duplicate-filled temporary
        # array and no sorting pass are required.
        candidate_nodes = np.zeros(
            node_count,
            dtype=bool,
        )

        for frontier_entity in frontier:
            adjacency_start = int(
                adjacency_offsets[frontier_entity]
            )
            adjacency_end = int(
                adjacency_offsets[frontier_entity + 1]
            )

            if adjacency_end > adjacency_start:
                candidate_nodes[
                    adjacency_neighbors[
                        adjacency_start:adjacency_end
                    ]
                ] = True

        # Exact hop h contains the accumulated neighbors excluding the anchor
        # and every entity already reached at a shorter distance.
        np.bitwise_and(
            candidate_nodes,
            np.logical_not(visited),
            out=candidate_nodes,
        )
        batch_rows[batch_row_index] = np.packbits(
            candidate_nodes,
            bitorder="little",
        )

        hop_seconds += time.perf_counter() - hop_started

    # Commit the completed target-hop batch with one contiguous assignment.
    # Ranges never overlap, and the parent performs the final file flush.
    output_rows = np.lib.format.open_memmap(
        target_hop_path,
        mode="r+",
    )
    output_rows[anchor_start:anchor_end] = batch_rows
    del output_rows

    return anchor_end - anchor_start, hop_seconds


class ExactHopBitmapBuilder:
    """Construct exact-hop, type-filtered bitmap sidecars.

    Entities and relationships can be supplied over multiple batches using
    ``add_entities()`` and ``add_adjacency()``. When ``build()`` is called,
    the one-hop adjacency is constructed first. Exact later hops are then
    derived by repeatedly expanding that one-hop adjacency.

    The builder creates the following files:

    ``entity_ids.npy``
        Maps each dense bitmap position to its original database entity ID.

    ``entity_degrees.npy``
        Stores the number of adjacency entries associated with every entity.

    ``adjacency_offsets.npy``
        Stores the start and end positions of each entity's adjacency list
        using compressed sparse row (CSR) indexing.

    ``adjacency_neighbors.npy``
        Stores the dense entity IDs of adjacent neighbors. Its positions are
        interpreted using ``adjacency_offsets.npy``.

    ``adjacency_relationship_ids.npy``
        Stores the relationship ID associated with every entry in
        ``adjacency_neighbors.npy``.

    ``entity_types.npy``
        Stores one packed membership bitmap per entity type. These rows are
        intersected with hop neighborhoods to retrieve entities by both
        distance and type.

    ``hop_1_offsets.npy``
        Stores the boundaries of each anchor's exact one-hop neighbor list.

    ``hop_1_nodes.npy``
        Stores the dense IDs reachable at exactly one hop from each anchor.

    ``hop_N_offsets.npy``
        Stores byte boundaries for each adaptively encoded exact-hop row.

    ``hop_N_encoding.npy``
        Records whether each row stores sparse uint32 IDs or packed bits.

    ``hop_N_payload.bin``
        Stores the adaptive exact-hop row payloads.

    ``metadata.json``
        Records the index dimensions, entity-type row ordering, and maximum
        hop supported by the constructed sidecar.
    """

    def __init__(
        self,
        output_dir: str | Path,
        max_hop: int = 3,
        build_workers: int | None = None,
        anchor_time_limit_seconds: float = 180.0,
        hybrid_igraph: bool = True,
        igraph_long_workers: int | None = None,
        igraph_work_threshold: int | None = None,
        quick_time_limit_seconds: float = 5.0,
        traversal_graph: ig.Graph | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.max_hop = int(max_hop)
        self.build_workers = int(
            build_workers
            if build_workers is not None
            else _available_build_workers()
        )
        self.anchor_time_limit_seconds = float(
            anchor_time_limit_seconds
        )
        self.hybrid_igraph = bool(hybrid_igraph)
        self.igraph_long_workers = int(
            igraph_long_workers
            if igraph_long_workers is not None
            else max(1, self.build_workers // 4)
        )
        self.igraph_work_threshold = (
            None
            if igraph_work_threshold is None
            else int(igraph_work_threshold)
        )
        self.quick_time_limit_seconds = float(
            quick_time_limit_seconds
        )
        self.traversal_graph = traversal_graph

        if self.max_hop < 1:
            raise ValueError("max_hop must be at least 1")

        if self.build_workers < 1:
            raise ValueError("build_workers must be at least 1")

        if self.anchor_time_limit_seconds <= 0:
            raise ValueError(
                "anchor_time_limit_seconds must be positive"
            )
        if self.igraph_long_workers < 1:
            raise ValueError("igraph_long_workers must be positive")
        if self.quick_time_limit_seconds <= 0:
            raise ValueError("quick_time_limit_seconds must be positive")

        # All sidecar output paths are defined together.
        self.metadata_path = self.output_dir / "metadata.json"
        self.entity_ids_path = self.output_dir / "entity_ids.npy"
        self.entity_degrees_path = self.output_dir / "entity_degrees.npy"
        self.adjacency_offsets_path = (
            self.output_dir / "adjacency_offsets.npy"
        )
        self.adjacency_neighbors_path = (
            self.output_dir / "adjacency_neighbors.npy"
        )
        self.adjacency_relationship_ids_path = (
            self.output_dir / "adjacency_relationship_ids.npy"
        )
        self.entity_types_path = self.output_dir / "entity_types.npy"
        self.hop_1_offsets_path = self.output_dir / "hop_1_offsets.npy"
        self.hop_1_nodes_path = self.output_dir / "hop_1_nodes.npy"
        self.hop_offsets_paths = {
            hop: self.output_dir / f"hop_{hop}_offsets.npy"
            for hop in range(2, self.max_hop + 1)
        }
        self.hop_encoding_paths = {
            hop: self.output_dir / f"hop_{hop}_encoding.npy"
            for hop in range(2, self.max_hop + 1)
        }
        self.hop_payload_paths = {
            hop: self.output_dir / f"hop_{hop}_payload.bin"
            for hop in range(2, self.max_hop + 1)
        }
        self.hop_paths = self.hop_payload_paths

        # Dense array position maps to the original external entity ID.
        self.entity_ids: list[int] = []

        # External entity ID maps back to its dense array position.
        self.entity_id_to_dense_id: dict[int, int] = {}

        # Store all types assigned to each external entity ID.
        self.entity_types_by_id: dict[int, set[str]] = {}

        # Store undirected one-hop adjacency:
        #
        # adjacency[entity_id][relation_id] = neighboring_entity_id
        self.adjacency: dict[int, dict[int, int]] = {}

        # Preserve each relationship once:
        #
        # relationships[relation_id] = (src_entity_id, dst_entity_id)
        self.relationships: dict[int, tuple[int, int]] = {}

    @staticmethod
    def _normalise_types(value) -> list[str]:
        """Return an entity's type attribute as a list."""
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        return [str(item) for item in value]

    @staticmethod
    def _set_packed_bits(
        row: np.ndarray,
        indices,
    ) -> None:
        """Mark the supplied dense entity IDs as present in a packed bitmap."""

        # Convert the dense entity IDs into a NumPy integer array.
        values = np.asarray(indices, dtype=np.int64)

        # Nothing needs to be marked when the input is empty.
        if values.size == 0:
            return

        # Determine which byte stores each entity's membership bit.
        # Shifting right by three is equivalent to integer division by eight.
        #
        # Example:
        # entity 10 -> byte 1
        byte_indices = values >> 3

        # Determine the bit within that byte.
        # values & 7 is equivalent to values % 8.
        #
        # Example:
        # entity 10 -> bit 2 -> mask 00000100
        bit_masks = np.left_shift(
            np.uint8(1),
            (values & 7).astype(np.uint8),
        )

        # OR every mask into its corresponding byte. np.bitwise_or.at is used
        # because multiple entities can occupy the same byte and every bit
        # must be preserved.
        np.bitwise_or.at(
            row,
            byte_indices,
            bit_masks,
        )

    def add_entities(self, entity_batch) -> dict:
        """Add ``(entity_id, entity_type)`` tuples to the index.

        A new entity receives the next available dense position. Existing
        entities retain their dense positions and receive any newly supplied
        type assignments.

        The same entity may appear multiple times when it carries multiple
        types. Entities without relationships remain indexed as islands.
        """
        entities_added = 0
        type_assignments_added = 0

        for entity_id, entity_type in entity_batch:
            entity_id = int(entity_id)

            # Any ID not already in entity_ids is a new entity.
            if entity_id not in self.entity_id_to_dense_id:
                dense_id = len(self.entity_ids)

                self.entity_id_to_dense_id[entity_id] = dense_id
                self.entity_ids.append(entity_id)
                self.entity_types_by_id[entity_id] = set()
                self.adjacency[entity_id] = {}

                entities_added += 1

            # An entity can possess one or multiple types. Repeated tuples for
            # the same entity simply extend its type membership.
            for normalised_type in self._normalise_types(entity_type):
                entity_types = self.entity_types_by_id[entity_id]

                if normalised_type not in entity_types:
                    entity_types.add(normalised_type)
                    type_assignments_added += 1

        return {
            "entities_added": entities_added,
            "type_assignments_added": type_assignments_added,
            "entity_count": len(self.entity_ids),
        }

    def _remove_relationship(self, relationship_id: int) -> None:
        """Remove an existing relationship from both endpoint lists."""
        endpoints = self.relationships.pop(relationship_id, None)

        if endpoints is None:
            return

        source_id, destination_id = endpoints

        self.adjacency[source_id].pop(
            relationship_id,
            None,
        )
        self.adjacency[destination_id].pop(
            relationship_id,
            None,
        )

    def add_adjacency(self, relationship_batch) -> dict:
        """Add ``(src_id, relation_id, dst_id)`` tuples to the index.

        Relationships are stored under both endpoints because the reachability
        index treats adjacency as undirected. Both endpoint entities must be
        added through ``add_entities()`` before their relationship is added.

        If an existing relationship ID is supplied with different endpoints,
        its previous adjacency entries are replaced.
        """
        relationships_added = 0
        relationships_replaced = 0
        relationships_unchanged = 0

        for source_id, relationship_id, destination_id in relationship_batch:
            source_id = int(source_id)
            relationship_id = int(relationship_id)
            destination_id = int(destination_id)

            # Every relationship endpoint must already possess a stable dense
            # ID before adjacency can be constructed.
            if source_id not in self.entity_id_to_dense_id:
                raise KeyError(
                    f"Unknown source entity ID {source_id}; add it first"
                )

            if destination_id not in self.entity_id_to_dense_id:
                raise KeyError(
                    f"Unknown destination entity ID "
                    f"{destination_id}; add it first"
                )

            new_endpoints = (
                source_id,
                destination_id,
            )
            existing_endpoints = self.relationships.get(
                relationship_id
            )

            # Repeating the same relationship tuple has no effect.
            if existing_endpoints == new_endpoints:
                relationships_unchanged += 1
                continue

            # If the relationship ID already exists with different endpoints,
            # remove its old adjacency entries before inserting the replacement.
            if existing_endpoints is not None:
                self._remove_relationship(relationship_id)
                relationships_replaced += 1
            else:
                relationships_added += 1

            self.relationships[relationship_id] = new_endpoints

            # Record the destination as a neighbor of the source.
            self.adjacency[source_id][
                relationship_id
            ] = destination_id

            # Record the source as a neighbor of the destination, creating the
            # reverse direction required for undirected traversal.
            self.adjacency[destination_id][
                relationship_id
            ] = source_id

        return {
            "relationships_added": relationships_added,
            "relationships_replaced": relationships_replaced,
            "relationships_unchanged": relationships_unchanged,
            "relationship_count": len(self.relationships),
        }

    def _save_entity_ids(self) -> np.ndarray:
        """Save external entity IDs in dense-index order."""
        entity_ids = np.asarray(
            self.entity_ids,
            dtype=np.int64,
        )

        np.save(
            self.entity_ids_path,
            entity_ids,
        )

        return entity_ids

    def _save_adjacency(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Save undirected graph adjacency in CSR form."""

        # Count the graph entities. Each entity receives one adjacency-list
        # segment in the CSR representation.
        node_count = len(self.entity_ids)
        from ._bitmap_bfs import build_hop_one_from_edges

        print("Building exact hop 1 with compiled degree counting")
        (
            entity_degrees,
            adjacency_offsets,
            adjacency_neighbors,
        ) = build_hop_one_from_edges(
            self._dense_edges,
            node_count,
            min(self.build_workers, node_count),
        )

        np.save(
            self.entity_degrees_path,
            entity_degrees,
        )
        np.save(
            self.adjacency_offsets_path,
            adjacency_offsets,
        )
        np.save(
            self.adjacency_neighbors_path,
            adjacency_neighbors,
        )
        np.save(self.hop_1_offsets_path, adjacency_offsets)
        np.save(
            self.hop_1_nodes_path,
            adjacency_neighbors,
        )
        return (
            entity_degrees,
            adjacency_offsets,
            adjacency_neighbors,
        )

    def _save_entity_types(
        self,
        bytes_per_row: int,
    ) -> list[str]:
        """Save one packed membership bitmap per entity type."""

        # Collect every unique entity type and sort them to give the bitmap
        # rows a stable order.
        type_names = sorted(
            {
                entity_type
                for entity_types in self.entity_types_by_id.values()
                for entity_type in entity_types
            }
        )

        # Create entity_types.npy as a two-dimensional memory-mapped array:
        #
        #     rows    = entity types
        #     columns = packed entity-membership bytes
        #
        # Each bit states whether one dense entity ID belongs to that type.
        type_rows = np.lib.format.open_memmap(
            self.entity_types_path,
            mode="w+",
            dtype=np.uint8,
            shape=(
                len(type_names),
                bytes_per_row,
            ),
        )

        # Initialize every membership bit to false.
        type_rows[:] = 0

        # Map each type name to its corresponding row in entity_types.npy.
        #
        # Example:
        # {
        #     "ORGANIZATION": 0,
        #     "PERSON": 1,
        #     "PRODUCT": 2,
        # }
        type_to_index = {
            name: index
            for index, name in enumerate(type_names)
        }

        # Visit every indexed entity. Its dense position determines which bit
        # is set in each applicable entity-type row.
        for entity_id in self.entity_ids:
            dense_id = self.entity_id_to_dense_id[
                entity_id
            ]

            # An entity may possess one or multiple types.
            for entity_type in self.entity_types_by_id[
                entity_id
            ]:
                # Mark this entity as present in the bitmap row associated with
                # the current type.
                self._set_packed_bits(
                    type_rows[
                        type_to_index[entity_type]
                    ],
                    [dense_id],
                )

        # Ensure the memory-mapped changes are written to disk.
        type_rows.flush()

        # Return the row ordering so it can be saved in metadata.json. The
        # reader needs this ordering to know which row represents each type.
        return type_names

    def _save_later_hops(
        self,
        indexed_anchor_count: int,
        bytes_per_row: int,
    ) -> None:
        """Build every exact hop after hop one with one generic recurrence."""

        global _IGRAPH_TRAVERSAL_GRAPH
        _IGRAPH_TRAVERSAL_GRAPH = self.traversal_graph
        if (
            _IGRAPH_TRAVERSAL_GRAPH is not None
            and _IGRAPH_TRAVERSAL_GRAPH.vcount()
            != indexed_anchor_count
        ):
            raise ValueError(
                "traversal_graph vertex count does not match the index"
            )
        if self.hybrid_igraph and _IGRAPH_TRAVERSAL_GRAPH is not None:
            print(
                "Using shared hybrid iGraph: "
                f"{_IGRAPH_TRAVERSAL_GRAPH.vcount():,} nodes, "
                f"{_IGRAPH_TRAVERSAL_GRAPH.ecount():,} edges"
            )

        # A one-hop-only index does not require additional hop files.
        if not self.hop_paths:
            return

        try:
            from ._bitmap_bfs import (
                build_next_hop_from_global_previous_indices,
                classify_frontiers_by_degree,
            )
        except ImportError as error:
            raise RuntimeError(
                "The compiled bitmap BFS extension is unavailable. "
                "Rebuild and reinstall DotVectorDB."
            ) from error

        adjacency_offsets = np.load(
            self.adjacency_offsets_path,
            mmap_mode="r",
        )
        adjacency_neighbors = np.load(
            self.adjacency_neighbors_path,
            mmap_mode="r",
        )
        routing_threshold = (
            self.igraph_work_threshold
            if self.igraph_work_threshold is not None
            else indexed_anchor_count * 4
        )

        # Dense-adjacency caching is disabled. Every compiled traversal uses
        # the canonical CSR neighbor arrays, while expensive anchors are sent
        # to the shared iGraph lane.
        heavy_row_by_node = np.full(
            indexed_anchor_count,
            -1,
            dtype=np.int32,
        )
        heavy_adjacency_bitmaps = np.empty(
            (0, bytes_per_row),
            dtype=np.uint8,
        )

        worker_count = min(
            self.build_workers,
            indexed_anchor_count,
        )
        anchors_per_batch = min(
            indexed_anchor_count,
            max(256, worker_count * 64),
        )
        later_hop_started = time.perf_counter()

        def completed_phase_is_valid(hop: int) -> bool:
            """Return whether a complete adaptive phase can be reused."""
            offsets_path = self.hop_offsets_paths[hop]
            encodings_path = self.hop_encoding_paths[hop]
            payload_path = self.hop_payload_paths[hop]
            if not all(
                path.exists()
                for path in (offsets_path, encodings_path, payload_path)
            ):
                return False
            try:
                offsets = np.load(offsets_path, mmap_mode="r")
                encodings = np.load(encodings_path, mmap_mode="r")
                valid = (
                    offsets.shape == (indexed_anchor_count + 1,)
                    and encodings.shape == (indexed_anchor_count,)
                    and int(offsets[0]) == 0
                    and int(offsets[-1]) == payload_path.stat().st_size
                    and bool(np.all(offsets[1:] >= offsets[:-1]))
                )
                del offsets
                del encodings
                return bool(valid)
            except (OSError, ValueError):
                return False

        def write_checkpoint(
            checkpoint_path: Path,
            hop: int,
            next_anchor: int,
            payload_bytes: int,
        ) -> None:
            """Atomically commit the last durable batch boundary."""
            checkpoint = {
                "format_version": 1,
                "algorithm": "adaptive_hybrid_v2",
                "hop": hop,
                "node_count": indexed_anchor_count,
                "next_anchor": next_anchor,
                "payload_bytes": payload_bytes,
            }
            temporary_path = checkpoint_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(checkpoint, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, checkpoint_path)

        def save_phase(
            hop: int,
            description: str,
            build_batch,
            save_frontier_work: bool = False,
        ) -> None:
            """Write or resume one adaptive exact-hop construction phase."""
            if completed_phase_is_valid(hop):
                print(
                    f"Reusing completed exact hop {hop}: "
                    f"{indexed_anchor_count:,} anchors"
                )
                return

            partial_offsets_path = self.output_dir / (
                f"hop_{hop}_offsets.partial.npy"
            )
            partial_encodings_path = self.output_dir / (
                f"hop_{hop}_encoding.partial.npy"
            )
            partial_payload_path = self.output_dir / (
                f"hop_{hop}_payload.partial.bin"
            )
            checkpoint_path = self.output_dir / (
                f"hop_{hop}_checkpoint.json"
            )
            incomplete_manifest_path = self.output_dir / (
                f"hop_{hop}_incomplete.jsonl"
            )
            partial_work_path = self.output_dir / (
                f"hop_{hop}_frontier_work.partial.npy"
            )

            anchor_start = 0
            payload_bytes = 0
            resume = False
            if checkpoint_path.exists():
                try:
                    checkpoint = json.loads(
                        checkpoint_path.read_text(encoding="utf-8")
                    )
                    anchor_start = int(checkpoint["next_anchor"])
                    payload_bytes = int(checkpoint["payload_bytes"])
                    resume = (
                        checkpoint.get("format_version") == 1
                        and checkpoint.get("algorithm")
                        == "adaptive_hybrid_v2"
                        and int(checkpoint.get("hop", -1)) == hop
                        and int(checkpoint.get("node_count", -1))
                        == indexed_anchor_count
                        and 0 <= anchor_start <= indexed_anchor_count
                        and payload_bytes >= 0
                        and partial_offsets_path.exists()
                        and partial_encodings_path.exists()
                        and partial_payload_path.exists()
                        and (
                            not save_frontier_work
                            or partial_work_path.exists()
                        )
                        and partial_payload_path.stat().st_size
                        >= payload_bytes
                    )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    resume = False

            if resume:
                offsets = np.lib.format.open_memmap(
                    partial_offsets_path, mode="r+"
                )
                encodings = np.lib.format.open_memmap(
                    partial_encodings_path, mode="r+"
                )
                if (
                    offsets.shape != (indexed_anchor_count + 1,)
                    or encodings.shape != (indexed_anchor_count,)
                    or int(offsets[anchor_start]) != payload_bytes
                ):
                    resume = False
                    del offsets
                    del encodings

            if not resume:
                anchor_start = 0
                payload_bytes = 0
                offsets = np.lib.format.open_memmap(
                    partial_offsets_path,
                    mode="w+",
                    dtype=np.uint64,
                    shape=(indexed_anchor_count + 1,),
                )
                encodings = np.lib.format.open_memmap(
                    partial_encodings_path,
                    mode="w+",
                    dtype=np.uint8,
                    shape=(indexed_anchor_count,),
                )
                with partial_payload_path.open("wb"):
                    pass
                incomplete_manifest_path.unlink(missing_ok=True)
                if save_frontier_work:
                    frontier_work = np.lib.format.open_memmap(
                        partial_work_path,
                        mode="w+",
                        dtype=np.uint64,
                        shape=(indexed_anchor_count,),
                    )
                write_checkpoint(checkpoint_path, hop, 0, 0)
            else:
                if save_frontier_work:
                    frontier_work = np.lib.format.open_memmap(
                        partial_work_path, mode="r+"
                    )
                print(
                    f"Resuming exact hop {hop} at anchor "
                    f"{anchor_start:,}/{indexed_anchor_count:,}"
                )

            # Discard bytes written after the last atomically committed batch.
            with partial_payload_path.open("r+b") as payload_file:
                payload_file.truncate(payload_bytes)
                payload_file.seek(payload_bytes)
                with tqdm(
                    total=indexed_anchor_count,
                    initial=anchor_start,
                    desc=description,
                    unit="anchor",
                    dynamic_ncols=True,
                ) as progress:
                    for anchor_start in range(
                        anchor_start, indexed_anchor_count, anchors_per_batch
                    ):
                        anchor_end = min(
                            anchor_start + anchors_per_batch,
                            indexed_anchor_count,
                        )
                        batch_result = build_batch(
                            anchor_start, anchor_end
                        )
                        (
                            batch_offsets,
                            batch_encodings,
                            batch_payload,
                        ) = batch_result[:3]
                        batch_frontier_work = (
                            batch_result[3]
                            if save_frontier_work
                            else None
                        )
                        incomplete_records = []
                        if len(batch_result) > (4 if save_frontier_work else 3):
                            (
                                batch_timed_out,
                                batch_elapsed,
                                batch_partial_counts,
                                batch_resume_positions,
                                batch_resume_strategies,
                            ) = batch_result[3:]
                            strategy_names = {
                                1: "direct",
                                2: "union",
                                3: "estimation",
                            }
                            for local_anchor in np.flatnonzero(
                                batch_timed_out
                            ):
                                dense_anchor = (
                                    anchor_start + int(local_anchor)
                                )
                                original_entity_id = int(
                                    self.entity_ids[dense_anchor]
                                )
                                elapsed_seconds = float(
                                    batch_elapsed[local_anchor]
                                )
                                partial_count = int(
                                    batch_partial_counts[local_anchor]
                                )
                                resume_strategy = int(
                                    batch_resume_strategies[local_anchor]
                                )
                                record = {
                                    "dense_anchor_index": dense_anchor,
                                    "original_entity_id": (
                                        original_entity_id
                                    ),
                                    "hop": hop,
                                    "elapsed_seconds": elapsed_seconds,
                                    "time_limit_seconds": (
                                        self.anchor_time_limit_seconds
                                    ),
                                    "partial_reachable_count": (
                                        partial_count
                                    ),
                                    "resume_position": int(
                                        batch_resume_positions[
                                            local_anchor
                                        ]
                                    ),
                                    "resume_strategy": (
                                        strategy_names.get(
                                            resume_strategy,
                                            "unknown",
                                        )
                                    ),
                                    "status": "partial",
                                }
                                incomplete_records.append(record)
                                tqdm.write(
                                    "Anchor original ID "
                                    f"{original_entity_id} "
                                    f"(dense {dense_anchor}) exceeded "
                                    f"{self.anchor_time_limit_seconds:.1f}s "
                                    f"at hop {hop}; saved partial row "
                                    f"with {partial_count:,} nodes"
                                )
                        payload_start = payload_file.tell()
                        payload_file.write(batch_payload.tobytes())
                        offsets[
                            anchor_start:anchor_end + 1
                        ] = batch_offsets + payload_start
                        encodings[
                            anchor_start:anchor_end
                        ] = batch_encodings
                        if save_frontier_work:
                            frontier_work[
                                anchor_start:anchor_end
                            ] = batch_frontier_work

                        # Commit the payload and row metadata before advancing
                        # the checkpoint. A killed process can safely discard
                        # anything beyond this durable boundary on restart.
                        payload_file.flush()
                        os.fsync(payload_file.fileno())
                        offsets.flush()
                        encodings.flush()
                        if save_frontier_work:
                            frontier_work.flush()
                        if incomplete_records:
                            with incomplete_manifest_path.open(
                                "a", encoding="utf-8"
                            ) as manifest_file:
                                for record in incomplete_records:
                                    manifest_file.write(
                                        json.dumps(record) + "\n"
                                    )
                                manifest_file.flush()
                                os.fsync(manifest_file.fileno())
                        write_checkpoint(
                            checkpoint_path,
                            hop,
                            anchor_end,
                            payload_file.tell(),
                        )
                        progress.update(anchor_end - anchor_start)

            del offsets
            del encodings
            if save_frontier_work:
                del frontier_work
            os.replace(partial_offsets_path, self.hop_offsets_paths[hop])
            os.replace(partial_encodings_path, self.hop_encoding_paths[hop])
            os.replace(partial_payload_path, self.hop_payload_paths[hop])
            checkpoint_path.unlink(missing_ok=True)

        def construct_shared_igraph() -> None:
            """Construct the read-only long-lane graph exactly once."""
            global _IGRAPH_TRAVERSAL_GRAPH
            if (
                _IGRAPH_TRAVERSAL_GRAPH is not None
                and _IGRAPH_TRAVERSAL_GRAPH.vcount()
                == indexed_anchor_count
            ):
                return
            hop_one_offsets = np.load(
                self.hop_1_offsets_path, mmap_mode="r"
            )
            hop_one_nodes = np.load(
                self.hop_1_nodes_path, mmap_mode="r"
            )
            graph = ig.Graph(n=indexed_anchor_count, directed=False)
            edge_buffer = []
            with tqdm(
                total=indexed_anchor_count,
                desc="Constructing shared hybrid iGraph",
                unit="anchor",
                dynamic_ncols=True,
            ) as graph_progress:
                for anchor in range(indexed_anchor_count):
                    start = int(hop_one_offsets[anchor])
                    end = int(hop_one_offsets[anchor + 1])
                    edge_buffer.extend(
                        (anchor, int(neighbor))
                        for neighbor in hop_one_nodes[start:end]
                        if anchor < int(neighbor)
                    )
                    if len(edge_buffer) >= 500_000:
                        graph.add_edges(edge_buffer)
                        edge_buffer.clear()
                    graph_progress.update(1)
            if edge_buffer:
                graph.add_edges(edge_buffer)
            _IGRAPH_TRAVERSAL_GRAPH = graph
            print(
                f"Shared hybrid iGraph ready: "
                f"{graph.vcount():,} nodes, {graph.ecount():,} edges"
            )

        def save_hybrid_phase(
            hop: int,
            previous_offsets,
            previous_encodings,
            previous_payload,
            two_back_offsets,
            two_back_encodings,
            two_back_payload,
            two_back_is_hop_one: bool,
        ) -> None:
            """Run cost-sorted compiled and iGraph lanes concurrently."""
            if completed_phase_is_valid(hop):
                print(
                    f"Reusing completed exact hop {hop}: "
                    f"{indexed_anchor_count:,} anchors"
                )
                return

            partial_offsets_path = self.output_dir / (
                f"hop_{hop}_offsets.partial.npy"
            )
            partial_encodings_path = self.output_dir / (
                f"hop_{hop}_encoding.partial.npy"
            )
            partial_payload_path = self.output_dir / (
                f"hop_{hop}_payload.partial.bin"
            )
            checkpoint_path = self.output_dir / (
                f"hop_{hop}_checkpoint.json"
            )
            prefix_anchor = 0
            prefix_payload_bytes = 0
            if checkpoint_path.exists():
                checkpoint = json.loads(
                    checkpoint_path.read_text(encoding="utf-8")
                )
                if (
                    checkpoint.get("algorithm") == "adaptive_hybrid_v2"
                    and int(checkpoint.get("hop", -1)) == hop
                    and partial_offsets_path.exists()
                    and partial_encodings_path.exists()
                    and partial_payload_path.exists()
                ):
                    prefix_anchor = int(checkpoint["next_anchor"])
                    prefix_payload_bytes = int(
                        checkpoint["payload_bytes"]
                    )

            stage_payload_path = self.output_dir / (
                f"hop_{hop}_hybrid_stage.bin"
            )
            stage_starts_path = self.output_dir / (
                f"hop_{hop}_hybrid_starts.npy"
            )
            stage_lengths_path = self.output_dir / (
                f"hop_{hop}_hybrid_lengths.npy"
            )
            stage_encodings_path = self.output_dir / (
                f"hop_{hop}_hybrid_encodings.npy"
            )
            stage_complete_path = self.output_dir / (
                f"hop_{hop}_hybrid_complete.npy"
            )
            stage_meta_path = self.output_dir / (
                f"hop_{hop}_hybrid.json"
            )
            resume_stage = False
            if stage_meta_path.exists():
                stage_meta = json.loads(
                    stage_meta_path.read_text(encoding="utf-8")
                )
                resume_stage = (
                    int(stage_meta.get("hop", -1)) == hop
                    and int(stage_meta.get("node_count", -1))
                    == indexed_anchor_count
                    and int(stage_meta.get("prefix_anchor", -1))
                    == prefix_anchor
                    and all(
                        path.exists()
                        for path in (
                            stage_payload_path,
                            stage_starts_path,
                            stage_lengths_path,
                            stage_encodings_path,
                            stage_complete_path,
                        )
                    )
                )
            mode = "r+" if resume_stage else "w+"
            stage_starts = np.lib.format.open_memmap(
                stage_starts_path,
                mode=mode,
                dtype=np.uint64,
                shape=(indexed_anchor_count,),
            )
            stage_lengths = np.lib.format.open_memmap(
                stage_lengths_path,
                mode=mode,
                dtype=np.uint64,
                shape=(indexed_anchor_count,),
            )
            stage_encodings = np.lib.format.open_memmap(
                stage_encodings_path,
                mode=mode,
                dtype=np.uint8,
                shape=(indexed_anchor_count,),
            )
            stage_complete = np.lib.format.open_memmap(
                stage_complete_path,
                mode=mode,
                dtype=np.uint8,
                shape=(indexed_anchor_count,),
            )
            if not resume_stage:
                stage_starts[:] = 0
                stage_lengths[:] = 0
                stage_encodings[:] = 0
                stage_complete[:] = 0
                with stage_payload_path.open("wb"):
                    pass
                stage_meta_path.write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "hop": hop,
                            "node_count": indexed_anchor_count,
                            "prefix_anchor": prefix_anchor,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            threshold = routing_threshold
            remaining = np.arange(
                prefix_anchor,
                indexed_anchor_count,
                dtype=np.uint32,
            )
            remaining = remaining[
                stage_complete[remaining] == 0
            ]
            print(
                f"Routing exact hop {hop} anchors during processing; "
                f"threshold={threshold:,} adjacency visits"
            )
            construct_shared_igraph()
            quick_workers = max(
                1, worker_count - self.igraph_long_workers
            )
            routing_batch_size = max(256, worker_count * 64)
            long_queue = deque()
            completed_before = int(
                np.count_nonzero(stage_complete[prefix_anchor:])
            )

            with stage_payload_path.open("ab+") as stage_payload:
                def commit_rows(rows) -> int:
                    """Durably append completed out-of-order rows."""
                    new_rows = [
                        row for row in rows
                        if not stage_complete[row[0]]
                    ]
                    if not new_rows:
                        return 0
                    starts = []
                    for anchor, encoding, payload, _ in new_rows:
                        starts.append(stage_payload.tell())
                        stage_payload.write(payload)
                    stage_payload.flush()
                    os.fsync(stage_payload.fileno())
                    for (
                        anchor,
                        encoding,
                        payload,
                        _,
                    ), payload_start in zip(new_rows, starts):
                        stage_starts[anchor] = payload_start
                        stage_lengths[anchor] = len(payload)
                        stage_encodings[anchor] = encoding
                        stage_complete[anchor] = 1
                    stage_starts.flush()
                    stage_lengths.flush()
                    stage_encodings.flush()
                    stage_complete.flush()
                    return len(new_rows)

                with tqdm(
                    total=indexed_anchor_count,
                    initial=prefix_anchor + completed_before,
                    desc=f"Building exact hop {hop} hybrid",
                    unit="anchor",
                    dynamic_ncols=True,
                ) as progress, ThreadPoolExecutor(
                    # Keep every worker available to iGraph after the fast
                    # lane drains. Submission below limits iGraph while the
                    # compiled lane is still active.
                    max_workers=worker_count
                ) as long_executor:
                    pending = set()
                    fast_lane_active = True

                    def fill_long_lane() -> None:
                        active_long_workers = (
                            min(self.igraph_long_workers, worker_count)
                            if fast_lane_active
                            else worker_count
                        )
                        while (
                            long_queue
                            and len(pending)
                            < active_long_workers
                        ):
                            anchor = long_queue.popleft()
                            if stage_complete[anchor]:
                                continue
                            pending.add(
                                long_executor.submit(
                                    _encode_igraph_exact_hop,
                                    anchor,
                                    hop,
                                    indexed_anchor_count,
                                    bytes_per_row,
                                )
                            )

                    def collect_long(wait_for_one=False) -> None:
                        if not pending:
                            return
                        done, _ = wait(
                            pending,
                            timeout=None if wait_for_one else 0,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in done:
                            pending.remove(future)
                            progress.update(
                                commit_rows([future.result()])
                            )
                        fill_long_lane()

                    for batch_start in range(
                        0, len(remaining), routing_batch_size
                    ):
                        routing_anchors = np.ascontiguousarray(
                            remaining[
                                batch_start:
                                batch_start + routing_batch_size
                            ],
                            dtype=np.uint32,
                        )
                        slow_flags = classify_frontiers_by_degree(
                            adjacency_offsets,
                            previous_offsets,
                            previous_encodings,
                            previous_payload,
                            routing_anchors,
                            threshold,
                            quick_workers,
                        ).astype(bool, copy=False)
                        long_queue.extend(
                            int(anchor)
                            for anchor in routing_anchors[slow_flags]
                        )
                        anchors = np.ascontiguousarray(
                            routing_anchors[~slow_flags],
                            dtype=np.uint32,
                        )
                        fill_long_lane()
                        collect_long()
                        if not len(anchors):
                            continue
                        result = (
                            build_next_hop_from_global_previous_indices(
                                adjacency_offsets,
                                adjacency_neighbors,
                                previous_offsets,
                                previous_encodings,
                                previous_payload,
                                two_back_offsets,
                                two_back_encodings,
                                two_back_payload,
                                two_back_is_hop_one,
                                heavy_row_by_node,
                                heavy_adjacency_bitmaps,
                                anchors,
                                quick_workers,
                                self.quick_time_limit_seconds,
                            )
                        )
                        batch_offsets, batch_encoding, payload = result[:3]
                        timed_out = result[3]
                        rows = []
                        for local_index, anchor in enumerate(anchors):
                            anchor = int(anchor)
                            if timed_out[local_index]:
                                tqdm.write(
                                    f"Quick timeout for original ID "
                                    f"{self.entity_ids[anchor]} "
                                    f"(dense {anchor}); moved to iGraph"
                                )
                                long_queue.append(anchor)
                                continue
                            start = int(batch_offsets[local_index])
                            end = int(batch_offsets[local_index + 1])
                            rows.append(
                                (
                                    anchor,
                                    int(batch_encoding[local_index]),
                                    payload[start:end].tobytes(),
                                    int(result[5][local_index]),
                                )
                            )
                        progress.update(commit_rows(rows))
                        collect_long()
                        fill_long_lane()

                    # No compiled batches remain. Give the entire worker
                    # allocation to queued iGraph anchors instead of leaving
                    # the former fast-lane workers idle.
                    fast_lane_active = False
                    fill_long_lane()
                    while pending or long_queue:
                        fill_long_lane()
                        collect_long(wait_for_one=True)

            if not np.all(stage_complete[prefix_anchor:]):
                raise RuntimeError("Hybrid phase ended with missing anchors")

            merge_offsets_path = self.output_dir / (
                f"hop_{hop}_offsets.merge.npy"
            )
            merge_encodings_path = self.output_dir / (
                f"hop_{hop}_encoding.merge.npy"
            )
            merge_payload_path = self.output_dir / (
                f"hop_{hop}_payload.merge.bin"
            )
            merge_offsets = np.lib.format.open_memmap(
                merge_offsets_path,
                mode="w+",
                dtype=np.uint64,
                shape=(indexed_anchor_count + 1,),
            )
            merge_encodings = np.lib.format.open_memmap(
                merge_encodings_path,
                mode="w+",
                dtype=np.uint8,
                shape=(indexed_anchor_count,),
            )
            with merge_payload_path.open("wb") as output_payload:
                if prefix_anchor:
                    old_offsets = np.load(
                        partial_offsets_path, mmap_mode="r"
                    )
                    old_encodings = np.load(
                        partial_encodings_path, mmap_mode="r"
                    )
                    merge_offsets[:prefix_anchor + 1] = old_offsets[
                        :prefix_anchor + 1
                    ]
                    merge_encodings[:prefix_anchor] = old_encodings[
                        :prefix_anchor
                    ]
                    with partial_payload_path.open("rb") as old_payload:
                        remaining_prefix_bytes = prefix_payload_bytes
                        while remaining_prefix_bytes:
                            chunk = old_payload.read(
                                min(
                                    remaining_prefix_bytes,
                                    16 * 1024 * 1024,
                                )
                            )
                            if not chunk:
                                raise RuntimeError(
                                    "Existing checkpoint payload is "
                                    "shorter than its committed size"
                                )
                            output_payload.write(chunk)
                            remaining_prefix_bytes -= len(chunk)
                with stage_payload_path.open("rb") as stage_input:
                    for anchor in tqdm(
                        range(prefix_anchor, indexed_anchor_count),
                        desc=f"Merging exact hop {hop}",
                        unit="anchor",
                        dynamic_ncols=True,
                    ):
                        stage_input.seek(int(stage_starts[anchor]))
                        payload = stage_input.read(
                            int(stage_lengths[anchor])
                        )
                        output_payload.write(payload)
                        merge_encodings[anchor] = stage_encodings[anchor]
                        merge_offsets[anchor + 1] = output_payload.tell()
                output_payload.flush()
                os.fsync(output_payload.fileno())
            merge_offsets.flush()
            merge_encodings.flush()
            del merge_offsets
            del merge_encodings
            del stage_starts
            del stage_lengths
            del stage_encodings
            del stage_complete
            os.replace(merge_offsets_path, self.hop_offsets_paths[hop])
            os.replace(merge_encodings_path, self.hop_encoding_paths[hop])
            os.replace(merge_payload_path, self.hop_payload_paths[hop])
            for path in (
                partial_offsets_path,
                partial_encodings_path,
                partial_payload_path,
                checkpoint_path,
                stage_payload_path,
                stage_starts_path,
                stage_lengths_path,
                stage_encodings_path,
                stage_complete_path,
                stage_meta_path,
                self.output_dir / f"hop_{hop}_incomplete.jsonl",
            ):
                path.unlink(missing_ok=True)

        # Every later phase uses the same recurrence. Hop one is exposed in
        # the same adaptive sparse-row shape in memory, so hop two requires no
        # separate algorithm or construction path.
        empty_offsets = np.empty(0, dtype=np.uint64)
        empty_encodings = np.empty(0, dtype=np.uint8)
        empty_payload = np.empty(0, dtype=np.uint8)
        hop_one_node_offsets = np.load(
            self.hop_1_offsets_path, mmap_mode="r"
        )
        hop_one_byte_offsets = np.asarray(
            hop_one_node_offsets * np.uint64(4), dtype=np.uint64
        )
        hop_one_encodings = np.zeros(
            indexed_anchor_count, dtype=np.uint8
        )
        hop_one_nodes = np.load(self.hop_1_nodes_path, mmap_mode="r")
        hop_one_payload = hop_one_nodes.view(np.uint8)

        for hop in range(2, self.max_hop + 1):
            previous_hop = hop - 1
            two_back_hop = hop - 2
            if previous_hop == 1:
                previous_offsets = hop_one_byte_offsets
                previous_encodings = hop_one_encodings
                previous_payload = hop_one_payload
            else:
                previous_offsets = np.load(
                    self.hop_offsets_paths[previous_hop], mmap_mode="r"
                )
                previous_encodings = np.load(
                    self.hop_encoding_paths[previous_hop], mmap_mode="r"
                )
                previous_payload_path = self.hop_payload_paths[previous_hop]
                if previous_payload_path.stat().st_size:
                    previous_payload = np.memmap(
                        previous_payload_path, mode="r", dtype=np.uint8
                    )
                else:
                    previous_payload = empty_payload

            # For hop two and hop three, removing the anchor's adjacency row
            # removes every node already known at distance one. For all later
            # hops the ordinary h-2 exact frontier is removed.
            two_back_is_hop_one = two_back_hop <= 1
            if two_back_is_hop_one:
                two_back_offsets = empty_offsets
                two_back_encodings = empty_encodings
                two_back_payload = empty_payload
            else:
                two_back_offsets = np.load(
                    self.hop_offsets_paths[two_back_hop], mmap_mode="r"
                )
                two_back_encodings = np.load(
                    self.hop_encoding_paths[two_back_hop], mmap_mode="r"
                )
                two_back_payload_path = self.hop_payload_paths[two_back_hop]
                if two_back_payload_path.stat().st_size:
                    two_back_payload = np.memmap(
                        two_back_payload_path, mode="r", dtype=np.uint8
                    )
                else:
                    two_back_payload = empty_payload

            save_hybrid_phase(
                hop,
                previous_offsets,
                previous_encodings,
                previous_payload,
                two_back_offsets,
                two_back_encodings,
                two_back_payload,
                two_back_is_hop_one,
            )

            del previous_offsets
            del previous_encodings
            if previous_hop != 1 and isinstance(previous_payload, np.memmap):
                del previous_payload
            if not two_back_is_hop_one:
                del two_back_offsets
                del two_back_encodings
                if isinstance(two_back_payload, np.memmap):
                    del two_back_payload

        del hop_one_node_offsets
        del hop_one_byte_offsets
        del hop_one_encodings
        del hop_one_payload
        del hop_one_nodes

        del adjacency_offsets
        del adjacency_neighbors
        del heavy_row_by_node
        del heavy_adjacency_bitmaps

        later_hop_seconds = time.perf_counter() - later_hop_started

        for hop in self.hop_paths:
            print(
                f"Built exact hop {hop}: "
                f"{indexed_anchor_count:,} anchors"
            )

        print(
            "Compiled later-hop construction: "
            f"{later_hop_seconds:.2f}s wall time with "
            f"{worker_count} worker(s)"
        )

    def _save_metadata(
        self,
        indexed_anchor_count: int,
        bytes_per_row: int,
        type_names: list[str],
        build_seconds: float,
    ) -> dict:
        """Save information required to read the sidecar."""
        incomplete_anchor_counts = {}
        incomplete_manifest_files = {}
        total_incomplete_rows = 0
        for hop in range(2, self.max_hop + 1):
            manifest_path = self.output_dir / (
                f"hop_{hop}_incomplete.jsonl"
            )
            if not manifest_path.exists():
                continue

            # A crash after the manifest fsync but before the checkpoint
            # replacement can replay a batch. Count unique anchors so the
            # metadata remains accurate even if the JSONL contains retries.
            incomplete_anchors = set()
            with manifest_path.open("r", encoding="utf-8") as manifest:
                for line in manifest:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    incomplete_anchors.add(
                        int(record["dense_anchor_index"])
                    )
            if incomplete_anchors:
                hop_key = str(hop)
                incomplete_anchor_counts[hop_key] = len(
                    incomplete_anchors
                )
                incomplete_manifest_files[hop_key] = (
                    manifest_path.name
                )
                total_incomplete_rows += len(incomplete_anchors)

        metadata = {
            "format_version": 4,
            "node_count": len(self.entity_ids),
            "edge_count": len(self.relationships),
            "indexed_anchor_count": (
                indexed_anchor_count
            ),
            "bytes_per_row": bytes_per_row,
            "max_hop": self.max_hop,
            "later_hop_storage": "adaptive_sparse_or_bitmap",
            "build_workers": self.build_workers,
            "hybrid_igraph": self.hybrid_igraph,
            "igraph_long_workers": self.igraph_long_workers,
            "igraph_work_threshold": self.igraph_work_threshold,
            "quick_time_limit_seconds": self.quick_time_limit_seconds,
            "anchor_time_limit_seconds": (
                self.anchor_time_limit_seconds
            ),
            "contains_partial_rows": total_incomplete_rows > 0,
            "incomplete_row_count": total_incomplete_rows,
            "incomplete_anchor_counts_by_hop": (
                incomplete_anchor_counts
            ),
            "incomplete_manifest_files_by_hop": (
                incomplete_manifest_files
            ),
            "entity_types": type_names,
            "build_seconds": build_seconds,
        }

        self.metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n"
        )

        return metadata

    def build(self) -> dict:
        """Construct all configured bitmap sidecars."""
        if not self.entity_ids:
            raise ValueError(
                "No entities have been added"
            )

        started = time.perf_counter()

        node_count = len(self.entity_ids)
        indexed_anchor_count = node_count
        bytes_per_row = (
            node_count + 7
        ) // 8

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # One contiguous integer edge array feeds both the native iGraph and
        # the compiled hop-one constructor.
        dense_edges = np.empty(
            (len(self.relationships), 2), dtype=np.uint32
        )
        for edge_index, (source_id, destination_id) in enumerate(
            tqdm(
                self.relationships.values(),
                total=len(self.relationships),
                desc="Preparing integer graph edges",
                unit="edge",
                dynamic_ncols=True,
            )
        ):
            dense_edges[edge_index, 0] = self.entity_id_to_dense_id[
                source_id
            ]
            dense_edges[edge_index, 1] = self.entity_id_to_dense_id[
                destination_id
            ]
        self._dense_edges = dense_edges

        # The hybrid long lane owns one shared iGraph for the entire build.
        if self.hybrid_igraph and self.traversal_graph is None:
            print("Constructing shared hybrid iGraph in one bulk call")
            traversal_graph = ig.Graph(
                n=node_count,
                edges=dense_edges,
                directed=False,
            )
            self.traversal_graph = traversal_graph
            print(
                "Shared hybrid iGraph ready before index phases: "
                f"{traversal_graph.vcount():,} nodes, "
                f"{traversal_graph.ecount():,} edges"
            )

        # Metadata is the completion marker and is recreated only after every
        # sidecar has been written successfully.
        if self.metadata_path.exists():
            self.metadata_path.unlink()

        # Phase 1: Save entity IDs in dense-index order.
        self._save_entity_ids()

        # Phase 2: Build degrees, CSR adjacency, and exact hop one together.
        # Hop one is complete and unconditional; routing begins only when a
        # later hop expands this stored frontier.
        (
            _,
            adjacency_offsets,
            adjacency_neighbors,
        ) = self._save_adjacency()
        del self._dense_edges

        # Phase 3: Save entity-type membership bitmaps.
        type_names = self._save_entity_types(
            bytes_per_row,
        )

        # Phase 4: Derive every later exact hop from one-hop adjacency.
        self._save_later_hops(
            indexed_anchor_count,
            bytes_per_row,
        )

        # Phase 5: Record index dimensions and supported hop count.
        metadata = self._save_metadata(
            indexed_anchor_count=(
                indexed_anchor_count
            ),
            bytes_per_row=bytes_per_row,
            type_names=type_names,
            build_seconds=(
                time.perf_counter() - started
            ),
        )

        print(
            f"Exact-hop bitmap index written to: "
            f"{self.output_dir}"
        )
        print(
            f"Build time: "
            f"{metadata['build_seconds']:.2f}s"
        )

        return metadata

