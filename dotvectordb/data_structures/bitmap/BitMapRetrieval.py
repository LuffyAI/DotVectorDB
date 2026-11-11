from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class OneHopBitmapRetriever:
    """Retrieve typed one-hop adjacency for iterative path construction."""

    def __init__(
        self,
        index_path: str | Path,
    ) -> None:
        """Open the one-hop adjacency and entity-type sidecars.

        Time complexity: O(V + T).
        The NumPy arrays are memory-mapped, but constructing the external-ID
        lookup requires visiting all V entity IDs.
        """
        started = time.perf_counter()

        self.index_path = Path(index_path).resolve()

        self.metadata = json.loads(
            (self.index_path / "metadata.json").read_text()
        )

        # Dense ID -> database entity ID.
        self.entity_ids = np.load(
            self.index_path / "entity_ids.npy",
            mmap_mode="r",
        )

        # CSR adjacency arrays.
        self.adjacency_offsets = np.load(
            self.index_path / "adjacency_offsets.npy",
            mmap_mode="r",
        )

        self.adjacency_neighbors = np.load(
            self.index_path / "adjacency_neighbors.npy",
            mmap_mode="r",
        )

        self.adjacency_relationship_ids = np.load(
            self.index_path
            / "adjacency_relationship_ids.npy",
            mmap_mode="r",
        )

        # One packed entity-membership bitmap per type.
        self.entity_type_bitmaps = np.load(
            self.index_path / "entity_types.npy",
            mmap_mode="r",
        )

        # Database entity ID -> dense ID.
        self.entity_id_to_dense_id = {
            int(entity_id): dense_id
            for dense_id, entity_id in enumerate(
                self.entity_ids
            )
        }

        # Entity-type name -> entity_types.npy row.
        self.entity_type_to_row = {
            entity_type: row
            for row, entity_type in enumerate(
                self.metadata["entity_types"]
            )
        }

        self.load_seconds = (
            time.perf_counter() - started
        )

    @property
    def entity_types(self) -> tuple[str, ...]:
        """Return every entity type represented by the index.

        Time complexity: O(T), because a new tuple is constructed.
        """
        return tuple(self.metadata["entity_types"])

    @staticmethod
    def _normalise_types(
        entity_types,
    ) -> tuple[str, ...]:
        """Normalize one or more requested entity types.

        Time complexity: O(K).
        """
        if entity_types is None:
            return ()

        if isinstance(entity_types, str):
            return (entity_types,)

        return tuple(
            dict.fromkeys(
                str(entity_type)
                for entity_type in entity_types
            )
        )

    @staticmethod
    def _bitmap_membership(
        bitmap: np.ndarray,
        dense_ids: np.ndarray,
    ) -> np.ndarray:
        """Test whether dense IDs are present in a packed bitmap.

        Time complexity: O(D), where D is the number of supplied dense IDs.
        An individual membership test is O(1).
        """
        if dense_ids.size == 0:
            return np.zeros(
                0,
                dtype=bool,
            )

        return (
            bitmap[dense_ids >> 3]
            & np.left_shift(
                np.uint8(1),
                (dense_ids & 7).astype(np.uint8),
            )
        ) != 0

    def _type_bitmap(
        self,
        entity_types,
        match_all_types: bool = False,
    ) -> np.ndarray | None:
        """Combine the packed rows for the requested entity types.

        Time complexity: O(KB), where B is the number of bytes per bitmap.
        """
        entity_types = self._normalise_types(
            entity_types
        )

        if not entity_types:
            return None

        if match_all_types:
            bitmap = np.full(
                int(self.metadata["bytes_per_row"]),
                255,
                dtype=np.uint8,
            )

            for entity_type in entity_types:
                row = self.entity_type_to_row.get(
                    entity_type
                )

                if row is None:
                    bitmap[:] = 0
                    break

                bitmap &= self.entity_type_bitmaps[
                    row
                ]

            return bitmap

        bitmap = np.zeros(
            int(self.metadata["bytes_per_row"]),
            dtype=np.uint8,
        )

        for entity_type in entity_types:
            row = self.entity_type_to_row.get(
                entity_type
            )

            if row is not None:
                bitmap |= self.entity_type_bitmaps[
                    row
                ]

        return bitmap

    def entity_types_for_id(
        self,
        entity_id: int,
    ) -> tuple[str, ...]:
        """Return every type assigned to one entity.

        Time complexity: O(T).
        Each individual type-membership test is O(1).
        """
        dense_id = self.entity_id_to_dense_id[
            int(entity_id)
        ]

        byte_index = dense_id >> 3
        bit_mask = np.uint8(
            1 << (dense_id & 7)
        )

        return tuple(
            entity_type
            for entity_type, row
            in self.entity_type_to_row.items()
            if (
                self.entity_type_bitmaps[
                    row,
                    byte_index,
                ]
                & bit_mask
            )
        )

    def entity_type_groups_by_id(
        self,
        entity_ids,
    ) -> dict[int, tuple[str, ...]]:
        """Return the type group assigned to each supplied entity.

        Time complexity: O(MT), where M is the number of unique entities.
        """
        unique_entity_ids = list(
            dict.fromkeys(
                int(entity_id)
                for entity_id in entity_ids
                if int(entity_id)
                in self.entity_id_to_dense_id
            )
        )

        return {
            entity_id: self.entity_types_for_id(
                entity_id
            )
            for entity_id in unique_entity_ids
        }

    def type_group_counts_for_ids(
        self,
        entity_ids,
    ) -> dict[tuple[str, ...], int]:
        """Count entities belonging to each type group.

        Time complexity: O(MT), where M is the number of supplied entities.
        """
        entity_ids = [
            int(entity_id)
            for entity_id in entity_ids
        ]

        type_groups = (
            self.entity_type_groups_by_id(
                entity_ids
            )
        )

        counts: dict[
            tuple[str, ...],
            int,
        ] = {}

        for entity_id in entity_ids:
            type_group = type_groups.get(
                entity_id,
                (),
            )

            if type_group:
                counts[type_group] = (
                    counts.get(type_group, 0) + 1
                )

        return counts

    def entity_ids_for_types(
        self,
        entity_types,
        match_all_types: bool = False,
    ) -> np.ndarray:
        """Return all entities carrying the requested types.

        Time complexity: O(KB + V).
        Combining the type bitmaps costs O(KB), while materializing all
        matching entity IDs requires unpacking V bits.
        """
        type_bitmap = self._type_bitmap(
            entity_types,
            match_all_types,
        )

        if type_bitmap is None:
            return self.entity_ids

        dense_ids = np.flatnonzero(
            np.unpackbits(
                type_bitmap,
                bitorder="little",
            )[: len(self.entity_ids)]
        )

        return self.entity_ids[dense_ids]

    def relationships_from_frontier(
        self,
        frontier_entity_ids,
        entity_types=None,
        match_all_types: bool = False,
        exclude_entity_ids=None,
    ) -> dict[int, tuple[int, int]]:
        """Return typed one-hop relationships leaving a frontier.

        Time complexity: O(KB + X + A + R).

        K is the number of requested types, X is the number of excluded
        entities, A is the total number of adjacency entries belonging to
        the frontier, and R is the number of retained relationships.
        """
        type_bitmap = self._type_bitmap(
            entity_types,
            match_all_types,
        )

        excluded_dense_ids = {
            self.entity_id_to_dense_id[
                int(entity_id)
            ]
            for entity_id in (
                exclude_entity_ids or ()
            )
            if int(entity_id)
            in self.entity_id_to_dense_id
        }

        relationships: dict[
            int,
            tuple[int, int],
        ] = {}

        for frontier_entity_id in frontier_entity_ids:
            frontier_entity_id = int(
                frontier_entity_id
            )

            frontier_dense_id = (
                self.entity_id_to_dense_id.get(
                    frontier_entity_id
                )
            )

            if frontier_dense_id is None:
                continue

            # Accessing the CSR boundaries is O(1).
            start = int(
                self.adjacency_offsets[
                    frontier_dense_id
                ]
            )
            end = int(
                self.adjacency_offsets[
                    frontier_dense_id + 1
                ]
            )

            # Inspecting the adjacency slice is O(degree(frontier)).
            neighbor_dense_ids = np.asarray(
                self.adjacency_neighbors[
                    start:end
                ],
                dtype=np.int64,
            )

            if neighbor_dense_ids.size == 0:
                continue

            keep = np.ones(
                neighbor_dense_ids.size,
                dtype=bool,
            )

            if excluded_dense_ids:
                keep &= ~np.isin(
                    neighbor_dense_ids,
                    tuple(excluded_dense_ids),
                )

            if type_bitmap is not None:
                keep &= self._bitmap_membership(
                    type_bitmap,
                    neighbor_dense_ids,
                )

            retained_positions = (
                np.flatnonzero(keep) + start
            )

            for position in retained_positions:
                relationship_id = int(
                    self.adjacency_relationship_ids[
                        position
                    ]
                )

                endpoint_dense_id = int(
                    self.adjacency_neighbors[
                        position
                    ]
                )

                endpoint_entity_id = int(
                    self.entity_ids[
                        endpoint_dense_id
                    ]
                )

                relationships[
                    relationship_id
                ] = (
                    frontier_entity_id,
                    endpoint_entity_id,
                )

        return relationships

    def expansions_from_paths(
        self,
        paths,
        entity_types=None,
        match_all_types: bool = False,
    ) -> list[dict]:
        """Expand paths by one hop without allowing path-local loops.

        Time complexity: O(PKB + L + A + R).

        P is the number of input paths, L is the total number of entities
        already contained across those paths, A is the total adjacency
        inspected, and R is the number of resulting expansions.
        """
        expansions = []

        for path in paths:
            entity_path = tuple(
                int(entity_id)
                for entity_id in path.get(
                    "entity_ids",
                    (),
                )
            )

            if not entity_path:
                continue

            frontier_entity_id = entity_path[-1]

            relationships = (
                self.relationships_from_frontier(
                    frontier_entity_ids=[
                        frontier_entity_id
                    ],
                    entity_types=entity_types,
                    match_all_types=(
                        match_all_types
                    ),
                    # Exclude the anchor and every entity already in
                    # this individual path.
                    exclude_entity_ids=entity_path,
                )
            )

            for relationship_id, (
                source_entity_id,
                endpoint_entity_id,
            ) in relationships.items():
                expansions.append({
                    "relationship_id": (
                        relationship_id
                    ),
                    "frontier_entity_id": (
                        source_entity_id
                    ),
                    "endpoint_entity_id": (
                        endpoint_entity_id
                    ),
                    "prefix": path,
                })

        return expansions

    def adjacent_type_groups(
        self,
        frontier_entity_ids,
        exclude_entity_ids=None,
    ) -> list[tuple[str, ...]]:
        """Return distinct type groups reachable in one hop.

        Time complexity: O(A + RT + G log G).

        A is the adjacency inspected, R is the number of retained endpoint
        entities, and G is the number of distinct type groups.
        """
        relationships = (
            self.relationships_from_frontier(
                frontier_entity_ids=(
                    frontier_entity_ids
                ),
                exclude_entity_ids=(
                    exclude_entity_ids
                ),
            )
        )

        type_groups = (
            self.entity_type_groups_by_id(
                endpoint_entity_id
                for _, endpoint_entity_id
                in relationships.values()
            )
        )

        return sorted(
            {
                type_group
                for type_group
                in type_groups.values()
                if type_group
            }
        )


class ExactHopBitmapRetriever:
    """Retrieve exact-hop typed endpoints from precomputed bitmaps."""

    def __init__(
        self,
        index_path: str | Path,
    ) -> None:
        """Open the adjacency, type, and exact-hop sidecars.

        Time complexity: O(V + T + H).
        Most NumPy arrays are memory-mapped; O(V) comes from constructing
        the database-ID-to-dense-ID dictionary.
        """
        started = time.perf_counter()

        self.index_path = Path(index_path).resolve()

        # Exact-hop retrieval still uses adjacency to recover the actual
        # relationships connecting a frontier to an allowed endpoint.
        self.one_hop = OneHopBitmapRetriever(
            self.index_path
        )

        self.metadata = self.one_hop.metadata
        self.entity_ids = self.one_hop.entity_ids
        self.entity_id_to_dense_id = (
            self.one_hop.entity_id_to_dense_id
        )
        self.entity_type_bitmaps = (
            self.one_hop.entity_type_bitmaps
        )
        self.entity_type_to_row = (
            self.one_hop.entity_type_to_row
        )

        # Exact hop one is stored as a compact CSR-style list.
        self.hop_1_offsets = np.load(
            self.index_path / "hop_1_offsets.npy",
            mmap_mode="r",
        )

        self.hop_1_nodes = np.load(
            self.index_path / "hop_1_nodes.npy",
            mmap_mode="r",
        )

        # Hops two through max_hop use adaptive sparse-or-packed rows.
        self.exact_hop_offsets = {
            hop: np.load(
                self.index_path / f"hop_{hop}_offsets.npy",
                mmap_mode="r",
            )
            for hop in range(
                2,
                int(self.metadata["max_hop"]) + 1,
            )
        }
        self.exact_hop_encodings = {
            hop: np.load(
                self.index_path / f"hop_{hop}_encoding.npy",
                mmap_mode="r",
            )
            for hop in range(2, int(self.metadata["max_hop"]) + 1)
        }
        self.exact_hop_payloads = {}
        for hop in range(2, int(self.metadata["max_hop"]) + 1):
            payload_path = self.index_path / f"hop_{hop}_payload.bin"
            if payload_path.stat().st_size:
                payload = np.memmap(
                    payload_path,
                    mode="r",
                    dtype=np.uint8,
                )
            else:
                payload = np.empty(0, dtype=np.uint8)
            self.exact_hop_payloads[hop] = payload
        self.load_seconds = (
            time.perf_counter() - started
        )

    @property
    def entity_types(self) -> tuple[str, ...]:
        """Return all indexed entity types.

        Time complexity: O(T).
        """
        return self.one_hop.entity_types

    @property
    def max_hop(self) -> int:
        """Return the maximum precomputed exact hop.

        Time complexity: O(1).
        """
        return int(self.metadata["max_hop"])

    def _anchor_dense_id(
        self,
        anchor_entity_id: int,
    ) -> int:
        """Translate an anchor ID into its dense bitmap position.

        Time complexity: O(1) average dictionary lookup.
        """
        dense_id = self.entity_id_to_dense_id[
            int(anchor_entity_id)
        ]

        indexed_anchor_count = int(
            self.metadata.get(
                "indexed_anchor_count",
                len(self.entity_ids),
            )
        )

        if dense_id >= indexed_anchor_count:
            raise ValueError(
                f"Entity ID {anchor_entity_id} "
                "was not indexed as an anchor"
            )

        return dense_id

    def _hop_bitmap(
        self,
        anchor_entity_id: int,
        hop: int,
    ) -> np.ndarray:
        """Return the packed exact-hop set for one anchor.

        Time complexity:
        - Hop 1: O(B + N_1), because its stored list is packed on demand.
        - Hop 2 or later: O(1), because the stored bitmap row is returned.
        """
        if hop < 1 or hop > self.max_hop:
            raise ValueError(
                f"hop must be between 1 and "
                f"{self.max_hop}"
            )

        anchor_dense_id = (
            self._anchor_dense_id(
                anchor_entity_id
            )
        )

        if hop > 1:
            start = int(self.exact_hop_offsets[hop][anchor_dense_id])
            end = int(self.exact_hop_offsets[hop][anchor_dense_id + 1])
            payload = self.exact_hop_payloads[hop][start:end]

            if int(self.exact_hop_encodings[hop][anchor_dense_id]) == 1:
                return payload

            bitmap = np.zeros(
                int(self.metadata["bytes_per_row"]),
                dtype=np.uint8,
            )
            dense_ids = np.frombuffer(payload, dtype=np.uint32)
            if dense_ids.size:
                np.bitwise_or.at(
                    bitmap,
                    dense_ids.astype(np.int64) >> 3,
                    np.left_shift(
                        np.uint8(1),
                        (dense_ids & 7).astype(np.uint8),
                    ),
                )
            return bitmap

        bitmap = np.zeros(
            int(self.metadata["bytes_per_row"]),
            dtype=np.uint8,
        )

        start = int(
            self.hop_1_offsets[
                anchor_dense_id
            ]
        )
        end = int(
            self.hop_1_offsets[
                anchor_dense_id + 1
            ]
        )

        dense_ids = np.asarray(
            self.hop_1_nodes[start:end],
            dtype=np.int64,
        )

        if dense_ids.size:
            np.bitwise_or.at(
                bitmap,
                dense_ids >> 3,
                np.left_shift(
                    np.uint8(1),
                    (
                        dense_ids & 7
                    ).astype(np.uint8),
                ),
            )

        return bitmap

    def dense_ids_at_hop(
        self,
        anchor_entity_id: int,
        hop: int,
    ) -> np.ndarray:
        """Materialize dense IDs at exactly the requested hop.

        Time complexity:
        - Hop 1: O(N_1).
        - Hop 2 or later: O(V), because the packed bitmap is unpacked.
        """
        if hop < 1 or hop > self.max_hop:
            raise ValueError(
                f"hop must be between 1 and "
                f"{self.max_hop}"
            )

        anchor_dense_id = (
            self._anchor_dense_id(
                anchor_entity_id
            )
        )

        if hop == 1:
            start = int(
                self.hop_1_offsets[
                    anchor_dense_id
                ]
            )
            end = int(
                self.hop_1_offsets[
                    anchor_dense_id + 1
                ]
            )

            return np.asarray(
                self.hop_1_nodes[start:end],
                dtype=np.int64,
            )

        start = int(self.exact_hop_offsets[hop][anchor_dense_id])
        end = int(self.exact_hop_offsets[hop][anchor_dense_id + 1])
        payload = self.exact_hop_payloads[hop][start:end]

        if int(self.exact_hop_encodings[hop][anchor_dense_id]) == 0:
            return np.frombuffer(payload, dtype=np.uint32).astype(
                np.int64,
                copy=False,
            )

        return np.flatnonzero(
            np.unpackbits(
                payload,
                bitorder="little",
            )[: len(self.entity_ids)]
        )

    def entity_ids_at_hop(
        self,
        anchor_entity_id: int,
        hop: int,
        entity_types=None,
        match_all_types: bool = False,
    ) -> np.ndarray:
        """Return database entity IDs at an exact hop.

        Time complexity:
        - Hop 1: O(KB + N_1).
        - Hop 2 or later: O(KB + V).
        """
        dense_ids = self.dense_ids_at_hop(
            anchor_entity_id,
            hop,
        )

        type_bitmap = (
            self.one_hop._type_bitmap(
                entity_types,
                match_all_types,
            )
        )

        if type_bitmap is not None:
            keep = (
                self.one_hop._bitmap_membership(
                    type_bitmap,
                    dense_ids,
                )
            )

            dense_ids = dense_ids[keep]

        return self.entity_ids[dense_ids]

    def entity_id_set_at_hop(
        self,
        anchor_entity_id: int,
        hop: int,
        entity_types=None,
        match_all_types: bool = False,
    ) -> set[int]:
        """Return the exact-hop result as a Python set.

        Time complexity:
        - Hop 1: O(KB + N_1).
        - Hop 2 or later: O(KB + V + N_h).
        """
        return {
            int(entity_id)
            for entity_id
            in self.entity_ids_at_hop(
                anchor_entity_id=(
                    anchor_entity_id
                ),
                hop=hop,
                entity_types=entity_types,
                match_all_types=(
                    match_all_types
                ),
            )
        }

    def relationships_from_frontier(
        self,
        anchor_entity_id: int,
        frontier_entity_ids,
        exact_hop: int,
        entity_types=None,
        match_all_types: bool = False,
        exclude_entity_ids=None,
    ) -> dict[int, tuple[int, int]]:
        """Return frontier edges entering an exact-hop/type set.

        Time complexity: O((K + 1)B + X + A + R).

        The exact-hop and type bitmaps are intersected once. The method then
        scans only the adjacency entries belonging to the supplied frontier.
        """
        allowed = np.array(
            self._hop_bitmap(
                anchor_entity_id,
                exact_hop,
            ),
            dtype=np.uint8,
            copy=True,
        )

        type_bitmap = (
            self.one_hop._type_bitmap(
                entity_types,
                match_all_types,
            )
        )

        if type_bitmap is not None:
            allowed &= type_bitmap

        excluded_dense_ids = {
            self.entity_id_to_dense_id[
                int(entity_id)
            ]
            for entity_id in (
                exclude_entity_ids or ()
            )
            if int(entity_id)
            in self.entity_id_to_dense_id
        }

        relationships: dict[
            int,
            tuple[int, int],
        ] = {}

        for frontier_entity_id in frontier_entity_ids:
            frontier_entity_id = int(
                frontier_entity_id
            )

            frontier_dense_id = (
                self.entity_id_to_dense_id.get(
                    frontier_entity_id
                )
            )

            if frontier_dense_id is None:
                continue

            start = int(
                self.one_hop.adjacency_offsets[
                    frontier_dense_id
                ]
            )
            end = int(
                self.one_hop.adjacency_offsets[
                    frontier_dense_id + 1
                ]
            )

            neighbor_dense_ids = np.asarray(
                self.one_hop.adjacency_neighbors[
                    start:end
                ],
                dtype=np.int64,
            )

            if neighbor_dense_ids.size == 0:
                continue

            keep = (
                self.one_hop._bitmap_membership(
                    allowed,
                    neighbor_dense_ids,
                )
            )

            if excluded_dense_ids:
                keep &= ~np.isin(
                    neighbor_dense_ids,
                    tuple(excluded_dense_ids),
                )

            retained_positions = (
                np.flatnonzero(keep) + start
            )

            for position in retained_positions:
                relationship_id = int(
                    self.one_hop
                    .adjacency_relationship_ids[
                        position
                    ]
                )

                endpoint_dense_id = int(
                    self.one_hop
                    .adjacency_neighbors[
                        position
                    ]
                )

                endpoint_entity_id = int(
                    self.entity_ids[
                        endpoint_dense_id
                    ]
                )

                relationships[
                    relationship_id
                ] = (
                    frontier_entity_id,
                    endpoint_entity_id,
                )

        return relationships

    def type_counts(
        self,
        anchor_entity_id: int,
        hop: int,
    ) -> dict[str, int]:
        """Count exact-hop entities belonging to each type.

        Time complexity:
        - Hop 1: O(N_1 + TN_1).
        - Hop 2 or later: O(V + TN_h).
        """
        dense_ids = self.dense_ids_at_hop(
            anchor_entity_id,
            hop,
        )

        counts = {}

        for entity_type, row in (
            self.entity_type_to_row.items()
        ):
            membership = (
                self.one_hop._bitmap_membership(
                    self.entity_type_bitmaps[
                        row
                    ],
                    dense_ids,
                )
            )

            counts[entity_type] = int(
                membership.sum()
            )

        return counts

    def reachable_types(
        self,
        anchor_entity_id: int,
        hop: int,
    ) -> dict[str, int]:
        """Return types present at the requested exact hop.

        Time complexity: the same as type_counts(), plus O(T) filtering.
        """
        return {
            entity_type: count
            for entity_type, count
            in self.type_counts(
                anchor_entity_id,
                hop,
            ).items()
            if count
        }

    def hop_overview(
        self,
        anchor_entity_id: int,
    ) -> dict[int, dict[str, int]]:
        """Return type counts for every precomputed hop.

        Time complexity: O(HV + T * sum(N_h)) in the general case.
        """
        return {
            hop: self.reachable_types(
                anchor_entity_id,
                hop,
            )
            for hop in range(
                1,
                self.max_hop + 1,
            )
        }

    def entity_types_for_id(
        self,
        entity_id: int,
    ) -> tuple[str, ...]:
        """Return all types assigned to an entity.

        Time complexity: O(T).
        """
        return self.one_hop.entity_types_for_id(
            entity_id
        )

    def entity_type_groups_by_id(
        self,
        entity_ids,
    ) -> dict[int, tuple[str, ...]]:
        """Return type groups for multiple entities.

        Time complexity: O(MT).
        """
        return (
            self.one_hop.entity_type_groups_by_id(
                entity_ids
            )
        )

    def type_group_counts_for_ids(
        self,
        entity_ids,
    ) -> dict[tuple[str, ...], int]:
        """Count entities belonging to each type group.

        Time complexity: O(MT).
        """
        return (
            self.one_hop.type_group_counts_for_ids(
                entity_ids
            )
        )

    def entity_ids_for_types(
        self,
        entity_types,
        match_all_types: bool = False,
    ) -> np.ndarray:
        """Return all entities carrying requested types.

        Time complexity: O(KB + V).
        """
        return self.one_hop.entity_ids_for_types(
            entity_types,
            match_all_types,
        )

    def adjacent_type_groups(
        self,
        frontier_entity_ids,
        exclude_entity_ids=None,
    ) -> list[tuple[str, ...]]:
        """Return type groups adjacent to a frontier.

        Time complexity: O(A + RT + G log G).
        """
        return self.one_hop.adjacent_type_groups(
            frontier_entity_ids,
            exclude_entity_ids,
        )
