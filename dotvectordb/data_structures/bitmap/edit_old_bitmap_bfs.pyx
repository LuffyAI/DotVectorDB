# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

"""Compiled exact-hop traversal over DotVectorDB's CSR adjacency arrays."""

import numpy as np
cimport numpy as cnp

from cython.parallel cimport prange, threadid
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector

cdef extern from "omp.h":
    double omp_get_wtime() noexcept nogil

cdef extern from *:
    """
    static inline void dotdb_atomic_or_u8(
        unsigned char *address,
        unsigned char mask
    ) {
        __atomic_fetch_or(address, mask, __ATOMIC_RELAXED);
    }
    """
    void dotdb_atomic_or_u8(uint8_t* address, uint8_t mask) noexcept nogil


cnp.import_array()


def build_hop_one_from_edges(
    const uint32_t[:, ::1] edges,
    Py_ssize_t node_count,
    int worker_count,
):
    """Build degrees, undirected CSR, and exact hop one in Cython."""
    cdef Py_ssize_t edge_count = edges.shape[0]
    cdef cnp.ndarray[uint64_t, ndim=1] degrees_array = np.zeros(
        node_count, dtype=np.uint64
    )
    cdef uint64_t[::1] degrees = degrees_array
    cdef cnp.ndarray[uint64_t, ndim=1] offsets_array = np.zeros(
        node_count + 1, dtype=np.uint64
    )
    cdef uint64_t[::1] offsets = offsets_array
    cdef Py_ssize_t edge_index
    cdef Py_ssize_t node_index
    cdef uint32_t source
    cdef uint32_t destination

    for edge_index in range(edge_count):
        source = edges[edge_index, 0]
        destination = edges[edge_index, 1]
        degrees[source] += 1
        if destination != source:
            degrees[destination] += 1

    for node_index in range(node_count):
        offsets[node_index + 1] = offsets[node_index] + degrees[node_index]

    cdef cnp.ndarray[uint32_t, ndim=1] neighbors_array = np.empty(
        offsets[node_count], dtype=np.uint32
    )
    cdef uint32_t[::1] neighbors = neighbors_array
    cdef cnp.ndarray[uint64_t, ndim=1] cursors_array = np.asarray(
        offsets_array[:node_count], dtype=np.uint64
    ).copy()
    cdef uint64_t[::1] cursors = cursors_array

    for edge_index in range(edge_count):
        source = edges[edge_index, 0]
        destination = edges[edge_index, 1]
        neighbors[cursors[source]] = destination
        cursors[source] += 1
        if destination != source:
            neighbors[cursors[destination]] = source
            cursors[destination] += 1

    return degrees_array, offsets_array, neighbors_array


cdef bint _frontier_reaches_threshold(
    const uint64_t[::1] adjacency_offsets,
    const uint64_t[::1] frontier_offsets,
    const uint8_t[::1] frontier_encodings,
    const uint8_t[::1] frontier_payload,
    Py_ssize_t anchor,
    Py_ssize_t node_count,
    Py_ssize_t bytes_per_row,
    uint64_t routing_threshold,
) noexcept nogil:
    cdef uint64_t total = 0
    cdef uint64_t degree
    cdef uint64_t cursor = frontier_offsets[anchor]
    cdef uint64_t row_end = frontier_offsets[anchor + 1]
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint8_t value
    cdef uint32_t node

    if frontier_encodings[anchor] == 0:
        while cursor < row_end:
            node = (
                <uint32_t>frontier_payload[cursor]
                | (<uint32_t>frontier_payload[cursor + 1] << 8)
                | (<uint32_t>frontier_payload[cursor + 2] << 16)
                | (<uint32_t>frontier_payload[cursor + 3] << 24)
            )
            degree = adjacency_offsets[node + 1] - adjacency_offsets[node]
            if degree >= routing_threshold or total >= routing_threshold - degree:
                return True
            total += degree
            cursor += 4
        return False

    for byte_index in range(bytes_per_row):
        value = frontier_payload[cursor + byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if value & (1 << bit_index):
                node = <uint32_t>(byte_index * 8 + bit_index)
                if node >= node_count:
                    break
                degree = adjacency_offsets[node + 1] - adjacency_offsets[node]
                if degree >= routing_threshold or total >= routing_threshold - degree:
                    return True
                total += degree
    return False


def classify_frontiers_by_degree(
    const uint64_t[::1] adjacency_offsets,
    const uint64_t[::1] frontier_offsets,
    const uint8_t[::1] frontier_encodings,
    const uint8_t[::1] frontier_payload,
    const uint32_t[::1] anchor_indices,
    uint64_t routing_threshold,
    int worker_count,
):
    """Classify current frontiers on demand, stopping each sum at threshold."""
    cdef Py_ssize_t count = anchor_indices.shape[0]
    cdef Py_ssize_t node_count = adjacency_offsets.shape[0] - 1
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef cnp.ndarray[uint8_t, ndim=1] slow_array = np.zeros(
        count, dtype=np.uint8
    )
    cdef uint8_t[::1] slow = slow_array
    cdef Py_ssize_t index
    for index in prange(
        count,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        slow[index] = _frontier_reaches_threshold(
            adjacency_offsets,
            frontier_offsets,
            frontier_encodings,
            frontier_payload,
            anchor_indices[index],
            node_count,
            bytes_per_row,
            routing_threshold,
        )
    return slow_array


cdef bint _residual_frontier_reaches_threshold(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] frontier_offsets,
    const uint8_t[::1] frontier_encodings,
    const uint8_t[::1] frontier_payload,
    const uint8_t[:, ::1] known_rows,
    const int32_t[::1] known_row_by_node,
    Py_ssize_t anchor,
    Py_ssize_t node_count,
    Py_ssize_t bytes_per_row,
    uint64_t routing_threshold,
) noexcept nogil:
    cdef uint64_t work = 0
    cdef uint64_t cursor = frontier_offsets[anchor]
    cdef uint64_t row_end = frontier_offsets[anchor + 1]
    cdef uint64_t adjacency_index
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint8_t value
    cdef uint32_t frontier_node
    cdef uint32_t destination
    cdef int32_t known_row = known_row_by_node[anchor]

    while cursor < row_end if frontier_encodings[anchor] == 0 else False:
        frontier_node = (
            <uint32_t>frontier_payload[cursor]
            | (<uint32_t>frontier_payload[cursor + 1] << 8)
            | (<uint32_t>frontier_payload[cursor + 2] << 16)
            | (<uint32_t>frontier_payload[cursor + 3] << 24)
        )
        for adjacency_index in range(
            adjacency_offsets[frontier_node],
            adjacency_offsets[frontier_node + 1],
        ):
            destination = adjacency_neighbors[adjacency_index]
            if not (
                (
                    known_rows[known_row, destination >> 3]
                    & (1 << (destination & 7))
                )
                or (
                    known_rows[destination, anchor >> 3]
                    & (1 << (anchor & 7))
                )
            ):
                work += 1
                if work >= routing_threshold:
                    return True
        cursor += 4
    if frontier_encodings[anchor] == 0:
        return False

    cursor = frontier_offsets[anchor]
    for byte_index in range(bytes_per_row):
        value = frontier_payload[cursor + byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if not (value & (1 << bit_index)):
                continue
            frontier_node = <uint32_t>(byte_index * 8 + bit_index)
            if frontier_node >= node_count:
                break
            for adjacency_index in range(
                adjacency_offsets[frontier_node],
                adjacency_offsets[frontier_node + 1],
            ):
                destination = adjacency_neighbors[adjacency_index]
                if not (
                    (
                        known_rows[known_row, destination >> 3]
                        & (1 << (destination & 7))
                    )
                    or (
                        known_rows[destination, anchor >> 3]
                        & (1 << (anchor & 7))
                    )
                ):
                    work += 1
                    if work >= routing_threshold:
                        return True
    return False


def classify_frontiers_by_residual_degree(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] frontier_offsets,
    const uint8_t[::1] frontier_encodings,
    const uint8_t[::1] frontier_payload,
    const uint8_t[:, ::1] known_rows,
    const int32_t[::1] known_row_by_node,
    const uint32_t[::1] anchor_indices,
    uint64_t routing_threshold,
    int worker_count,
):
    """Route using only expansions whose destination is not yet known."""
    cdef Py_ssize_t count = anchor_indices.shape[0]
    cdef Py_ssize_t node_count = adjacency_offsets.shape[0] - 1
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef cnp.ndarray[uint8_t, ndim=1] slow_array = np.zeros(
        count, dtype=np.uint8
    )
    cdef uint8_t[::1] slow = slow_array
    cdef Py_ssize_t index
    for index in prange(
        count,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        slow[index] = _residual_frontier_reaches_threshold(
            adjacency_offsets,
            adjacency_neighbors,
            frontier_offsets,
            frontier_encodings,
            frontier_payload,
            known_rows,
            known_row_by_node,
            anchor_indices[index],
            node_count,
            bytes_per_row,
            routing_threshold,
        )
    return slow_array


cdef void _transpose_one_encoded_row(
    uint32_t anchor,
    uint64_t cursor,
    uint64_t row_end,
    uint8_t encoding,
    const uint8_t[::1] row_payload,
    uint8_t[:, ::1] dense_rows,
    const int32_t[::1] dense_row_by_node,
    Py_ssize_t node_count,
    Py_ssize_t bytes_per_row,
) noexcept nogil:
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint32_t destination
    cdef uint8_t value
    cdef int32_t anchor_row = dense_row_by_node[anchor]
    cdef int32_t destination_row
    if encoding == 0:
        while cursor < row_end:
            destination = (
                <uint32_t>row_payload[cursor]
                | (<uint32_t>row_payload[cursor + 1] << 8)
                | (<uint32_t>row_payload[cursor + 2] << 16)
                | (<uint32_t>row_payload[cursor + 3] << 24)
            )
            if anchor_row >= 0:
                dotdb_atomic_or_u8(
                    &dense_rows[anchor_row, destination >> 3],
                    <uint8_t>(1 << (destination & 7)),
                )
            destination_row = dense_row_by_node[destination]
            if destination_row >= 0:
                dotdb_atomic_or_u8(
                    &dense_rows[destination_row, anchor >> 3],
                    <uint8_t>(1 << (anchor & 7)),
                )
            cursor += 4
        return
    for byte_index in range(bytes_per_row):
        value = row_payload[cursor + byte_index]
        if value == 0:
            continue
        if anchor_row >= 0:
            dotdb_atomic_or_u8(&dense_rows[anchor_row, byte_index], value)
        for bit_index in range(8):
            if value & (1 << bit_index):
                destination = <uint32_t>(byte_index * 8 + bit_index)
                if destination >= node_count:
                    break
                destination_row = dense_row_by_node[destination]
                if destination_row >= 0:
                    dotdb_atomic_or_u8(
                        &dense_rows[destination_row, anchor >> 3],
                        <uint8_t>(1 << (anchor & 7)),
                    )


def transpose_encoded_rows_into_dense(
    const uint32_t[::1] anchor_indices,
    const uint64_t[::1] row_offsets,
    const uint8_t[::1] row_encodings,
    const uint8_t[::1] row_payload,
    const uint8_t[::1] skip_rows,
    uint8_t[:, ::1] dense_rows,
    const int32_t[::1] dense_row_by_node,
    Py_ssize_t node_count,
    int worker_count,
):
    """Write exact-hop rows and their symmetric transpose atomically."""
    cdef Py_ssize_t row_count = anchor_indices.shape[0]
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef Py_ssize_t local_row
    for local_row in prange(
        row_count,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        if skip_rows[local_row]:
            continue
        _transpose_one_encoded_row(
            anchor_indices[local_row],
            row_offsets[local_row],
            row_offsets[local_row + 1],
            row_encodings[local_row],
            row_payload,
            dense_rows,
            dense_row_by_node,
            node_count,
            bytes_per_row,
        )


cdef void _write_one_encoded_row(
    uint32_t anchor,
    uint64_t cursor,
    uint64_t row_end,
    uint8_t encoding,
    const uint8_t[::1] row_payload,
    uint8_t[:, ::1] dense_rows,
    Py_ssize_t bytes_per_row,
) noexcept nogil:
    cdef Py_ssize_t byte_index
    cdef uint32_t destination
    if encoding == 0:
        while cursor < row_end:
            destination = _read_uint32_le(row_payload, cursor)
            dense_rows[anchor, destination >> 3] |= <uint8_t>(
                1 << (destination & 7)
            )
            cursor += 4
        return
    for byte_index in range(bytes_per_row):
        dense_rows[anchor, byte_index] = row_payload[cursor + byte_index]


def write_encoded_rows_into_dense(
    const uint32_t[::1] anchor_indices,
    const uint64_t[::1] row_offsets,
    const uint8_t[::1] row_encodings,
    const uint8_t[::1] row_payload,
    const uint8_t[::1] skip_rows,
    uint8_t[:, ::1] dense_rows,
    Py_ssize_t node_count,
    int worker_count,
):
    """Write each encoded result only to its own fixed-width row."""
    cdef Py_ssize_t row_count = anchor_indices.shape[0]
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef Py_ssize_t local_row
    for local_row in prange(
        row_count,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        if skip_rows[local_row]:
            continue
        _write_one_encoded_row(
            anchor_indices[local_row],
            row_offsets[local_row],
            row_offsets[local_row + 1],
            row_encodings[local_row],
            row_payload,
            dense_rows,
            bytes_per_row,
        )


def collect_unallocated_reverse_destinations(
    const uint64_t[::1] row_offsets,
    const uint8_t[::1] row_encodings,
    const uint8_t[::1] row_payload,
    const uint8_t[::1] skip_rows,
    const uint8_t[::1] completed_rows,
    const int32_t[::1] dense_row_by_node,
    Py_ssize_t node_count,
):
    """Return unique unfinished reverse destinations lacking temp rows."""
    cdef Py_ssize_t row_count = row_encodings.shape[0]
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef Py_ssize_t local_row
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint64_t cursor
    cdef uint64_t row_end
    cdef uint8_t value
    cdef uint32_t destination
    cdef vector[uint32_t] candidates
    cdef vector[uint32_t] unique_destinations

    with nogil:
        for local_row in range(row_count):
            if skip_rows[local_row]:
                continue
            cursor = row_offsets[local_row]
            row_end = row_offsets[local_row + 1]
            if row_encodings[local_row] == 0:
                while cursor < row_end:
                    destination = _read_uint32_le(row_payload, cursor)
                    if (
                        not completed_rows[destination]
                        and dense_row_by_node[destination] < 0
                    ):
                        candidates.push_back(destination)
                    cursor += 4
            else:
                for byte_index in range(bytes_per_row):
                    value = row_payload[cursor + byte_index]
                    if value == 0:
                        continue
                    for bit_index in range(8):
                        if value & (1 << bit_index):
                            destination = <uint32_t>(
                                byte_index * 8 + bit_index
                            )
                            if destination >= node_count:
                                break
                            if (
                                not completed_rows[destination]
                                and dense_row_by_node[destination] < 0
                            ):
                                candidates.push_back(destination)
        if candidates.size():
            sort(candidates.begin(), candidates.end())
            unique_destinations.push_back(candidates[0])
            for local_row in range(1, candidates.size()):
                if candidates[local_row] != candidates[local_row - 1]:
                    unique_destinations.push_back(candidates[local_row])

    cdef cnp.ndarray[uint32_t, ndim=1] output = np.empty(
        unique_destinations.size(), dtype=np.uint32
    )
    for local_row in range(unique_destinations.size()):
        output[local_row] = unique_destinations[local_row]
    return output


cdef void _buffer_one_transpose_row(
    Py_ssize_t local_source,
    uint64_t cursor,
    uint64_t row_end,
    uint8_t encoding,
    const uint8_t[::1] payload,
    uint8_t[:, ::1] transpose_buffer,
    const int32_t[::1] deferred_row_by_node,
    Py_ssize_t node_count,
    Py_ssize_t bytes_per_row,
) noexcept nogil:
    cdef uint32_t destination
    cdef int32_t deferred_row
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint8_t value
    if encoding == 0:
        while cursor < row_end:
            destination = _read_uint32_le(payload, cursor)
            deferred_row = deferred_row_by_node[destination]
            if deferred_row >= 0:
                dotdb_atomic_or_u8(
                    &transpose_buffer[
                        deferred_row, local_source >> 3
                    ],
                    <uint8_t>(1 << (local_source & 7)),
                )
            cursor += 4
        return
    for byte_index in range(bytes_per_row):
        value = payload[cursor + byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if value & (1 << bit_index):
                destination = <uint32_t>(byte_index * 8 + bit_index)
                if destination >= node_count:
                    break
                deferred_row = deferred_row_by_node[destination]
                if deferred_row >= 0:
                    dotdb_atomic_or_u8(
                        &transpose_buffer[
                            deferred_row, local_source >> 3
                        ],
                        <uint8_t>(1 << (local_source & 7)),
                    )


def transpose_encoded_rows_into_dense_buffered(
    const uint32_t[::1] source_anchors,
    const uint64_t[::1] row_offsets,
    const uint8_t[::1] row_encodings,
    const uint8_t[::1] row_payload,
    const uint8_t[::1] skip_rows,
    uint8_t[:, ::1] dense_rows,
    const int32_t[::1] deferred_row_by_node,
    Py_ssize_t node_count,
    int worker_count,
):
    """Transpose through a bounded RAM block, then write by destination."""
    cdef Py_ssize_t source_count = source_anchors.shape[0]
    cdef Py_ssize_t deferred_count = dense_rows.shape[0]
    cdef Py_ssize_t block_bytes = (source_count + 7) // 8
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef cnp.ndarray[uint8_t, ndim=2] buffer_array = np.zeros(
        (deferred_count, block_bytes), dtype=np.uint8
    )
    cdef uint8_t[:, ::1] transpose_buffer = buffer_array
    cdef Py_ssize_t local_source
    cdef Py_ssize_t deferred_row
    cdef Py_ssize_t source_byte
    cdef int bit_index
    cdef uint8_t value
    cdef uint32_t source

    for local_source in prange(
        source_count, nogil=True, schedule="dynamic", num_threads=worker_count
    ):
        if not skip_rows[local_source]:
            _buffer_one_transpose_row(
                local_source,
                row_offsets[local_source],
                row_offsets[local_source + 1],
                row_encodings[local_source],
                row_payload,
                transpose_buffer,
                deferred_row_by_node,
                node_count,
                bytes_per_row,
            )

    # One worker owns each destination row, so mmap writes need no atomics and
    # advance through source IDs in sorted batch order.
    for deferred_row in prange(
        deferred_count, nogil=True, schedule="static", num_threads=worker_count
    ):
        for source_byte in range(block_bytes):
            value = transpose_buffer[deferred_row, source_byte]
            if value == 0:
                continue
            for bit_index in range(8):
                if value & (1 << bit_index):
                    local_source = source_byte * 8 + bit_index
                    if local_source >= source_count:
                        break
                    source = source_anchors[local_source]
                    dense_rows[
                        deferred_row, source >> 3
                    ] |= <uint8_t>(1 << (source & 7))


cdef void _build_anchor_hops(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    int64_t[:, ::1] seen,
    uint32_t[:, ::1] frontier_a,
    uint32_t[:, ::1] frontier_b,
    vector[vector[uint32_t]]& output,
    Py_ssize_t batch_size,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    int max_hop,
    int worker,
) noexcept nogil:
    """Traverse one anchor and write all requested exact-hop rows."""
    cdef Py_ssize_t current_index
    cdef Py_ssize_t current_count = 1
    cdef Py_ssize_t next_count
    cdef Py_ssize_t adjacency_index
    cdef Py_ssize_t adjacency_start
    cdef Py_ssize_t adjacency_end
    cdef uint32_t current_node
    cdef uint32_t neighbor
    cdef int64_t stamp = anchor + 1
    cdef int hop
    cdef bint current_is_a = True

    seen[worker, anchor] = stamp
    frontier_a[worker, 0] = <uint32_t>anchor

    for hop in range(1, max_hop + 1):
        next_count = 0

        for current_index in range(current_count):
            if current_is_a:
                current_node = frontier_a[worker, current_index]
            else:
                current_node = frontier_b[worker, current_index]

            adjacency_start = adjacency_offsets[current_node]
            adjacency_end = adjacency_offsets[current_node + 1]

            for adjacency_index in range(adjacency_start, adjacency_end):
                neighbor = adjacency_neighbors[adjacency_index]

                if seen[worker, neighbor] == stamp:
                    continue

                seen[worker, neighbor] = stamp

                if current_is_a:
                    frontier_b[worker, next_count] = neighbor
                else:
                    frontier_a[worker, next_count] = neighbor
                next_count += 1

                if hop >= 2:
                    output[
                        (hop - 2) * batch_size + local_anchor
                    ].push_back(neighbor)

        current_count = next_count
        current_is_a = not current_is_a

        if current_count == 0:
            break


cdef void _encode_hop_rows(
    vector[vector[uint32_t]]& output,
    Py_ssize_t batch_size,
    Py_ssize_t hop_index,
    Py_ssize_t bytes_per_row,
    const uint64_t[::1] offsets,
    const uint8_t[::1] encodings,
    uint8_t[::1] payload,
    int worker_count,
) noexcept nogil:
    """Encode one hop's rows concurrently into a contiguous payload."""
    cdef Py_ssize_t local_anchor
    cdef Py_ssize_t result_index
    cdef Py_ssize_t node_index
    cdef Py_ssize_t row_start
    cdef uint32_t node

    for local_anchor in prange(
        batch_size,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        result_index = hop_index * batch_size + local_anchor
        row_start = offsets[local_anchor]

        if encodings[local_anchor] == 0:
            # Sparse rows preserve the same ascending dense-ID order returned
            # by unpacking a bitmap, keeping retrieval behavior deterministic.
            sort(
                output[result_index].begin(),
                output[result_index].end(),
            )
            for node_index in range(output[result_index].size()):
                node = output[result_index][node_index]
                payload[row_start + node_index * 4] = <uint8_t>(node & 255)
                payload[row_start + node_index * 4 + 1] = <uint8_t>((node >> 8) & 255)
                payload[row_start + node_index * 4 + 2] = <uint8_t>((node >> 16) & 255)
                payload[row_start + node_index * 4 + 3] = <uint8_t>((node >> 24) & 255)
        else:
            # Dense rows are already zero-initialized. Each entity discovered
            # by this anchor is unique, so workers can set its membership bit
            # without atomics inside their disjoint row ranges.
            for node_index in range(output[result_index].size()):
                node = output[result_index][node_index]
                payload[
                    row_start + (node >> 3)
                ] |= <uint8_t>(1 << (node & 7))


cdef inline uint32_t _read_uint32_le(
    const uint8_t[::1] payload,
    uint64_t position,
) noexcept nogil:
    """Read one little-endian uint32 ID from an adaptive payload."""
    return (
        <uint32_t>payload[position]
        | (<uint32_t>payload[position + 1] << 8)
        | (<uint32_t>payload[position + 2] << 16)
        | (<uint32_t>payload[position + 3] << 24)
    )


cdef void _build_anchor_hop2_from_global_hop1(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    int64_t[:, ::1] seen,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    int worker,
) noexcept nogil:
    """Union globally stored hop-one rows into one exact-hop-two row."""
    cdef int64_t stamp = anchor + 1
    cdef Py_ssize_t hop_1_index
    cdef Py_ssize_t hop_1_start = adjacency_offsets[anchor]
    cdef Py_ssize_t hop_1_end = adjacency_offsets[anchor + 1]
    cdef Py_ssize_t neighbor_index
    cdef Py_ssize_t neighbor_start
    cdef Py_ssize_t neighbor_end
    cdef uint32_t hop_1_node
    cdef uint32_t candidate

    # H0 and H1 are excluded before any row unions are performed.
    seen[worker, anchor] = stamp
    for hop_1_index in range(hop_1_start, hop_1_end):
        seen[worker, adjacency_neighbors[hop_1_index]] = stamp

    # H2(a) = union(H1(u) for u in H1(a)) - H0(a) - H1(a).
    for hop_1_index in range(hop_1_start, hop_1_end):
        hop_1_node = adjacency_neighbors[hop_1_index]
        neighbor_start = adjacency_offsets[hop_1_node]
        neighbor_end = adjacency_offsets[hop_1_node + 1]
        for neighbor_index in range(neighbor_start, neighbor_end):
            candidate = adjacency_neighbors[neighbor_index]
            if seen[worker, candidate] != stamp:
                seen[worker, candidate] = stamp
                output[local_anchor].push_back(candidate)


cdef void _build_anchor_next_hop(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const int32_t[::1] heavy_row_by_node,
    const uint8_t[:, ::1] heavy_adjacency_bitmaps,
    int64_t[:, ::1] seen,
    uint8_t[:, ::1] bitmap_workspace,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    Py_ssize_t bytes_per_row,
    int worker,
) noexcept nogil:
    """Build one exact hop from globally completed preceding hop rows."""
    cdef int64_t stamp = anchor + 1
    cdef uint64_t row_start = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef uint64_t two_back_start
    cdef uint64_t two_back_end
    cdef uint64_t position
    cdef uint64_t neighbor_row_start
    cdef uint64_t neighbor_row_end
    cdef Py_ssize_t adjacency_index
    cdef Py_ssize_t adjacency_start
    cdef Py_ssize_t adjacency_end
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint8_t value
    cdef uint32_t node
    cdef uint32_t neighbor
    cdef uint64_t direct_work = 0
    cdef uint64_t union_work = 0
    cdef int32_t heavy_row

    seen[worker, anchor] = stamp

    adjacency_start = adjacency_offsets[anchor]
    adjacency_end = adjacency_offsets[anchor + 1]

    # Sparse anchor rows support both strategies. Estimate their actual work:
    # direct expansion scans the adjacency entries of H2(a), while cached
    # union reads the encoded H2 rows belonging to H1(a). Choose the cheaper
    # path per anchor instead of imposing one strategy on the whole graph.
    if previous_encodings[anchor] == 0:
        position = row_start
        while position < row_end:
            node = _read_uint32_le(previous_payload, position)
            heavy_row = heavy_row_by_node[node]
            if heavy_row >= 0:
                direct_work += bytes_per_row // 4
            else:
                direct_work += (
                    adjacency_offsets[node + 1] - adjacency_offsets[node]
                )
            position += 4

        for adjacency_index in range(adjacency_start, adjacency_end):
            neighbor = adjacency_neighbors[adjacency_index]
            union_work += (
                previous_offsets[neighbor + 1] - previous_offsets[neighbor]
            )

        if direct_work * 4 <= union_work + bytes_per_row * 2:
            # Expand the preceding frontier into a bitmap. High-degree nodes
            # use prepacked adjacency rows; lower-degree nodes scan CSR.
            for byte_index in range(bytes_per_row):
                bitmap_workspace[worker, byte_index] = 0

            position = row_start
            while position < row_end:
                node = _read_uint32_le(previous_payload, position)
                heavy_row = heavy_row_by_node[node]
                if heavy_row >= 0:
                    for byte_index in range(bytes_per_row):
                        bitmap_workspace[worker, byte_index] |= (
                            heavy_adjacency_bitmaps[heavy_row, byte_index]
                        )
                else:
                    for adjacency_index in range(
                        adjacency_offsets[node], adjacency_offsets[node + 1]
                    ):
                        neighbor = adjacency_neighbors[adjacency_index]
                        bitmap_workspace[worker, neighbor >> 3] |= <uint8_t>(
                            1 << (neighbor & 7)
                        )
                position += 4

            # A neighbor of H(h-1) can only land in H(h-2), H(h-1), or H(h).
            # Remove the two completed shorter rows to isolate exact hop h.
            if two_back_is_hop_one:
                for adjacency_index in range(adjacency_start, adjacency_end):
                    node = adjacency_neighbors[adjacency_index]
                    bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                        ~(1 << (node & 7))
                    )
            else:
                two_back_start = two_back_offsets[anchor]
                two_back_end = two_back_offsets[anchor + 1]
                if two_back_encodings[anchor] == 0:
                    position = two_back_start
                    while position < two_back_end:
                        node = _read_uint32_le(two_back_payload, position)
                        bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                            ~(1 << (node & 7))
                        )
                        position += 4
                else:
                    for byte_index in range(bytes_per_row):
                        value = two_back_payload[two_back_start + byte_index]
                        if value == 0:
                            continue
                        for bit_index in range(8):
                            if value & (1 << bit_index):
                                node = <uint32_t>(byte_index * 8 + bit_index)
                                if node < adjacency_offsets.shape[0] - 1:
                                    bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                                        ~(1 << (node & 7))
                                    )

            # Remove the complete preceding exact-hop row.
            position = row_start
            while position < row_end:
                node = _read_uint32_le(previous_payload, position)
                bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                    ~(1 << (node & 7))
                )
                position += 4

            bitmap_workspace[worker, anchor >> 3] &= <uint8_t>(
                ~(1 << (anchor & 7))
            )
            for byte_index in range(bytes_per_row):
                value = bitmap_workspace[worker, byte_index]
                if value == 0:
                    continue
                for bit_index in range(8):
                    if value & (1 << bit_index):
                        node = <uint32_t>(byte_index * 8 + bit_index)
                        if node < adjacency_offsets.shape[0] - 1:
                            output[local_anchor].push_back(node)
            return

    # Cached union is cheaper for dense rows and sparse rows whose frontier
    # expansion would scan more data than their required stored H2 rows.
    for byte_index in range(bytes_per_row):
        bitmap_workspace[worker, byte_index] = 0

    for adjacency_index in range(adjacency_start, adjacency_end):
        neighbor = adjacency_neighbors[adjacency_index]
        if seen[worker, neighbor] == stamp:
            continue
        seen[worker, neighbor] = stamp

        neighbor_row_start = previous_offsets[neighbor]
        neighbor_row_end = previous_offsets[neighbor + 1]
        if previous_encodings[neighbor] == 0:
            position = neighbor_row_start
            while position < neighbor_row_end:
                node = _read_uint32_le(previous_payload, position)
                bitmap_workspace[worker, node >> 3] |= <uint8_t>(
                    1 << (node & 7)
                )
                position += 4
        else:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[worker, byte_index] |= previous_payload[
                    neighbor_row_start + byte_index
                ]

    # Remove H0 and the exact row two hops behind the requested output.
    bitmap_workspace[worker, anchor >> 3] &= <uint8_t>(
        ~(1 << (anchor & 7))
    )
    if two_back_is_hop_one:
        for adjacency_index in range(adjacency_start, adjacency_end):
            neighbor = adjacency_neighbors[adjacency_index]
            bitmap_workspace[worker, neighbor >> 3] &= <uint8_t>(
                ~(1 << (neighbor & 7))
            )
    else:
        two_back_start = two_back_offsets[anchor]
        two_back_end = two_back_offsets[anchor + 1]
        if two_back_encodings[anchor] == 0:
            position = two_back_start
            while position < two_back_end:
                node = _read_uint32_le(two_back_payload, position)
                bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                    ~(1 << (node & 7))
                )
                position += 4
        else:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[worker, byte_index] &= <uint8_t>(
                    ~two_back_payload[two_back_start + byte_index]
                )

    # Remove the complete preceding exact-hop row.
    if previous_encodings[anchor] == 0:
        position = row_start
        while position < row_end:
            node = _read_uint32_le(previous_payload, position)
            bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                ~(1 << (node & 7))
            )
            position += 4
    else:
        for byte_index in range(bytes_per_row):
            bitmap_workspace[worker, byte_index] &= <uint8_t>(
                ~previous_payload[row_start + byte_index]
            )

    # Materialize the remaining exact-hop-three IDs for adaptive encoding.
    for byte_index in range(bytes_per_row):
        value = bitmap_workspace[worker, byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if value & (1 << bit_index):
                node = <uint32_t>(byte_index * 8 + bit_index)
                if node < adjacency_offsets.shape[0] - 1:
                    output[local_anchor].push_back(node)


cdef inline uint8_t _merge_worker_bitmap_byte(
    const uint8_t[:, ::1] bitmap_workspace,
    Py_ssize_t byte_index,
    int worker_count,
) noexcept nogil:
    """OR one byte across all private worker bitmaps."""
    cdef uint8_t merged_value = 0
    cdef int worker
    for worker in range(worker_count):
        merged_value = <uint8_t>(
            merged_value | bitmap_workspace[worker, byte_index]
        )
    return merged_value


cdef void _build_anchor_direct_parallel(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const int32_t[::1] heavy_row_by_node,
    const uint8_t[:, ::1] heavy_adjacency_bitmaps,
    uint8_t[:, ::1] bitmap_workspace,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    Py_ssize_t bytes_per_row,
    int worker_count,
) noexcept nogil:
    """Split one expensive sparse-frontier expansion across every worker."""
    cdef uint64_t row_start = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef Py_ssize_t frontier_count = (row_end - row_start) // 4
    cdef Py_ssize_t frontier_index
    cdef Py_ssize_t adjacency_index
    cdef Py_ssize_t byte_index
    cdef Py_ssize_t adjacency_start = adjacency_offsets[anchor]
    cdef Py_ssize_t adjacency_end = adjacency_offsets[anchor + 1]
    cdef uint64_t position
    cdef uint64_t two_back_start
    cdef uint64_t two_back_end
    cdef uint32_t node
    cdef uint32_t neighbor
    cdef uint8_t value
    cdef int bit_index
    cdef int worker
    cdef int reduce_worker
    cdef int32_t heavy_row
    for worker in range(worker_count):
        for byte_index in range(bytes_per_row):
            bitmap_workspace[worker, byte_index] = 0

    for frontier_index in prange(
        frontier_count,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        worker = threadid()
        node = _read_uint32_le(
            previous_payload,
            row_start + frontier_index * 4,
        )
        heavy_row = heavy_row_by_node[node]
        if heavy_row >= 0:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[worker, byte_index] |= (
                    heavy_adjacency_bitmaps[heavy_row, byte_index]
                )
        else:
            for adjacency_index in range(
                adjacency_offsets[node], adjacency_offsets[node + 1]
            ):
                neighbor = adjacency_neighbors[adjacency_index]
                bitmap_workspace[worker, neighbor >> 3] |= <uint8_t>(
                    1 << (neighbor & 7)
                )

    # Reduce all private rows in one parallel region.  The previous version
    # created one OpenMP team per worker row, which could strand workers at
    # different barriers on the large MAG anchors.
    for byte_index in prange(
        bytes_per_row,
        schedule="static",
        num_threads=worker_count,
    ):
        bitmap_workspace[0, byte_index] = _merge_worker_bitmap_byte(
            bitmap_workspace,
            byte_index,
            worker_count,
        )

    # Remove exact hop h-2.
    if two_back_is_hop_one:
        for adjacency_index in range(adjacency_start, adjacency_end):
            node = adjacency_neighbors[adjacency_index]
            bitmap_workspace[0, node >> 3] &= <uint8_t>(
                ~(1 << (node & 7))
            )
    else:
        two_back_start = two_back_offsets[anchor]
        two_back_end = two_back_offsets[anchor + 1]
        if two_back_encodings[anchor] == 0:
            position = two_back_start
            while position < two_back_end:
                node = _read_uint32_le(two_back_payload, position)
                bitmap_workspace[0, node >> 3] &= <uint8_t>(
                    ~(1 << (node & 7))
                )
                position += 4
        else:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[0, byte_index] &= <uint8_t>(
                    ~two_back_payload[two_back_start + byte_index]
                )

    # Remove exact hop h-1 and the anchor itself.
    position = row_start
    while position < row_end:
        node = _read_uint32_le(previous_payload, position)
        bitmap_workspace[0, node >> 3] &= <uint8_t>(
            ~(1 << (node & 7))
        )
        position += 4
    bitmap_workspace[0, anchor >> 3] &= <uint8_t>(
        ~(1 << (anchor & 7))
    )

    for byte_index in range(bytes_per_row):
        value = bitmap_workspace[0, byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if value & (1 << bit_index):
                node = <uint32_t>(byte_index * 8 + bit_index)
                if node < adjacency_offsets.shape[0] - 1:
                    output[local_anchor].push_back(node)


cdef inline void _or_sparse_payload_private(
    uint8_t[:, ::1] bitmap_workspace,
    int worker,
    const uint8_t[::1] payload,
    uint64_t row_start,
    uint64_t row_end,
) noexcept nogil:
    cdef uint64_t cursor = row_start
    cdef uint32_t node
    while cursor < row_end:
        node = _read_uint32_le(payload, cursor)
        bitmap_workspace[worker, node >> 3] |= <uint8_t>(
            1 << (node & 7)
        )
        cursor += 4


cdef void _build_anchor_union_parallel(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    uint8_t[:, ::1] bitmap_workspace,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    Py_ssize_t bytes_per_row,
    int worker_count,
) noexcept nogil:
    """Split one expensive global-row union across private worker bitmaps."""
    cdef Py_ssize_t adjacency_start = adjacency_offsets[anchor]
    cdef Py_ssize_t adjacency_end = adjacency_offsets[anchor + 1]
    cdef Py_ssize_t adjacency_index
    cdef Py_ssize_t byte_index
    cdef uint64_t row_start = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef uint64_t neighbor_row_start
    cdef uint64_t neighbor_row_end
    cdef uint64_t two_back_start
    cdef uint64_t two_back_end
    cdef uint64_t position
    cdef uint32_t neighbor
    cdef uint32_t node
    cdef uint8_t value
    cdef int bit_index
    cdef int worker

    for worker in range(worker_count):
        for byte_index in range(bytes_per_row):
            bitmap_workspace[worker, byte_index] = 0

    for adjacency_index in prange(
        adjacency_start,
        adjacency_end,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        worker = threadid()
        neighbor = adjacency_neighbors[adjacency_index]
        neighbor_row_start = previous_offsets[neighbor]
        neighbor_row_end = previous_offsets[neighbor + 1]
        if previous_encodings[neighbor] == 0:
            _or_sparse_payload_private(
                bitmap_workspace,
                worker,
                previous_payload,
                neighbor_row_start,
                neighbor_row_end,
            )
        else:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[worker, byte_index] |= previous_payload[
                    neighbor_row_start + byte_index
                ]

    for byte_index in prange(
        bytes_per_row,
        schedule="static",
        num_threads=worker_count,
    ):
        bitmap_workspace[0, byte_index] = _merge_worker_bitmap_byte(
            bitmap_workspace,
            byte_index,
            worker_count,
        )

    bitmap_workspace[0, anchor >> 3] &= <uint8_t>(
        ~(1 << (anchor & 7))
    )
    if two_back_is_hop_one:
        for adjacency_index in range(adjacency_start, adjacency_end):
            node = adjacency_neighbors[adjacency_index]
            bitmap_workspace[0, node >> 3] &= <uint8_t>(
                ~(1 << (node & 7))
            )
    else:
        two_back_start = two_back_offsets[anchor]
        two_back_end = two_back_offsets[anchor + 1]
        if two_back_encodings[anchor] == 0:
            position = two_back_start
            while position < two_back_end:
                node = _read_uint32_le(two_back_payload, position)
                bitmap_workspace[0, node >> 3] &= <uint8_t>(
                    ~(1 << (node & 7))
                )
                position += 4
        else:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[0, byte_index] &= <uint8_t>(
                    ~two_back_payload[two_back_start + byte_index]
                )

    if previous_encodings[anchor] == 0:
        position = row_start
        while position < row_end:
            node = _read_uint32_le(previous_payload, position)
            bitmap_workspace[0, node >> 3] &= <uint8_t>(
                ~(1 << (node & 7))
            )
            position += 4
    else:
        for byte_index in range(bytes_per_row):
            bitmap_workspace[0, byte_index] &= <uint8_t>(
                ~previous_payload[row_start + byte_index]
            )

    for byte_index in range(bytes_per_row):
        value = bitmap_workspace[0, byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if value & (1 << bit_index):
                node = <uint32_t>(byte_index * 8 + bit_index)
                if node < adjacency_offsets.shape[0] - 1:
                    output[local_anchor].push_back(node)


def build_exact_hops_batch(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    Py_ssize_t anchor_start,
    Py_ssize_t anchor_end,
    int max_hop,
    int worker_count,
    uint64_t routing_threshold=18446744073709551615,
):
    """Return CSR-style exact-hop node IDs for one contiguous anchor batch.

    Output axis zero represents hops two through ``max_hop``. Every anchor is
    traversed once: the frontier discovered for hop h is reused immediately
    to construct hop h+1.
    """
    cdef Py_ssize_t node_count = adjacency_offsets.shape[0] - 1
    cdef Py_ssize_t batch_size = anchor_end - anchor_start
    cdef int output_hop_count = max_hop - 1

    if max_hop < 2:
        raise ValueError("max_hop must be at least 2")
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if anchor_start < 0 or anchor_end > node_count:
        raise ValueError("anchor range is outside the graph")
    if anchor_end < anchor_start:
        raise ValueError("anchor_end must not precede anchor_start")

    # Each OpenMP worker owns one visited-stamp row and two reusable frontier
    # buffers. Stamps avoid clearing node_count entries before every anchor.
    cdef cnp.ndarray[int64_t, ndim=2] seen_array = np.zeros(
        (worker_count, node_count),
        dtype=np.int64,
    )
    cdef cnp.ndarray[uint32_t, ndim=2] frontier_a_array = np.empty(
        (worker_count, node_count),
        dtype=np.uint32,
    )
    cdef cnp.ndarray[uint32_t, ndim=2] frontier_b_array = np.empty(
        (worker_count, node_count),
        dtype=np.uint32,
    )
    cdef int64_t[:, ::1] seen = seen_array
    cdef uint32_t[:, ::1] frontier_a = frontier_a_array
    cdef uint32_t[:, ::1] frontier_b = frontier_b_array
    cdef vector[vector[uint32_t]] output
    output.resize(output_hop_count * batch_size)

    cdef Py_ssize_t local_anchor
    cdef Py_ssize_t anchor
    cdef int worker

    for local_anchor in prange(
        batch_size,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        worker = threadid()
        anchor = anchor_start + local_anchor
        _build_anchor_hops(
            adjacency_offsets,
            adjacency_neighbors,
            seen,
            frontier_a,
            frontier_b,
            output,
            batch_size,
            local_anchor,
            anchor,
            max_hop,
            worker,
        )

    # Encode each hop into one contiguous payload. Row sizes are calculated
    # first so every worker receives a disjoint output range.
    cdef list hop_results = []
    cdef cnp.ndarray[uint64_t, ndim=1] offsets_array
    cdef cnp.ndarray[uint8_t, ndim=1] encodings_array
    cdef cnp.ndarray[uint8_t, ndim=1] payload_array
    cdef cnp.ndarray[uint64_t, ndim=1] frontier_work_array
    cdef uint64_t[::1] offsets
    cdef uint8_t[::1] encodings
    cdef uint8_t[::1] payload
    cdef uint64_t[::1] frontier_work
    cdef Py_ssize_t hop_index
    cdef Py_ssize_t result_index
    cdef Py_ssize_t reachable_count
    cdef Py_ssize_t encoded_size
    cdef Py_ssize_t total_bytes
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef Py_ssize_t node_index
    cdef uint32_t frontier_node
    cdef uint64_t frontier_degree

    for hop_index in range(output_hop_count):
        offsets_array = np.zeros(batch_size + 1, dtype=np.uint64)
        encodings_array = np.zeros(batch_size, dtype=np.uint8)
        offsets = offsets_array
        encodings = encodings_array
        total_bytes = 0

        for local_anchor in range(batch_size):
            result_index = hop_index * batch_size + local_anchor
            reachable_count = output[result_index].size()
            if reachable_count * 4 < bytes_per_row:
                encoded_size = reachable_count * 4
                encodings[local_anchor] = 0
            else:
                encoded_size = bytes_per_row
                encodings[local_anchor] = 1
            total_bytes += encoded_size
            offsets[local_anchor + 1] = total_bytes

        payload_array = np.zeros(total_bytes, dtype=np.uint8)
        payload = payload_array
        _encode_hop_rows(
            output,
            batch_size,
            hop_index,
            bytes_per_row,
            offsets,
            encodings,
            payload,
            worker_count,
        )

        # The exact frontier is already resident in output while this row is
        # encoded. Persist the cost of expanding it now so the next hop can
        # route the anchor with one uint64 lookup instead of rescanning it.
        frontier_work_array = np.zeros(batch_size, dtype=np.uint64)
        frontier_work = frontier_work_array
        for local_anchor in range(batch_size):
            result_index = hop_index * batch_size + local_anchor
            for node_index in range(output[result_index].size()):
                frontier_node = output[result_index][node_index]
                frontier_degree = (
                    adjacency_offsets[frontier_node + 1]
                    - adjacency_offsets[frontier_node]
                )
                if (
                    frontier_degree >= routing_threshold
                    or frontier_work[local_anchor]
                    >= routing_threshold - frontier_degree
                ):
                    frontier_work[local_anchor] = routing_threshold
                    break
                frontier_work[local_anchor] += frontier_degree

        hop_results.append(
            (
                offsets_array,
                encodings_array,
                payload_array,
                frontier_work_array,
            )
        )

    return hop_results


cdef void _finalize_partial_next_hop_bitmap(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    uint8_t[:, ::1] bitmap_workspace,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    Py_ssize_t bytes_per_row,
    int worker,
) noexcept nogil:
    """Remove shorter hops and materialize a complete or partial bitmap."""
    cdef uint64_t row_start = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef uint64_t two_back_start
    cdef uint64_t two_back_end
    cdef uint64_t cursor
    cdef Py_ssize_t adjacency_index
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint32_t node
    cdef uint8_t value

    bitmap_workspace[worker, anchor >> 3] &= <uint8_t>(
        ~(1 << (anchor & 7))
    )
    if two_back_is_hop_one:
        for adjacency_index in range(
            adjacency_offsets[anchor], adjacency_offsets[anchor + 1]
        ):
            node = adjacency_neighbors[adjacency_index]
            bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                ~(1 << (node & 7))
            )
    else:
        two_back_start = two_back_offsets[anchor]
        two_back_end = two_back_offsets[anchor + 1]
        if two_back_encodings[anchor] == 0:
            cursor = two_back_start
            while cursor < two_back_end:
                node = _read_uint32_le(two_back_payload, cursor)
                bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                    ~(1 << (node & 7))
                )
                cursor += 4
        else:
            for byte_index in range(bytes_per_row):
                bitmap_workspace[worker, byte_index] &= <uint8_t>(
                    ~two_back_payload[two_back_start + byte_index]
                )

    if previous_encodings[anchor] == 0:
        cursor = row_start
        while cursor < row_end:
            node = _read_uint32_le(previous_payload, cursor)
            bitmap_workspace[worker, node >> 3] &= <uint8_t>(
                ~(1 << (node & 7))
            )
            cursor += 4
    else:
        for byte_index in range(bytes_per_row):
            bitmap_workspace[worker, byte_index] &= <uint8_t>(
                ~previous_payload[row_start + byte_index]
            )

    for byte_index in range(bytes_per_row):
        value = bitmap_workspace[worker, byte_index]
        if value == 0:
            continue
        for bit_index in range(8):
            if value & (1 << bit_index):
                node = <uint32_t>(byte_index * 8 + bit_index)
                if node < adjacency_offsets.shape[0] - 1:
                    output[local_anchor].push_back(node)


cdef bint _build_anchor_next_hop_timed(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const int32_t[::1] heavy_row_by_node,
    const uint8_t[:, ::1] heavy_adjacency_bitmaps,
    const uint8_t[:, ::1] known_rows,
    const int32_t[::1] known_row_by_node,
    bint use_known_rows,
    uint8_t[:, ::1] bitmap_workspace,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    Py_ssize_t bytes_per_row,
    int worker,
    double time_limit_seconds,
    uint64_t* resume_position,
    uint8_t* resume_strategy,
    double* elapsed_seconds,
) noexcept nogil:
    """Build until complete or the wall-clock budget is exhausted."""
    cdef double started = omp_get_wtime()
    cdef double deadline = started + time_limit_seconds
    cdef uint64_t row_start = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef uint64_t cursor
    cdef uint64_t neighbor_row_start
    cdef uint64_t neighbor_row_end
    cdef uint64_t direct_work = 0
    cdef uint64_t union_work = 0
    cdef Py_ssize_t adjacency_start = adjacency_offsets[anchor]
    cdef Py_ssize_t adjacency_end = adjacency_offsets[anchor + 1]
    cdef Py_ssize_t adjacency_index
    cdef Py_ssize_t byte_index
    cdef Py_ssize_t checked_units = 0
    cdef uint32_t node
    cdef uint32_t neighbor
    cdef uint8_t value
    cdef int32_t heavy_row
    cdef bint use_direct = False
    cdef bint timed_out = False
    cdef int32_t known_row = -1

    resume_position[0] = 0
    resume_strategy[0] = 0
    if use_known_rows:
        known_row = known_row_by_node[anchor]

    # Select the cheaper recurrence, but keep the estimator under the same
    # wall-clock budget so even prediction cannot hold the batch hostage.
    if previous_encodings[anchor] == 0:
        cursor = row_start
        while cursor < row_end:
            node = _read_uint32_le(previous_payload, cursor)
            heavy_row = heavy_row_by_node[node]
            if heavy_row >= 0:
                direct_work += bytes_per_row // 4
            else:
                direct_work += (
                    adjacency_offsets[node + 1] - adjacency_offsets[node]
                )
            cursor += 4
            checked_units += 1
            if checked_units % 64 == 0 and omp_get_wtime() >= deadline:
                timed_out = True
                break

    if not timed_out:
        for adjacency_index in range(adjacency_start, adjacency_end):
            neighbor = adjacency_neighbors[adjacency_index]
            union_work += (
                previous_offsets[neighbor + 1] - previous_offsets[neighbor]
            )
            checked_units += 1
            if checked_units % 64 == 0 and omp_get_wtime() >= deadline:
                timed_out = True
                break

    # Tier-two retries begin with everything already learned by symmetric
    # transpose writes. Tier one passes use_known_rows=False and retain the
    # original empty-workspace behavior.
    for byte_index in range(bytes_per_row):
        if use_known_rows:
            bitmap_workspace[worker, byte_index] = known_rows[
                known_row, byte_index
            ]
        else:
            bitmap_workspace[worker, byte_index] = 0

    if timed_out:
        resume_strategy[0] = 3
        elapsed_seconds[0] = omp_get_wtime() - started
        return True

    use_direct = (
        previous_encodings[anchor] == 0
        and direct_work * 4 <= union_work + bytes_per_row * 2
    )

    if use_direct:
        resume_strategy[0] = 1
        cursor = row_start
        while cursor < row_end:
            node = _read_uint32_le(previous_payload, cursor)
            heavy_row = heavy_row_by_node[node]
            if heavy_row >= 0:
                for byte_index in range(bytes_per_row):
                    bitmap_workspace[worker, byte_index] |= (
                        heavy_adjacency_bitmaps[heavy_row, byte_index]
                    )
            else:
                for adjacency_index in range(
                    adjacency_offsets[node], adjacency_offsets[node + 1]
                ):
                    neighbor = adjacency_neighbors[adjacency_index]
                    # Hard-anchor retries can infer this exact-hop result
                    # directly from an already-completed transpose cell.
                    # The fast pass never enters this branch because it calls
                    # the builder with use_known_rows=False.
                    if use_known_rows and (
                        known_rows[neighbor, anchor >> 3]
                        & (1 << (anchor & 7))
                    ):
                        bitmap_workspace[
                            worker, neighbor >> 3
                        ] |= <uint8_t>(1 << (neighbor & 7))
                        continue
                    if not (
                        bitmap_workspace[worker, neighbor >> 3]
                        & (1 << (neighbor & 7))
                    ):
                        bitmap_workspace[
                            worker, neighbor >> 3
                        ] |= <uint8_t>(1 << (neighbor & 7))
            cursor += 4
            checked_units += 1
            if checked_units % 32 == 0 and omp_get_wtime() >= deadline:
                timed_out = True
                break
        resume_position[0] = cursor
    else:
        resume_strategy[0] = 2
        adjacency_index = adjacency_start
        while adjacency_index < adjacency_end:
            neighbor = adjacency_neighbors[adjacency_index]
            neighbor_row_start = previous_offsets[neighbor]
            neighbor_row_end = previous_offsets[neighbor + 1]
            if previous_encodings[neighbor] == 0:
                cursor = neighbor_row_start
                while cursor < neighbor_row_end:
                    node = _read_uint32_le(previous_payload, cursor)
                    # As above, this lookup is exclusive to hard-anchor
                    # retries and reuses the known symmetric result.
                    if use_known_rows and (
                        known_rows[node, anchor >> 3]
                        & (1 << (anchor & 7))
                    ):
                        bitmap_workspace[
                            worker, node >> 3
                        ] |= <uint8_t>(1 << (node & 7))
                        cursor += 4
                        continue
                    if not (
                        bitmap_workspace[worker, node >> 3]
                        & (1 << (node & 7))
                    ):
                        bitmap_workspace[
                            worker, node >> 3
                        ] |= <uint8_t>(1 << (node & 7))
                    cursor += 4
            else:
                for byte_index in range(bytes_per_row):
                    bitmap_workspace[worker, byte_index] |= previous_payload[
                        neighbor_row_start + byte_index
                    ]
            adjacency_index += 1
            if omp_get_wtime() >= deadline:
                timed_out = True
                break
        resume_position[0] = adjacency_index

    _finalize_partial_next_hop_bitmap(
        adjacency_offsets,
        adjacency_neighbors,
        previous_offsets,
        previous_encodings,
        previous_payload,
        two_back_offsets,
        two_back_encodings,
        two_back_payload,
        two_back_is_hop_one,
        bitmap_workspace,
        output,
        local_anchor,
        anchor,
        bytes_per_row,
        worker,
    )
    elapsed_seconds[0] = omp_get_wtime() - started
    if not timed_out:
        resume_position[0] = 0
        resume_strategy[0] = 0
    return timed_out


cdef inline uint64_t _classify_collaborative_strategy(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const int32_t[::1] heavy_row_by_node,
    Py_ssize_t anchor,
    Py_ssize_t bytes_per_row,
    uint64_t collaborative_threshold,
) noexcept nogil:
    """Pack estimated work bytes and strategy into one uint64 value."""
    cdef uint64_t row_start = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef uint64_t cursor
    cdef uint64_t direct_work = 0
    cdef uint64_t union_work = 0
    cdef uint64_t chosen_work_bytes
    cdef Py_ssize_t adjacency_index
    cdef uint32_t node
    cdef uint32_t neighbor
    cdef int32_t heavy_row

    for adjacency_index in range(
        adjacency_offsets[anchor], adjacency_offsets[anchor + 1]
    ):
        neighbor = adjacency_neighbors[adjacency_index]
        union_work += (
            previous_offsets[neighbor + 1] - previous_offsets[neighbor]
        )

    if previous_encodings[anchor] == 0:
        cursor = row_start
        while cursor < row_end:
            node = _read_uint32_le(previous_payload, cursor)
            heavy_row = heavy_row_by_node[node]
            if heavy_row >= 0:
                direct_work += bytes_per_row // 4
            else:
                direct_work += (
                    adjacency_offsets[node + 1] - adjacency_offsets[node]
                )
            cursor += 4

        if direct_work * 4 <= union_work + bytes_per_row * 2:
            chosen_work_bytes = direct_work * 4
            if chosen_work_bytes > collaborative_threshold:
                return (chosen_work_bytes << 2) | 1
            return chosen_work_bytes << 2

    chosen_work_bytes = union_work
    if chosen_work_bytes > collaborative_threshold:
        return (chosen_work_bytes << 2) | 2
    return chosen_work_bytes << 2


def build_next_hop_from_global_previous_indices(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const int32_t[::1] heavy_row_by_node,
    const uint8_t[:, ::1] heavy_adjacency_bitmaps,
    const uint32_t[::1] anchor_indices,
    int worker_count,
    double anchor_time_limit_seconds=180.0,
    known_rows_array=None,
    known_row_by_node_array=None,
):
    """Build rows, optionally seeded by tier-two transpose knowledge."""
    cdef Py_ssize_t node_count = adjacency_offsets.shape[0] - 1
    cdef Py_ssize_t batch_size = anchor_indices.shape[0]
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef cnp.ndarray[uint8_t, ndim=2] empty_known_array
    cdef const uint8_t[:, ::1] known_rows
    cdef cnp.ndarray[int32_t, ndim=1] empty_known_map
    cdef const int32_t[::1] known_row_by_node
    cdef bint use_known_rows = known_rows_array is not None
    if anchor_time_limit_seconds <= 0:
        raise ValueError("anchor_time_limit_seconds must be positive")

    if use_known_rows:
        if known_rows_array.shape[1] != bytes_per_row:
            raise ValueError("known_rows has the wrong shape")
        if known_row_by_node_array is None or known_row_by_node_array.shape != (node_count,):
            raise ValueError("known_row_by_node has the wrong shape")
        known_rows = known_rows_array
        known_row_by_node = known_row_by_node_array
    else:
        empty_known_array = np.empty((0, 0), dtype=np.uint8)
        empty_known_map = np.empty(0, dtype=np.int32)
        known_rows = empty_known_array
        known_row_by_node = empty_known_map

    cdef cnp.ndarray[uint8_t, ndim=2] bitmap_array = np.empty(
        (worker_count, bytes_per_row), dtype=np.uint8
    )
    cdef uint8_t[:, ::1] bitmap_workspace = bitmap_array
    cdef cnp.ndarray[uint8_t, ndim=1] timed_out_array = np.zeros(
        batch_size, dtype=np.uint8
    )
    cdef uint8_t[::1] timed_out = timed_out_array
    cdef cnp.ndarray[uint64_t, ndim=1] resume_positions_array = np.zeros(
        batch_size, dtype=np.uint64
    )
    cdef uint64_t[::1] resume_positions = resume_positions_array
    cdef cnp.ndarray[uint8_t, ndim=1] resume_strategies_array = np.zeros(
        batch_size, dtype=np.uint8
    )
    cdef uint8_t[::1] resume_strategies = resume_strategies_array
    cdef cnp.ndarray[cnp.float64_t, ndim=1] elapsed_array = np.zeros(
        batch_size, dtype=np.float64
    )
    cdef double[::1] elapsed = elapsed_array
    cdef cnp.ndarray[uint64_t, ndim=1] partial_counts_array = np.zeros(
        batch_size, dtype=np.uint64
    )
    cdef uint64_t[::1] partial_counts = partial_counts_array
    cdef vector[vector[uint32_t]] output
    output.resize(batch_size)
    cdef Py_ssize_t local_anchor
    cdef Py_ssize_t anchor
    cdef int worker

    # No prediction decides whether an anchor may block progress. Every row
    # receives the same wall-clock budget and returns a valid partial bitmap
    # when that budget expires.
    for local_anchor in prange(
        batch_size,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        worker = threadid()
        anchor = anchor_indices[local_anchor]
        timed_out[local_anchor] = _build_anchor_next_hop_timed(
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
            known_rows,
            known_row_by_node,
            use_known_rows,
            bitmap_workspace,
            output,
            local_anchor,
            anchor,
            bytes_per_row,
            worker,
            anchor_time_limit_seconds,
            &resume_positions[local_anchor],
            &resume_strategies[local_anchor],
            &elapsed[local_anchor],
        )

    cdef cnp.ndarray[uint64_t, ndim=1] offsets_array = np.zeros(
        batch_size + 1, dtype=np.uint64
    )
    cdef cnp.ndarray[uint8_t, ndim=1] encodings_array = np.zeros(
        batch_size, dtype=np.uint8
    )
    cdef uint64_t[::1] offsets = offsets_array
    cdef uint8_t[::1] encodings = encodings_array
    cdef Py_ssize_t reachable_count
    cdef Py_ssize_t encoded_size
    cdef Py_ssize_t total_bytes = 0

    for local_anchor in range(batch_size):
        reachable_count = output[local_anchor].size()
        partial_counts[local_anchor] = reachable_count
        if reachable_count * 4 < bytes_per_row:
            encoded_size = reachable_count * 4
            encodings[local_anchor] = 0
        else:
            encoded_size = bytes_per_row
            encodings[local_anchor] = 1
        total_bytes += encoded_size
        offsets[local_anchor + 1] = total_bytes

    cdef cnp.ndarray[uint8_t, ndim=1] payload_array = np.zeros(
        total_bytes, dtype=np.uint8
    )
    cdef uint8_t[::1] payload = payload_array
    _encode_hop_rows(
        output,
        batch_size,
        0,
        bytes_per_row,
        offsets,
        encodings,
        payload,
        worker_count,
    )

    return (
        offsets_array,
        encodings_array,
        payload_array,
        timed_out_array,
        elapsed_array,
        partial_counts_array,
        resume_positions_array,
        resume_strategies_array,
    )


cdef bint _build_hard_anchor_residual_once(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const uint8_t[:, ::1] known_rows,
    uint8_t[:, ::1] bitmap_workspace,
    vector[vector[uint32_t]]& output,
    Py_ssize_t local_anchor,
    Py_ssize_t anchor,
    Py_ssize_t node_count,
    Py_ssize_t bytes_per_row,
    int worker,
    uint64_t routing_threshold,
    double time_limit_seconds,
    double* elapsed_seconds,
) noexcept nogil:
    """Reuse known bits and collect unresolved hard-anchor work once."""
    cdef double started = omp_get_wtime()
    cdef double deadline = started + time_limit_seconds
    cdef uint64_t cursor = previous_offsets[anchor]
    cdef uint64_t row_end = previous_offsets[anchor + 1]
    cdef uint64_t adjacency_index
    cdef uint64_t unresolved_count = 0
    cdef uint64_t checked_count = 0
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint8_t value
    cdef uint32_t frontier_node
    cdef uint32_t destination
    cdef uint8_t anchor_mask = <uint8_t>(1 << (anchor & 7))
    cdef Py_ssize_t anchor_byte = anchor >> 3

    for byte_index in range(bytes_per_row):
        bitmap_workspace[worker, byte_index] = known_rows[
            anchor, byte_index
        ]

    while cursor < row_end if previous_encodings[anchor] == 0 else False:
        frontier_node = _read_uint32_le(previous_payload, cursor)
        for adjacency_index in range(
            adjacency_offsets[frontier_node],
            adjacency_offsets[frontier_node + 1],
        ):
            destination = adjacency_neighbors[adjacency_index]
            if bitmap_workspace[worker, destination >> 3] & (
                1 << (destination & 7)
            ):
                continue
            if known_rows[destination, anchor_byte] & anchor_mask:
                bitmap_workspace[worker, destination >> 3] |= <uint8_t>(
                    1 << (destination & 7)
                )
                continue
            unresolved_count += 1
            if unresolved_count >= routing_threshold:
                elapsed_seconds[0] = omp_get_wtime() - started
                return True
            bitmap_workspace[worker, destination >> 3] |= <uint8_t>(
                1 << (destination & 7)
            )
            checked_count += 1
            if checked_count % 4096 == 0 and omp_get_wtime() >= deadline:
                elapsed_seconds[0] = omp_get_wtime() - started
                return True
        cursor += 4

    if previous_encodings[anchor] != 0:
        cursor = previous_offsets[anchor]
        for byte_index in range(bytes_per_row):
            value = previous_payload[cursor + byte_index]
            if value == 0:
                continue
            for bit_index in range(8):
                if not (value & (1 << bit_index)):
                    continue
                frontier_node = <uint32_t>(byte_index * 8 + bit_index)
                if frontier_node >= node_count:
                    break
                for adjacency_index in range(
                    adjacency_offsets[frontier_node],
                    adjacency_offsets[frontier_node + 1],
                ):
                    destination = adjacency_neighbors[adjacency_index]
                    if bitmap_workspace[worker, destination >> 3] & (
                        1 << (destination & 7)
                    ):
                        continue
                    if known_rows[destination, anchor_byte] & anchor_mask:
                        bitmap_workspace[
                            worker, destination >> 3
                        ] |= <uint8_t>(1 << (destination & 7))
                        continue
                    unresolved_count += 1
                    if unresolved_count >= routing_threshold:
                        elapsed_seconds[0] = omp_get_wtime() - started
                        return True
                    bitmap_workspace[
                        worker, destination >> 3
                    ] |= <uint8_t>(1 << (destination & 7))
                    checked_count += 1
                    if (
                        checked_count % 4096 == 0
                        and omp_get_wtime() >= deadline
                    ):
                        elapsed_seconds[0] = omp_get_wtime() - started
                        return True

    _finalize_partial_next_hop_bitmap(
        adjacency_offsets,
        adjacency_neighbors,
        previous_offsets,
        previous_encodings,
        previous_payload,
        two_back_offsets,
        two_back_encodings,
        two_back_payload,
        two_back_is_hop_one,
        bitmap_workspace,
        output,
        local_anchor,
        anchor,
        bytes_per_row,
        worker,
    )
    elapsed_seconds[0] = omp_get_wtime() - started
    return False


def build_hard_anchor_residual_batch(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const uint32_t[::1] anchor_indices,
    const uint8_t[:, ::1] known_rows,
    uint64_t routing_threshold,
    int worker_count,
    double time_limit_seconds,
):
    """Build manageable hard anchors without recollecting residual work."""
    cdef Py_ssize_t node_count = adjacency_offsets.shape[0] - 1
    cdef Py_ssize_t batch_size = anchor_indices.shape[0]
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef cnp.ndarray[uint8_t, ndim=2] bitmap_array = np.empty(
        (worker_count, bytes_per_row), dtype=np.uint8
    )
    cdef uint8_t[:, ::1] bitmap_workspace = bitmap_array
    cdef cnp.ndarray[uint8_t, ndim=1] deferred_array = np.zeros(
        batch_size, dtype=np.uint8
    )
    cdef uint8_t[::1] deferred = deferred_array
    cdef cnp.ndarray[cnp.float64_t, ndim=1] elapsed_array = np.zeros(
        batch_size, dtype=np.float64
    )
    cdef double[::1] elapsed = elapsed_array
    cdef vector[vector[uint32_t]] output
    output.resize(batch_size)
    cdef Py_ssize_t local_anchor
    cdef int worker

    for local_anchor in prange(
        batch_size,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        worker = threadid()
        deferred[local_anchor] = _build_hard_anchor_residual_once(
            adjacency_offsets,
            adjacency_neighbors,
            previous_offsets,
            previous_encodings,
            previous_payload,
            two_back_offsets,
            two_back_encodings,
            two_back_payload,
            two_back_is_hop_one,
            known_rows,
            bitmap_workspace,
            output,
            local_anchor,
            anchor_indices[local_anchor],
            node_count,
            bytes_per_row,
            worker,
            routing_threshold,
            time_limit_seconds,
            &elapsed[local_anchor],
        )

    cdef cnp.ndarray[uint64_t, ndim=1] offsets_array = np.zeros(
        batch_size + 1, dtype=np.uint64
    )
    cdef cnp.ndarray[uint8_t, ndim=1] encodings_array = np.zeros(
        batch_size, dtype=np.uint8
    )
    cdef cnp.ndarray[uint64_t, ndim=1] counts_array = np.zeros(
        batch_size, dtype=np.uint64
    )
    cdef uint64_t[::1] offsets = offsets_array
    cdef uint8_t[::1] encodings = encodings_array
    cdef uint64_t[::1] counts = counts_array
    cdef Py_ssize_t reachable_count
    cdef Py_ssize_t encoded_size
    cdef Py_ssize_t total_bytes = 0

    for local_anchor in range(batch_size):
        if deferred[local_anchor]:
            offsets[local_anchor + 1] = total_bytes
            continue
        reachable_count = output[local_anchor].size()
        counts[local_anchor] = reachable_count
        if reachable_count * 4 < bytes_per_row:
            encoded_size = reachable_count * 4
            encodings[local_anchor] = 0
        else:
            encoded_size = bytes_per_row
            encodings[local_anchor] = 1
        total_bytes += encoded_size
        offsets[local_anchor + 1] = total_bytes

    cdef cnp.ndarray[uint8_t, ndim=1] payload_array = np.zeros(
        total_bytes, dtype=np.uint8
    )
    cdef uint8_t[::1] payload = payload_array
    _encode_hop_rows(
        output,
        batch_size,
        0,
        bytes_per_row,
        offsets,
        encodings,
        payload,
        worker_count,
    )
    return (
        offsets_array,
        encodings_array,
        payload_array,
        deferred_array,
        elapsed_array,
        counts_array,
    )


def build_next_hop_from_global_previous_batch(
    const uint64_t[::1] adjacency_offsets,
    const uint32_t[::1] adjacency_neighbors,
    const uint64_t[::1] previous_offsets,
    const uint8_t[::1] previous_encodings,
    const uint8_t[::1] previous_payload,
    const uint64_t[::1] two_back_offsets,
    const uint8_t[::1] two_back_encodings,
    const uint8_t[::1] two_back_payload,
    bint two_back_is_hop_one,
    const int32_t[::1] heavy_row_by_node,
    const uint8_t[:, ::1] heavy_adjacency_bitmaps,
    Py_ssize_t anchor_start,
    Py_ssize_t anchor_end,
    int worker_count,
    double anchor_time_limit_seconds=180.0,
):
    """Backward-compatible contiguous-range wrapper."""
    anchor_indices = np.arange(
        anchor_start, anchor_end, dtype=np.uint32
    )
    return build_next_hop_from_global_previous_indices(
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
        anchor_indices,
        worker_count,
        anchor_time_limit_seconds,
    )


cdef uint64_t _estimate_anchor_frontier_degree_work(
    const uint64_t[::1] adjacency_offsets,
    const uint64_t[::1] frontier_offsets,
    const uint8_t[::1] frontier_encodings,
    const uint8_t[::1] frontier_payload,
    Py_ssize_t anchor,
    Py_ssize_t node_count,
    Py_ssize_t bytes_per_row,
) noexcept nogil:
    cdef Py_ssize_t byte_index
    cdef int bit_index
    cdef uint64_t cursor = frontier_offsets[anchor]
    cdef uint64_t row_end = frontier_offsets[anchor + 1]
    cdef uint32_t node
    cdef uint8_t value
    cdef uint64_t total = 0
    if frontier_encodings[anchor] == 0:
        while cursor < row_end:
            node = _read_uint32_le(frontier_payload, cursor)
            total += (
                adjacency_offsets[node + 1] - adjacency_offsets[node]
            )
            cursor += 4
    else:
        for byte_index in range(bytes_per_row):
            value = frontier_payload[cursor + byte_index]
            if value == 0:
                continue
            for bit_index in range(8):
                if value & (1 << bit_index):
                    node = <uint32_t>(byte_index * 8 + bit_index)
                    if node < node_count:
                        total += (
                            adjacency_offsets[node + 1]
                            - adjacency_offsets[node]
                        )
    return total


def estimate_frontier_degree_work(
    const uint64_t[::1] adjacency_offsets,
    const uint64_t[::1] frontier_offsets,
    const uint8_t[::1] frontier_encodings,
    const uint8_t[::1] frontier_payload,
    int worker_count,
):
    """Sum cached one-hop degrees over every anchor's current frontier."""
    cdef Py_ssize_t anchor_count = frontier_encodings.shape[0]
    cdef Py_ssize_t node_count = adjacency_offsets.shape[0] - 1
    cdef Py_ssize_t bytes_per_row = (node_count + 7) // 8
    cdef cnp.ndarray[uint64_t, ndim=1] work_array = np.zeros(
        anchor_count, dtype=np.uint64
    )
    cdef uint64_t[::1] work = work_array
    cdef Py_ssize_t anchor

    for anchor in prange(
        anchor_count,
        nogil=True,
        schedule="dynamic",
        num_threads=worker_count,
    ):
        work[anchor] = _estimate_anchor_frontier_degree_work(
            adjacency_offsets,
            frontier_offsets,
            frontier_encodings,
            frontier_payload,
            anchor,
            node_count,
            bytes_per_row,
        )

    return work_array
