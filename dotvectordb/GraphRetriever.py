"""
Author: Larnell Moore
Purpose: This module provides the GraphRetriever class, which extends the DotVectorDB class to include advanced search capabilities. It allows users to perform BM25 and vector-based searches on chunks, entities, and relationships within the database. The class also supports hybrid search, combining both BM25 and vector search results for improved retrieval performance.
"""
from .DotVDB import DotVectorDB
import traceback
import re
import asyncio
import math
import numpy as np
from pathlib import Path
from .data_structures.bitmap.RoaringBitMapRetriever import (
    OneHopBitmapRetriever,
    ExactHopBitmapRetriever,
)
class GraphRetriever(DotVectorDB):
    def __init__(
        self,
        *args,
        bitmap_mode: str = "one_hop",
        load_all_into_memory: bool = False,
        **kwargs,
    ) -> None:
        # Initialize DotVectorDB normally with every supplied argument.
        super().__init__(*args, **kwargs)

        self.bitmap_mode = bitmap_mode

        # Preserve the existing DotRAG bitmap path convention.
        database_path = Path(self.path).resolve()
        graph_path = database_path.with_suffix("")
        self.bitmap_path = Path(
            f"{graph_path}__graph.pkl.memory_map"
        )

        if not self.bitmap_path.is_dir():
            raise FileNotFoundError(
                f"Bitmap index not found: {self.bitmap_path}"
            )

        if bitmap_mode == "one_hop":
            self.bitmap_retriever = OneHopBitmapRetriever(
                self.bitmap_path,
                load_all_into_memory=load_all_into_memory,
            )

        elif bitmap_mode == "exact_hop":
            self.bitmap_retriever = ExactHopBitmapRetriever(
                self.bitmap_path,
                load_all_into_memory=load_all_into_memory,
            )

        else:
            raise ValueError(
                "bitmap_mode must be 'one_hop' or 'exact_hop'; "
                f"received {bitmap_mode!r}"
            )
    async def search_BM25(self, table_name: str, query: str, top_k: int = 10) -> tuple[list[dict], int]:
            """Searches and ranks rows from a table's BM25 index."""
            try:
                self._require_open_connection()
                if table_name not in {"chunk", "entity", "relationship"}:
                    raise ValueError(
                        "table_name must be 'chunk', 'entity', or 'relationship'"
                    )
                search_terms = query.split()
                if not search_terms:
                    return [], 0
                fts_query = " OR ".join(
                    f'"{term.replace(chr(34), chr(34) * 2)}"'
                    for term in search_terms
                )
                index_name = f"{table_name}_fts"
                cursor = await self.conn.execute(
                    f"""
                    SELECT
                        fts.rowid AS bm25_rowid,
                        records.*,
                        bm25({index_name}) AS bm25_score
                    FROM {index_name} AS fts
                    JOIN {table_name} AS records
                        ON records.rowid = fts.rowid
                    WHERE {index_name} MATCH ?
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    (fts_query, top_k)
                )
                rows = await cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                results = [dict(zip(column_names, row)) for row in rows]
                return_fields = {
                    "chunk": (
                        "bm25_rowid", "chunk_id", "source_id", "content",
                        "bm25_score",
                    ),
                    "entity": (
                        "bm25_rowid", "entity_id", "entity_name",
                        "entity_description", "entity_types", "bm25_score",
                    ),
                    "relationship": (
                        "bm25_rowid", "relation_id", "src_entity_id",
                        "dst_entity_id", "relation_description", "bm25_score",
                    ),
                }
                if table_name == "entity" and results:
                    entity_ids = [result["entity_id"] for result in results]
                    placeholders = ", ".join("?" for _ in entity_ids)
                    entity_types = {entity_id: [] for entity_id in entity_ids}
                    async with self.conn.execute(
                        f"""
                        SELECT entity_id, entity_type
                        FROM entity_to_entity_type_mapping
                        WHERE entity_id IN ({placeholders})
                        ORDER BY entity_id, entity_type
                        """,
                        entity_ids,
                    ) as type_cursor:
                        for entity_id, entity_type in await type_cursor.fetchall():
                            entity_types[entity_id].append(entity_type)
                    for result in results:
                        result["entity_types"] = entity_types[result["entity_id"]]
                results = [
                    {field: result[field] for field in return_fields[table_name]}
                    for result in results
                ]
                return results, 0
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(
                    f"Error in the search_BM25 function inside DotVectorDB: {e}"
                )
    async def batch_get_embeddings(self, ids, table, batch_size=10_000):
        """Return {id: numpy_embedding} for stored embeddings matching the supplied IDs."""
        self._require_open_connection()
        if table not in {"chunk", "entity", "relationship"}:
            raise ValueError("table must be 'chunk', 'entity', or 'relationship'")
        self._require_size_positive(batch_size, "batch_size")
        ids = list(set(ids))
        embeddings = {}
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            async with self.conn.execute(
                f"SELECT rowid, embedding FROM {table}_embeddings WHERE rowid IN ({placeholders})",
                batch,
            ) as cursor:
                for rowid, embedding in await cursor.fetchall():
                    embeddings[rowid] = np.frombuffer(embedding, dtype=np.float32).copy()
        return embeddings
    async def vector_search(self, query, table, top_k=40, embedding_table=None, KNN=True, distance_metric="l2", embedding=None) -> tuple[list[dict], int]:
            """Search using a supplied embedding, or generate one from query when omitted."""
            self._require_open_connection()
            if embedding is None:
                query_embedding, query_tokens = (
                    await self._get_or_create_query_embedding(query)
                )
            else:
                query_embedding, query_tokens = embedding, 0
            distance_func = self._require_distance_func(distance_metric)
            if embedding_table is None:
                embedding_table = f"{table}_embeddings"
            if KNN:
                SQL = f"""
                SELECT
                    v.rowid AS vector_rowid,
                    m.*,
                    1.0 / (1.0 + {distance_func}(:query_embedding, v.embedding)) AS similarity_score
                FROM {embedding_table}(:query_embedding, '{{k: {top_k}}}') AS v
                JOIN {table} AS m
                    ON m.rowid = v.rowid
                ORDER BY similarity_score DESC
                LIMIT :top_k;
                """
            else:
                rerank_k = top_k * 10

                SQL = f"""
                WITH candidates AS (
                    SELECT
                        v.rowid,
                        v.embedding
                    FROM {embedding_table}(:query_embedding, '{{k: {rerank_k}, nprobe: 0.08}}') AS v
                )
                SELECT
                    c.rowid AS vector_rowid,
                    m.*,
                    1.0 / (1.0 + {distance_func}(:query_embedding, c.embedding)) AS similarity_score
                FROM candidates AS c
                JOIN {table} AS m
                    ON m.rowid = c.rowid
                ORDER BY similarity_score DESC
                LIMIT :top_k;
                """
            try: 
                async with self.conn.execute(
                        SQL,
                        {
                            "query_embedding": query_embedding,
                            "top_k": top_k,
                        },
                    ) as cursor:
                        rows = await cursor.fetchall()
                        column_names = [description[0] for description in cursor.description]
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the vector_search function inside DotVectorDB: {e}")
            results = [dict(zip(column_names, row)) for row in rows]
            return_fields = {
                "chunk": (
                    "vector_rowid", "chunk_id", "source_id", "content",
                    "similarity_score",
                ),
                "entity": (
                    "vector_rowid", "entity_id", "entity_name",
                    "entity_description", "entity_types", "similarity_score",
                ),
                "relationship": (
                    "vector_rowid", "relation_id", "src_entity_id",
                    "dst_entity_id", "relation_description", "similarity_score",
                ),
            }
            if table not in return_fields:
                raise ValueError(
                    "table must be 'chunk', 'entity', or 'relationship'"
                )
            if table == "entity" and results:
                entity_ids = [result["entity_id"] for result in results]
                placeholders = ", ".join("?" for _ in entity_ids)
                entity_types = {entity_id: [] for entity_id in entity_ids}
                async with self.conn.execute(
                    f"""
                    SELECT entity_id, entity_type
                    FROM entity_to_entity_type_mapping
                    WHERE entity_id IN ({placeholders})
                    ORDER BY entity_id, entity_type
                    """,
                    entity_ids,
                ) as cursor:
                    for entity_id, entity_type in await cursor.fetchall():
                        entity_types[entity_id].append(entity_type)
                for result in results:
                    result["entity_types"] = entity_types[result["entity_id"]]
            results = [
                {field: result[field] for field in return_fields[table]}
                for result in results
            ]
            return results, query_tokens

    async def hybrid_vector_BM25_search(self, query, table, top_k=40, embedding_table=None, KNN=True, distance_metric="l2", embedding=None) -> tuple[list[dict], int]:
                    """Perform a hybrid search combining vector and BM25 results using reciprocal rank fusion."""
                    self._require_open_connection()
                    try:
                        candidate_k = top_k * 2
                        rrf_k = 60
        
                        (vector_results, query_tokens), (bm25_results, _) = await asyncio.gather(
                            self.vector_search(
                                query=query,
                                table=table,
                                top_k=candidate_k,
                                embedding_table=embedding_table,
                                KNN=KNN,
                                distance_metric=distance_metric,
                                embedding=embedding,
                            ),
                            self.search_BM25(
                                table_name=table,
                                query=query,
                                top_k=candidate_k,
                            ),
                        )
        
                        fused_results = {}
        
                        for rank, result in enumerate(vector_results, start=1):
                            rowid = result["vector_rowid"]
                            fused_results[rowid] = {
                                "result": dict(result),
                                "rrf_score": 1.0 / (rrf_k + rank),
                            }
        
                        for rank, result in enumerate(bm25_results, start=1):
                            rowid = result["bm25_rowid"]
                            rrf_score = 1.0 / (rrf_k + rank)
        
                            if rowid in fused_results:
                                fused_results[rowid]["rrf_score"] += rrf_score
                                fused_results[rowid]["result"]["bm25_rowid"] = rowid
                                fused_results[rowid]["result"]["bm25_score"] = result["bm25_score"]
                            else:
                                fused_results[rowid] = {
                                    "result": dict(result),
                                    "rrf_score": rrf_score,
                                }
                                fused_results[rowid]["result"]["vector_rowid"] = rowid
                                fused_results[rowid]["result"]["similarity_score"] = None
        
                        ranked_results = sorted(
                            fused_results.values(),
                            key=lambda result: result["rrf_score"],
                            reverse=True,
                        )[:top_k]
        
                        results = []
                        for ranked_result in ranked_results:
                            result = ranked_result["result"]
                            result.setdefault("bm25_rowid", result["vector_rowid"])
                            result.setdefault("bm25_score", None)
                            result["rrf_score"] = ranked_result["rrf_score"]
                            results.append(result)
        
                        return_fields = {
                            "chunk": (
                                "vector_rowid", "bm25_rowid", "chunk_id", "source_id",
                                "content", "similarity_score", "bm25_score", "rrf_score",
                            ),
                            "entity": (
                                "vector_rowid", "bm25_rowid", "entity_id", "entity_name",
                                "entity_description", "entity_types", "similarity_score",
                                "bm25_score", "rrf_score",
                            ),
                            "relationship": (
                                "vector_rowid", "bm25_rowid", "relation_id",
                                "src_entity_id", "dst_entity_id", "relation_description",
                                "similarity_score", "bm25_score", "rrf_score",
                            ),
                        }
                        if table not in return_fields:
                            raise ValueError(
                                "table must be 'chunk', 'entity', or 'relationship'"
                            )
                        results = [
                            {field: result[field] for field in return_fields[table]}
                            for result in results
                        ]
                        return results, query_tokens
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(
                            "Error in the hybrid_vector_BM25_search function "
                            f"inside DotVectorDB: {e}"
                        )
        
    

    async def search_id_batch(
        self,
        query,
        table,
        rowids,
        top_k=None,
        embedding_table=None,
        distance_metric="l2",
        search_type="hybrid",
    ) -> tuple[list[dict], int]:
            """If the exact rowids of the relevant rows are known, this function can be used to search and rank them based on a query. The results will be ranked according to the specified search_type, which can be either vector-based, BM25-based, or a hybrid of both. The ``top_k`` parameter limits the number of results returned. The ``embedding_table`` parameter specifies the table containing the embeddings for vector search. The ``distance_metric`` parameter determines how similarity is calculated for vector search.

            ``search_type`` may be ``"vector"``, ``"bm25"``, or
            ``"hybrid"``. Hybrid search combines vector and BM25 ranks with
            reciprocal rank fusion.
            """
            self._require_open_connection()
            if table not in {"chunk", "entity", "relationship"}:
                raise ValueError(
                    "table must be 'chunk', 'entity', or 'relationship'"
                )
            if not isinstance(search_type, str):
                raise TypeError("search_type must be a string")
            search_type = search_type.strip().lower()
            if search_type not in {"vector", "bm25", "hybrid"}:
                raise ValueError(
                    "search_type must be 'vector', 'bm25', or 'hybrid'"
                )
            if not isinstance(query, str):
                raise TypeError("query must be a string")
            if not isinstance(rowids, list):
                raise TypeError("rowids must be a list")
            if not rowids:
                return [], 0
            if any(
                not isinstance(rowid, int) or isinstance(rowid, bool)
                for rowid in rowids
            ):
                raise TypeError("every rowid must be an integer")

            unique_rowids = list(dict.fromkeys(rowids))
            if top_k is None:
                top_k = len(unique_rowids)
            if (
                not isinstance(top_k, int)
                or isinstance(top_k, bool)
                or top_k <= 0
            ):
                raise ValueError("top_k must be a positive integer or None")

            if embedding_table is None:
                embedding_table = f"{table}_embeddings"

            rowid_parameters = {
                f"rowid_{index}": rowid
                for index, rowid in enumerate(unique_rowids)
            }
            rowid_placeholders = ", ".join(
                f":rowid_{index}" for index in range(len(unique_rowids))
            )
            result_limit = min(top_k, len(unique_rowids))

            async def run_vector_search(limit):
                query_embedding, query_tokens = (
                    await self._get_or_create_query_embedding(query)
                )
                distance_func = self._require_distance_func(distance_metric)
                sql = f"""
                SELECT
                    v.rowid AS vector_rowid,
                    m.*,
                    1.0 / (
                        1.0 + {distance_func}(
                            :query_embedding,
                            v.embedding
                        )
                    ) AS similarity_score
                FROM {embedding_table} AS v
                JOIN {table} AS m
                    ON m.rowid = v.rowid
                WHERE v.rowid IN ({rowid_placeholders})
                ORDER BY similarity_score DESC
                LIMIT :top_k;
                """
                parameters = {
                    "query_embedding": query_embedding,
                    "top_k": limit,
                    **rowid_parameters,
                }
                async with self.conn.execute(sql, parameters) as cursor:
                    rows = await cursor.fetchall()
                    column_names = [
                        description[0] for description in cursor.description
                    ]
                return [dict(zip(column_names, row)) for row in rows], query_tokens

            async def run_bm25_search(limit):
                query_terms = list(dict.fromkeys(
                    re.findall(r"\w+", query.casefold())
                ))
                if not query_terms:
                    return []

                # The candidate IDs are already known, so score only their
                # text instead of running a global FTS MATCH and intersecting
                # its potentially huge posting lists afterward.
                sql = f"""
                SELECT
                    records.rowid AS bm25_rowid,
                    records.*
                FROM {table} AS records
                WHERE records.rowid IN ({rowid_placeholders});
                """
                async with self.conn.execute(
                    sql,
                    rowid_parameters,
                ) as cursor:
                    rows = await cursor.fetchall()
                    column_names = [
                        description[0] for description in cursor.description
                    ]
                if not rows:
                    return []

                text_fields = {
                    "chunk": ("content",),
                    "entity": (
                        "entity_name",
                        "entity_description",
                    ),
                    "relationship": ("relation_description",),
                }[table]

                def score_candidates_locally():
                    candidates = [
                        dict(zip(column_names, row)) for row in rows
                    ]
                    tokenized_documents = []
                    document_frequencies = {
                        term: 0 for term in query_terms
                    }
                    for candidate in candidates:
                        document_text = " ".join(
                            str(candidate.get(field) or "")
                            for field in text_fields
                        )
                        document_tokens = re.findall(
                            r"\w+",
                            document_text.casefold(),
                        )
                        tokenized_documents.append(document_tokens)
                        document_terms = set(document_tokens)
                        for term in query_terms:
                            if term in document_terms:
                                document_frequencies[term] += 1

                    document_count = len(candidates)
                    average_document_length = (
                        sum(
                            len(tokens)
                            for tokens in tokenized_documents
                        )
                        / document_count
                    )
                    if average_document_length == 0:
                        return []

                    k1 = 1.5
                    b = 0.75
                    scored_candidates = []
                    for candidate, document_tokens in zip(
                        candidates,
                        tokenized_documents,
                    ):
                        term_frequencies = {}
                        for token in document_tokens:
                            term_frequencies[token] = (
                                term_frequencies.get(token, 0) + 1
                            )
                        length_normalization = (
                            1.0 - b
                            + b
                            * len(document_tokens)
                            / average_document_length
                        )
                        score = 0.0
                        for term in query_terms:
                            term_frequency = term_frequencies.get(term, 0)
                            if term_frequency == 0:
                                continue
                            document_frequency = document_frequencies[term]
                            inverse_document_frequency = math.log(
                                1.0
                                + (
                                    document_count
                                    - document_frequency
                                    + 0.5
                                )
                                / (document_frequency + 0.5)
                            )
                            score += inverse_document_frequency * (
                                term_frequency * (k1 + 1.0)
                                / (
                                    term_frequency
                                    + k1 * length_normalization
                                )
                            )
                        if score > 0:
                            candidate["bm25_score"] = score
                            scored_candidates.append(candidate)

                    scored_candidates.sort(
                        key=lambda candidate: (
                            -candidate["bm25_score"],
                            candidate["bm25_rowid"],
                        )
                    )
                    return scored_candidates[:limit]

                return await asyncio.to_thread(score_candidates_locally)

            vector_return_fields = {
                "chunk": (
                    "vector_rowid", "chunk_id", "source_id", "content",
                    "similarity_score",
                ),
                "entity": (
                    "vector_rowid", "entity_id", "entity_name",
                    "entity_description", "entity_types", "similarity_score",
                ),
                "relationship": (
                    "vector_rowid", "relation_id", "src_entity_id",
                    "dst_entity_id", "relation_description", "similarity_score",
                ),
            }
            bm25_return_fields = {
                "chunk": (
                    "bm25_rowid", "chunk_id", "source_id", "content",
                    "bm25_score",
                ),
                "entity": (
                    "bm25_rowid", "entity_id", "entity_name",
                    "entity_description", "entity_types", "bm25_score",
                ),
                "relationship": (
                    "bm25_rowid", "relation_id", "src_entity_id",
                    "dst_entity_id", "relation_description", "bm25_score",
                ),
            }
            hybrid_return_fields = {
                "chunk": (
                    "vector_rowid", "bm25_rowid", "chunk_id", "source_id",
                    "content", "similarity_score", "bm25_score", "rrf_score",
                ),
                "entity": (
                    "vector_rowid", "bm25_rowid", "entity_id", "entity_name",
                    "entity_description", "entity_types", "similarity_score",
                    "bm25_score", "rrf_score",
                ),
                "relationship": (
                    "vector_rowid", "bm25_rowid", "relation_id",
                    "src_entity_id", "dst_entity_id", "relation_description",
                    "similarity_score", "bm25_score", "rrf_score",
                ),
            }

            try:
                if search_type == "vector":
                    results, query_tokens = await run_vector_search(
                        result_limit
                    )
                    return_fields = vector_return_fields
                elif search_type == "bm25":
                    results = await run_bm25_search(result_limit)
                    query_tokens = 0
                    return_fields = bm25_return_fields
                else:
                    candidate_k = min(
                        top_k * 2,
                        len(unique_rowids),
                    )
                    (vector_results, query_tokens), bm25_results = (
                        await asyncio.gather(
                            run_vector_search(candidate_k),
                            run_bm25_search(candidate_k),
                        )
                    )
                    rrf_k = 60
                    fused_results = {}
                    for rank, result in enumerate(
                        vector_results,
                        start=1,
                    ):
                        rowid = result["vector_rowid"]
                        fused_results[rowid] = {
                            "result": dict(result),
                            "rrf_score": 1.0 / (rrf_k + rank),
                        }
                    for rank, result in enumerate(
                        bm25_results,
                        start=1,
                    ):
                        rowid = result["bm25_rowid"]
                        score = 1.0 / (rrf_k + rank)
                        if rowid in fused_results:
                            fused_results[rowid]["rrf_score"] += score
                            fused_results[rowid]["result"].update({
                                "bm25_rowid": rowid,
                                "bm25_score": result["bm25_score"],
                            })
                        else:
                            bm25_result = dict(result)
                            bm25_result.update({
                                "vector_rowid": rowid,
                                "similarity_score": None,
                            })
                            fused_results[rowid] = {
                                "result": bm25_result,
                                "rrf_score": score,
                            }
                    ranked_results = sorted(
                        fused_results.values(),
                        key=lambda item: item["rrf_score"],
                        reverse=True,
                    )[:result_limit]
                    results = []
                    for ranked_result in ranked_results:
                        result = ranked_result["result"]
                        result.setdefault(
                            "bm25_rowid",
                            result["vector_rowid"],
                        )
                        result.setdefault("bm25_score", None)
                        result["rrf_score"] = ranked_result["rrf_score"]
                        results.append(result)
                    return_fields = hybrid_return_fields
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(
                    "Error in the vector_search_id_batch function inside "
                    f"DotVectorDB: {e}"
                ) from e

            if table == "entity" and results:
                entity_ids = [result["entity_id"] for result in results]
                placeholders = ", ".join("?" for _ in entity_ids)
                entity_types = {entity_id: [] for entity_id in entity_ids}
                async with self.conn.execute(
                    f"""
                    SELECT entity_id, entity_type
                    FROM entity_to_entity_type_mapping
                    WHERE entity_id IN ({placeholders})
                    ORDER BY entity_id, entity_type
                    """,
                    entity_ids,
                ) as cursor:
                    for entity_id, entity_type in await cursor.fetchall():
                        entity_types[entity_id].append(entity_type)
                for result in results:
                    result["entity_types"] = entity_types[result["entity_id"]]
            results = [
                {field: result[field] for field in return_fields[table]}
                for result in results
            ]
            return results, query_tokens

   
