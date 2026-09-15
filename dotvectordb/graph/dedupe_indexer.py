import json

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class DedupeIndexer:
    def __init__(self, conn, lock):
        self.conn = conn
        self.lock = lock

    def _decode_embedding(self, embedding):
        if isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32, copy=False).reshape(-1)

        if isinstance(embedding, (bytes, bytearray, memoryview)):
            return np.frombuffer(embedding, dtype=np.float32).copy()

        if isinstance(embedding, str):
            return np.asarray(json.loads(embedding), dtype=np.float32).reshape(-1)

        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    async def get_entity_embedding_matrix(self):
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT entity.entity_id, entity.entity_name, entity_embeddings.embedding
                FROM entity
                JOIN entity_embeddings
                ON entity_embeddings.rowid = entity.entity_id
                WHERE entity.is_embedded = 1
                ORDER BY entity.entity_id
                """
            )
            rows = await cursor.fetchall()
            await cursor.close()

        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)

        entities = []
        embeddings = []
        expected_dimension = None

        for entity_id, entity_name, embedding in rows:
            vector = self._decode_embedding(embedding)
            if expected_dimension is None:
                expected_dimension = vector.size
            elif vector.size != expected_dimension:
                raise ValueError(
                    f"Entity {entity_id} has embedding dimension {vector.size}; "
                    f"expected {expected_dimension}."
                )

            if not np.all(np.isfinite(vector)):
                raise ValueError(f"Entity {entity_id} has a non-finite embedding.")
            if np.linalg.norm(vector) == 0:
                raise ValueError(f"Entity {entity_id} has a zero-length embedding.")

            entities.append({"entity_id": entity_id, "entity_name": entity_name})
            embeddings.append(vector)

        return entities, np.vstack(embeddings)

    async def cluster_entity_embeddings(
        self,
        similarity_threshold=0.60,
        linkage="average",
        verbose=True,
    ):
        if not 0 < similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be greater than 0 and at most 1")

        entities, embedding_matrix = await self.get_entity_embedding_matrix()
        if len(entities) < 2:
            if verbose:
                print("Entity deduplication clusters: no candidate clusters found.")
            return []

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=linkage,
            distance_threshold=1.0 - similarity_threshold,
            compute_full_tree=True,
        )
        labels = clustering.fit_predict(embedding_matrix)

        members_by_label = {}
        for entity, label in zip(entities, labels):
            members_by_label.setdefault(int(label), []).append(entity)

        clusters = []
        for members in members_by_label.values():
            if len(members) < 2:
                continue
            clusters.append(
                {
                    "cluster_id": len(clusters) + 1,
                    "entities": members,
                }
            )

        clusters.sort(
            key=lambda cluster: (
                -len(cluster["entities"]),
                cluster["entities"][0]["entity_name"],
            )
        )
        for cluster_id, cluster in enumerate(clusters, start=1):
            cluster["cluster_id"] = cluster_id

        if verbose:
            print(
                f"Entity deduplication clusters at similarity >= "
                f"{similarity_threshold:.2f}: {len(clusters)}"
            )
            for cluster in clusters:
                names = [entity["entity_name"] for entity in cluster["entities"]]
                print(f"Cluster {cluster['cluster_id']}: {names}")

        return clusters
