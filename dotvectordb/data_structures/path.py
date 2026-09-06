import json
import random
from .trie import PathTrie
class Node:
    __slots__ = ("entity_id", "entity_name", "entity_type", "path_count")
    def __init__(self, entity_id, entity_name, entity_type):
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.path_count = 0
class Edge:
    __slots__ = ("rel_id", "src_entity", "dst_entity", "description")
    def __init__(self, rel_id, src_entity, dst_entity, description):
        self.rel_id = rel_id
        self.src_entity = src_entity
        self.dst_entity = dst_entity
        self.description = description
class Path:
    __slots__ = ("entities","relationships", "relational_description", "path_text", "usefulness_score")
    def __init__(self, entities, relationships, description):
        self.entities = entities
        self.relationships = relationships
        self.relational_description = description
        self.path_text = " -> ".join(str(e.entity_name) for e in entities)
        self.usefulness_score = 0.0
class PathManager:
    """Keep new candidate paths separate from the current best paths."""
    def __init__(self, top_k=10, BUFFER_SIZE=40):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        self.top_k = top_k
        self.BUFFER_SIZE = BUFFER_SIZE
        self.path_buffer = []
        self.current_top_paths = []
        self.path_trie = PathTrie()
    @staticmethod
    def _relationship_ids(path):
        """Return the ordered relationship sequence used for deduplication."""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path instance")
        if not path.relationships:
            raise ValueError("path must contain at least one relationship")
        return tuple(relationship.rel_id for relationship in path.relationships)
    def add_path(self, path):
        """Add a previously unseen relationship path to the candidate buffer.
        A candidate is redundant when its relationship sequence is already a
        prefix represented by the trie. The returned value indicates whether
        the candidate was accepted.
        """
        if len(self.path_buffer) >= self.BUFFER_SIZE:
            return
        relationship_ids = self._relationship_ids(path)
        if relationship_ids in self.path_trie:
            return False
        self.path_trie.add(relationship_ids)
        self.path_buffer.append(path)
        for entity in path.entities:
            entity.path_count += 1
        return 
    def build_and_add_path(self, graph, vertex_indices):
        """Build a Path from an igraph vertex route and add it to the buffer."""
        if len(vertex_indices) < 2:
            raise ValueError("vertex_indices must contain at least two vertices")
        entities = []
        relationships = []
        relational_description = ""
        for vertex_index in vertex_indices:
            vertex = graph.vs[vertex_index]
            entity = Node(
                entity_id=vertex["name"],
                entity_name=vertex["entity_name"],
                entity_type=vertex["entity_type"],
            )
            entities.append(entity)
            if len(entities) > 1:
                previous_vertex_index = vertex_indices[len(entities) - 2]
                edge_id = graph.get_eid(
                    previous_vertex_index,
                    vertex_index,
                    directed=True,
                    error=False,
                )
                if edge_id == -1:
                    edge_id = graph.get_eid(
                        vertex_index,
                        previous_vertex_index,
                        directed=True,
                        error=False,
                    )
                if edge_id == -1:
                    raise ValueError(
                        "consecutive vertices in vertex_indices are not connected"
                    )
                relationship_data = graph.es[edge_id]
                previous_entity = entities[-2]
                relationship = Edge(
                    rel_id=relationship_data["relation_id"],
                    src_entity=previous_entity,
                    dst_entity=entity,
                    description=relationship_data["relation_description"],
                )
                relationships.append(relationship)
                relational_description += (
                    f"**{previous_entity.entity_name} ({previous_entity.entity_type})** "
                    f"is related to "
                    f"**{entity.entity_name} ({entity.entity_type})** "
                    f"because [{relationship.description}]."
                )
        path = Path(
            entities=entities,
            relationships=relationships,
            description=relational_description,
        )
        return self.add_path(path)
    def update_top_paths(self, k=None):
        """Promote the best buffered paths and clear the candidate buffer."""
        if k is None:
            k = self.top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        candidates = self.current_top_paths + self.path_buffer
        self.current_top_paths = sorted(
            candidates,
            key=lambda path: path.usefulness_score,
            reverse=True,
        )[:k]
        self.path_buffer.clear()
        return list(self.current_top_paths)
    def get_top_k_helpful_paths(self, k=None):
        """Return a snapshot of the current top-path memory."""
        if k is None:
            k = self.top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        return list(self.current_top_paths[:k])
    def sample_topk_helpful_paths(self, k=None):
        """Sample paths from the current top-path memory without replacement."""
        if k is None:
            k = self.top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        return random.sample(
            self.current_top_paths,
            k=min(k, len(self.current_top_paths)),
        )
    def get_buffer_json(self):
        return [
            {
                "id": index,
                "path_text": path.path_text,
                "relational_description": path.relational_description,
            }
            for index, path in enumerate(self.path_buffer)
        ]
