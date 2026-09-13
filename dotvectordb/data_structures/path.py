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
    def __init__(self, tokenizer, max_tokens, top_k=40, BUFFER_SIZE=40):
        """
        top k: The number of top paths to maintain in memory.
        BUFFER_SIZE: The number of candidate paths to buffer before promoting the best ones.
        tokenizer: A tokenizer instance to estimate token counts for batching.
        max_tokens: The maximum number of tokens allowed in a batch.

        Additional:
        path_buffer: A list to hold candidate paths before they are promoted to the top paths.
        current_top_paths: A list to hold the current top paths based on usefulness scores.
        path_trie: A PathTrie instance to ensure we don't repeat paths with the same relationship sequence or paths
        that don't add new information (i.e., shorter paths that are subsets of longer paths). It also saves money since
        LLMs won't re-judge paths that are already judged.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        self.top_k = top_k
        self.BUFFER_SIZE = BUFFER_SIZE
        self.path_buffer = []
        self.current_top_paths = []
        self.path_trie = PathTrie()
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
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

        # First, we get the relationship IDs from the path to use as a key for deduplication.
        relationship_ids = self._relationship_ids(path)

        # We check if the relationship IDs are already in the trie. If they are, we reject the candidate.
        accepted, old_value = self.path_trie.add(relationship_ids, value=path)

        # If the candidate was accepted and there was an old value, we remove the old value from the buffer.
        if accepted and old_value is not None:
            self.remove_path_from_buffer(old_value)
            self.remove_path_from_top_paths(old_value)
            
        # Return False if the candidate was rejected, True if it was accepted.
        if not accepted:
            return False
        self.path_buffer.append(path)
        return True
    def remove_path_from_buffer(self, path):
        """Remove a path from the candidate buffer."""
        if path in self.path_buffer:
            self.path_buffer.remove(path)
    def remove_path_from_top_paths(self, path):
        """Remove a path from the current top paths."""
        if path in self.current_top_paths:
            self.current_top_paths.remove(path)
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
    def prepare_batches(self, batch_size, judge_prompt):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(judge_prompt, bool) or not isinstance(judge_prompt, str) or not judge_prompt.strip():
            raise ValueError("judge_prompt must be a non-empty string")
        if (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer")
        buffer_json = self.get_buffer_json()
        judge_prompt_token_size = len(self.tokenizer.encode(judge_prompt))
        batches = []
        for start_index in range(0, len(buffer_json), batch_size):
            buffer_batch = buffer_json[start_index:start_index + batch_size]
            original_indices = [item["id"] for item in buffer_batch]
            batch = [
                {
                    "id": batch_index,
                    "path_text": item["path_text"],
                    "relational_description": item["relational_description"],
                }
                for batch_index, item in enumerate(buffer_batch)
            ]
            batch_token_size = len(
            self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            approx_tokens = judge_prompt_token_size + batch_token_size
            if approx_tokens < self.max_tokens:   
                batches.append((batch, original_indices))
            else:
                overflow = approx_tokens - self.max_tokens
                extra_batches = self.fix_batch(approx_tokens, overflow, batch, original_indices)
                for batch, original_indices in extra_batches:
                    batches.append((batch, original_indices))
        return batches
    def fix_batch(
        self,
        approx_tokens,
        overflow,
        batch,
        original_indices,
    ):
        reduction_percentage = (overflow / approx_tokens) * 100
        # Try removing optional data unless the batch is far over
        # and can still be split.
        if reduction_percentage < 50 or len(batch) == 1:
            tokens_before = len(
                self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            self.remove_path_text_from_batch(batch)
            tokens_after = len(
                self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            tokens_removed = tokens_before - tokens_after
            approx_tokens -= tokens_removed
            overflow -= tokens_removed
            if overflow <= 0:
                return [(batch, original_indices)]
            if reduction_percentage >= 10:
                overflow_before = overflow
                overflow = self.remove_extra_evidence_from_batch(
                    batch,
                    overflow,
                )
                approx_tokens -= overflow_before - overflow
                if overflow <= 0:
                    return [(batch, original_indices)]
        # Trimming did not work, so split the batch.
        if len(batch) > 1:
            current_batch_tokens = len(
                self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            batch_size = self.halve_batch_size(len(batch))
            extra_batches = []

            for start_index in range(0, len(batch), batch_size):
                split_batch = batch[
                    start_index:start_index + batch_size
                ]
                split_original_indices = original_indices[
                    start_index:start_index + batch_size
                ]

                for batch_index, relationship in enumerate(split_batch):
                    relationship["id"] = batch_index
                split_batch_tokens = len(
                    self.tokenizer.encode(
                        json.dumps(split_batch, indent=4)
                    )
                )
                split_approx_tokens = (
                    approx_tokens
                    - current_batch_tokens
                    + split_batch_tokens
                )
                if split_approx_tokens <= self.max_tokens:
                    extra_batches.append(
                        (split_batch, split_original_indices)
                    )
                    continue
                split_overflow = (
                    split_approx_tokens - self.max_tokens
                )
                extra_batches.extend(
                    self.fix_batch(
                        split_approx_tokens,
                        split_overflow,
                        split_batch,
                        split_original_indices,
                    )
                )
            return extra_batches
        raise ValueError(
            "A single path could not be reduced enough to fit "
            f"within the {self.max_tokens}-token limit"
        )
    def halve_batch_size(self, batch_size):
        return max(1, batch_size // 2)
    def remove_path_text_from_batch(self, batch):
        for relationship in batch:
            if "path_text" in relationship:
                del relationship["path_text"]
    def remove_extra_evidence_from_batch(self, batch, overflow):
        for relationship in batch:
            description = relationship["relational_description"]
            if "<SEP>" not in description:
                continue
            tokens_before = len(
                self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            before_sep, after_sep = description.split("<SEP>", 1)
            if "]" in after_sep:
                _, remaining_description = after_sep.split("]", 1)
                relationship["relational_description"] = (
                    before_sep + "]" + remaining_description
                )
            else:
                relationship["relational_description"] = before_sep
            tokens_after = len(  
                self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            overflow -= tokens_before - tokens_after
            if overflow <= 0:
                return 0
        return overflow
    def save_paths(self, choices, original_indices):
        for choice in choices:
            batch_id = choice["id"]
            original_index = original_indices[batch_id]
            self.path_buffer[original_index].usefulness_score = choice["judgement"]
