import json
import random
import numpy as np
# from model2vec import StaticModel
from .trie import PathTrie
class Node:
    __slots__ = ("entity_id", "entity_name", "entity_type", "entity_description", "path_count")
    def __init__(self, entity_id, entity_name, entity_type, entity_description):
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.entity_description = entity_description
        self.path_count = 0
class Edge:
    __slots__ = ("rel_id", "src_entity", "dst_entity", "description")
    def __init__(self, rel_id, src_entity, dst_entity, description):
        self.rel_id = rel_id
        self.src_entity = src_entity
        self.dst_entity = dst_entity
        self.description = description
class Path:
    __slots__ = ("entities","relationships", "relational_description", "path_text", "usefulness_score", "weight")
    def __init__(self, entities, relationships, description, weight=None):
        self.entities = entities
        self.relationships = relationships
        self.relational_description = description
        self.path_text = " -> ".join(str(e.entity_name) for e in entities)
        self.usefulness_score = [0.0, 0.0, 0.0]
        self.weight = weight
class PathManager:
    """Keep new candidate paths separate from the current best paths."""
    _description_model = None

    def __init__(self, tokenizer, max_tokens, top_k=40, BUFFER_SIZE=40, query=None):
        """
        top k: The number of top paths to maintain in memory.
        BUFFER_SIZE: The number of candidate paths to buffer before promoting the best ones.
        tokenizer: A tokenizer instance to estimate token counts for batching.
        max_tokens: The maximum number of tokens allowed in a batch.

        Additional:
        path_buffer: A list to hold candidate paths before they are promoted to the top paths.
        current_top_paths: A dictionary of top-path lists keyed by id.
        path_trie: A dictionary of PathTrie instances keyed by id to ensure
        exact or reversed relationship sequences are not sent to the LLM
        judge more than once within the same DOT.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        self.top_k = top_k
        self.BUFFER_SIZE = BUFFER_SIZE
        self.path_buffer = {}
        self.current_top_paths = {}
        self.path_trie = {}
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.query = query
        self.small_llm = None
        self.query_embedding = None
        self.entity_description_cache = {}
        # if self.query:
        #     if PathManager._description_model is None:
        #         PathManager._description_model = StaticModel.from_pretrained(
        #             "minishlab/potion-base-32M"
        #         )
        #     self.small_llm = PathManager._description_model
        #     self.query_embedding = self.small_llm.encode([self.query])[0]
        #     query_norm = np.linalg.norm(self.query_embedding)
        #     if query_norm:
        #         self.query_embedding = self.query_embedding / query_norm
    @staticmethod
    def _relationship_ids(path):
        """Return the ordered relationship sequence used for deduplication."""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path instance")
        if not path.relationships:
            raise ValueError("path must contain at least one relationship")
        return tuple(relationship.rel_id for relationship in path.relationships)
    @staticmethod
    def _validate_usefulness_score(score):
        if not isinstance(score, (list, tuple)) or len(score) != 3:
            raise ValueError(
                "usefulness_score must contain exactly three scores"
            )
        normalized_score = []
        for dimension_score in score:
            if (
                isinstance(dimension_score, bool)
                or not isinstance(dimension_score, (int, float))
                or not 0 <= dimension_score <= 100
            ):
                raise ValueError(
                    "Each usefulness score must be a number from 0 to 100"
                )
            normalized_score.append(float(dimension_score))
        return normalized_score
    @staticmethod
    def _usefulness_average(path):
        score = PathManager._validate_usefulness_score(
            path.usefulness_score
        )
        return sum(score) / len(score)
    def prepare_buffer(self, id):
        """Initialize path state for a given id."""
        if id not in self.path_buffer:
            self.path_buffer[id] = []
        if id not in self.current_top_paths:
            self.current_top_paths[id] = []
        if id not in self.path_trie:
            self.path_trie[id] = PathTrie()
    def add_path(self, id, path):
        """Add a previously unseen relationship path to the candidate buffer.
        A candidate is redundant when its relationship sequence is already a
        prefix represented by the trie. The returned value indicates whether
        the candidate was accepted.
        """
        if len(self.path_buffer[id]) >= self.BUFFER_SIZE:
            return

        # First, we get the relationship IDs from the path to use as a key for deduplication.
        relationship_ids = self._relationship_ids(path)

        # We check if the relationship IDs are already in the trie. If they are, we reject the candidate.
        accepted, old_value = self.path_trie[id].add(
            relationship_ids,
            value=path,
        )

        # If the candidate was accepted and there was an old value, we remove the old value from the buffer.
        if accepted and old_value is not None:
            self.remove_path_from_buffer(id, old_value)
            self.remove_path_from_top_paths(id, old_value)
            
        # Return False if the candidate was rejected, True if it was accepted.
        if not accepted:
            return False
        self.path_buffer[id].append(path)
        return True
    def remove_path_from_buffer(self, id, path):
        """Remove a path from the candidate buffer for an id."""
        if path in self.path_buffer[id]:
            self.path_buffer[id].remove(path)
            return True
        return False
    def remove_path_from_top_paths(self, id, path):
        """Remove a path from the current top paths for an id."""
        if path in self.current_top_paths[id]:
            self.current_top_paths[id].remove(path)
    def build_and_add_path(self, id, graph, vertex_indices, weight=None):
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
                entity_description=vertex["entity_description"],
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
                    f"[{previous_entity.entity_description}] "
                    f"is related to "
                    f"**{entity.entity_name} ({entity.entity_type})** "
                    f"[{entity.entity_description}] "
                    f"because [{relationship.description}]."
                )
        path = Path(
            entities=entities,
            relationships=relationships,
            description=relational_description,
            weight=weight,
        )
        return self.add_path(id, path)
    def update_top_paths(self, id, k=None):
        """Promote the best buffered paths and clear the candidate buffer."""
        if k is None:
            k = self.top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        candidates = self.current_top_paths[id] + self.path_buffer[id]
        self.current_top_paths[id] = sorted(
            candidates,
            key=self._usefulness_average,
            reverse=True,
        )[:k]
        self.path_buffer[id].clear()
        return list(self.current_top_paths[id])
    def get_top_k_helpful_paths(self, id, k=None):
        """Return a snapshot of the current top paths for an id."""
        if k is None:
            k = self.top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        return list(self.current_top_paths[id][:k])
    def get_entities_from_paths(self, path):
        """Return the name and description of every entity in a path."""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path instance")
        return [
            {
                "name": entity.entity_name,
                "description": entity.entity_description,
            }
            for entity in path.entities
        ]
    def get_relationships_from_paths(self, path):
        """Return the source, destination, and description of every relationship in a path."""
        if not isinstance(path, Path):
            raise TypeError("path must be a Path instance")
        return [
            {
                "src_name": relationship.src_entity.entity_name,
                "dst_name": relationship.dst_entity.entity_name,
                "description": relationship.description,
            }
            for relationship in path.relationships
        ]
    def sample_topk_helpful_paths(self, id, k=None):
        """Sample paths for an id from top-path memory without replacement."""
        if k is None:
            k = self.top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        return random.sample(
            self.current_top_paths[id],
            k=min(k, len(self.current_top_paths[id])),
        )
    def get_buffer_json(self, id):
        return [
            {
                "id": index,
                "path_text": path.path_text,
                "relational_description": path.relational_description,
            }
            for index, path in enumerate(self.path_buffer[id])
        ]
    def prepare_batches(self, id, batch_size, judge_prompt):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(judge_prompt, bool) or not isinstance(judge_prompt, str) or not judge_prompt.strip():
            raise ValueError("judge_prompt must be a non-empty string")
        if (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer")

        judge_prompt_token_size = len(self.tokenizer.encode(judge_prompt))
        batches = []

        current_paths = []
        current_entities = {}
        current_relationships = {}
        current_original_indices = []

        def select_entity_description(description):
            if not description:
                return ""
            if description in self.entity_description_cache:
                return self.entity_description_cache[description]
            definitions = [
                definition.strip()
                for definition in description.split("<SEP>")
                if definition.strip()
            ]
            if not definitions:
                return ""
            if len(definitions) == 1:
                selected_definition = definitions[0]
            elif self.query_embedding is None:
                selected_definition = definitions[0]
            else:
                definition_embeddings = self.small_llm.encode(definitions)
                definition_norms = np.linalg.norm(
                    definition_embeddings,
                    axis=1,
                    keepdims=True,
                )
                definition_norms[definition_norms == 0] = 1.0
                definition_embeddings = (
                    definition_embeddings / definition_norms
                )
                similarities = definition_embeddings @ self.query_embedding
                selected_definition = definitions[int(np.argmax(similarities))]
            self.entity_description_cache[description] = selected_definition
            return selected_definition

        def make_batch(paths, entities, relationships):
            return {
                "paths": paths,
                "entities": list(entities.values()),
                "relationships": list(relationships.values()),
            }

        def batch_fits(batch):
            batch_token_size = len(
                self.tokenizer.encode(json.dumps(batch, indent=4))
            )
            return judge_prompt_token_size + batch_token_size < self.max_tokens

        for original_index, path in enumerate(self.path_buffer[id]):
            if len(current_paths) >= batch_size:
                batches.append((
                    make_batch(
                        current_paths,
                        current_entities,
                        current_relationships,
                    ),
                    current_original_indices,
                ))
                current_paths = []
                current_entities = {}
                current_relationships = {}
                current_original_indices = []

            candidate_paths = list(current_paths)
            candidate_entities = dict(current_entities)
            candidate_relationships = dict(current_relationships)

            candidate_paths.append({
                "id": len(candidate_paths),
                "path_text": path.path_text,
            })

            for entity in path.entities:
                if entity.entity_id not in candidate_entities:
                    candidate_entities[entity.entity_id] = {
                        "name": entity.entity_name,
                        "description": select_entity_description(
                            entity.entity_description
                        ),
                    }

            for relationship in path.relationships:
                if relationship.rel_id not in candidate_relationships:
                    candidate_relationships[relationship.rel_id] = {
                        "src_name": relationship.src_entity.entity_name,
                        "dst_name": relationship.dst_entity.entity_name,
                        "description": relationship.description,
                    }

            candidate_batch = make_batch(
                candidate_paths,
                candidate_entities,
                candidate_relationships,
            )

            if batch_fits(candidate_batch):
                current_paths = candidate_paths
                current_entities = candidate_entities
                current_relationships = candidate_relationships
                current_original_indices.append(original_index)
                continue

            if current_paths:
                batches.append((
                    make_batch(
                        current_paths,
                        current_entities,
                        current_relationships,
                    ),
                    current_original_indices,
                ))

            single_path_entities = {
                entity.entity_id: {
                    "name": entity.entity_name,
                    "description": select_entity_description(
                        entity.entity_description
                    ),
                }
                for entity in path.entities
            }
            single_path_relationships = {
                relationship.rel_id: {
                    "src_name": relationship.src_entity.entity_name,
                    "dst_name": relationship.dst_entity.entity_name,
                    "description": relationship.description,
                }
                for relationship in path.relationships
            }
            single_path_batch = make_batch(
                [{"id": 0, "path_text": path.path_text}],
                single_path_entities,
                single_path_relationships,
            )

            if not batch_fits(single_path_batch):
                raise ValueError(
                    "A single path and its required entity/relationship "
                    f"evidence could not fit within the {self.max_tokens}-token limit"
                )

            current_paths = single_path_batch["paths"]
            current_entities = single_path_entities
            current_relationships = single_path_relationships
            current_original_indices = [original_index]

        if current_paths:
            batches.append((
                make_batch(
                    current_paths,
                    current_entities,
                    current_relationships,
                ),
                current_original_indices,
            ))

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
        print("Offending batch:", batch)
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
        while overflow > 0:
            removed_evidence = False
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
                removed_evidence = True
                if overflow <= 0:
                    return 0
            if not removed_evidence:
                break
        return overflow
    def save_paths(self, id, choices, original_indices):
        if isinstance(choices, dict):
            choices = choices["judgements"]

        for choice in choices:
            batch_id = choice["id"]
            original_index = original_indices[batch_id]
            ordered_judgements = sorted(
                choice["judgements"],
                key=lambda judgement: judgement["checklist_item"],
            )
            checklist_items = [
                judgement["checklist_item"]
                for judgement in ordered_judgements
            ]
            if checklist_items != [1, 2, 3]:
                raise ValueError(
                    "Each path must contain checklist items 1, 2, and 3 "
                    "exactly once"
                )
            usefulness_score = [
                judgement["judgement"]
                for judgement in ordered_judgements
            ]
            self.path_buffer[id][original_index].usefulness_score = (
                self._validate_usefulness_score(usefulness_score)
            )
