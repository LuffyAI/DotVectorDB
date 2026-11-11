"""
Author: Larnell Moore
Purpose: This module provides the structured GraphConstructor class, which extends the DotVectorDB class to include advanced graph construction capabilities. It allows users to ingest documents, extract and embed entities and relationships, generate global descriptions, and construct a graph representation of the database. The class also supports creating a BM25 index for efficient text retrieval.
"""
import asyncio
import re
import traceback
import unicodedata
import warnings

from tqdm.asyncio import tqdm_asyncio

from UnstructuredGraphConstructor import UnstructuredGraphConstructor
from dotvectordb.graph.entity_indexer import EntityIndexer
from dotvectordb.graph.relationship_indexer import RelationshipIndexer

class StructuredGraphConstructor(UnstructuredGraphConstructor):
        """Construct graphs from structured inputs using the shared pipeline."""
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        @staticmethod
        def __does_entity_name_overlap_chunk(
            entity_name: str,
            chunk_content: str,
        ) -> tuple[bool, list[str]]:
            """Require every normalized entity-name token to occur in the chunk."""
            def normalize(value):
                decomposed = unicodedata.normalize("NFKD", value.casefold())
                return "".join(
                    character
                    for character in decomposed
                    if not unicodedata.combining(character)
                )
            entity_tokens = set(re.findall(r"\w+", normalize(entity_name)))
            chunk_tokens = set(re.findall(r"\w+", normalize(chunk_content)))
            missing_tokens = sorted(entity_tokens - chunk_tokens)
            return bool(entity_tokens) and not missing_tokens, missing_tokens
           
    
    
        def __verify_source_json(self, source_json: list[dict]) -> dict[str, str]:
                    """
                    Expected Schema for the source:
                    [
                        {
                            "source_name": "The name of the source.",
                            "source_description": "A brief description of the source.",
                        }
                    ]
                    Returns a mapping of each unique source name to its generated source ID.
                    """
                    source_name_to_id = {}
                    if not isinstance(source_json, list):
                        raise TypeError("Source JSON must be a list of source objects.")
                    for index, source in enumerate(source_json):
                        if not isinstance(source, dict):
                            raise TypeError(
                                f"Source at index {index} must be an object (dictionary)."
                            )
                        if "source_name" not in source:
                            raise ValueError(
                                f"Source at index {index} is missing the required "
                                "'source_name' key."
                            )
                        source_name = source["source_name"]
                        if not isinstance(source_name, str):
                            raise TypeError(
                                f"'source_name' for source at index {index} must be a string."
                            )
                        if not source_name.strip():
                            raise ValueError(
                                f"'source_name' for source at index {index} cannot be empty."
                            )
                        source_id = str(
                            self.hashing_func.content_hash(source_name)
                        )
                        source["source_id"] = source_id
                        source_name_to_id[source_name] = source_id
                    return source_name_to_id
        def __verify_prechunked_corpus(
                self,
                source_name_to_id: dict[str, str],
                chunk_json_list: list[dict]
            ) -> tuple[dict, list[dict]]:
                """
                Expected Schema for each chunk in the list:
                [
                    {
                        "chunk_id": "Unique identifier for the chunk.",
                        "content": "The actual text content of the chunk.",
                        "source_name": "Unique identifier for the source.",
                    }
                ]
                Returns the local-to-generated chunk ID mapping and the enriched chunk
                JSON list.
                """
                if not isinstance(source_name_to_id, dict):
                    raise TypeError("source_name_to_id must be a dictionary.")
                if not isinstance(chunk_json_list, list):
                    raise TypeError("Chunk JSON must be a list of chunk objects.")
        
                local_chunk_ids_to_new_chunk_ids = {}
                seen_chunk_ids = set()
                seen_content = set()
        
                # Validate the complete input before modifying any chunk dictionaries.
                for index, chunk in enumerate(chunk_json_list):
                    if not isinstance(chunk, dict):
                        raise TypeError(
                            f"Chunk at index {index} must be an object (dictionary)."
                        )
                    if "chunk_id" not in chunk:
                        raise ValueError(
                            f"Chunk at index {index} is missing the required 'chunk_id' key."
                        )
                    local_chunk_id = chunk["chunk_id"]
                    try:
                        if local_chunk_id in seen_chunk_ids:
                            raise ValueError(
                                f"Duplicate chunk_id '{local_chunk_id}' found at index {index}."
                            )
                        seen_chunk_ids.add(local_chunk_id)
                    except TypeError as exc:
                        raise TypeError(
                            f"'chunk_id' for chunk at index {index} must be hashable."
                        ) from exc
                    if "source_name" not in chunk:
                        raise ValueError(
                            f"Chunk '{local_chunk_id}' is missing the required "
                            "'source_name' key."
                        )
                    source_name = chunk["source_name"]
                    if not isinstance(source_name, str):
                        raise TypeError(
                            f"'source_name' for chunk '{local_chunk_id}' must be a string."
                        )
                    if source_name not in source_name_to_id:
                        raise ValueError(
                            f"Chunk '{local_chunk_id}' references unknown source_name "
                            f"'{source_name}'."
                        )
                    if "content" not in chunk:
                        raise ValueError(
                            f"Chunk '{local_chunk_id}' is missing the required 'content' key."
                        )
                    content = chunk["content"]
                    if not isinstance(content, str):
                        raise TypeError(
                            f"'content' for chunk '{local_chunk_id}' must be a string."
                        )
                    if not content.strip():
                        raise ValueError(
                            f"'content' for chunk '{local_chunk_id}' cannot be empty."
                        )
                    if content in seen_content:
                        raise ValueError(
                            f"Duplicate content found for chunk '{local_chunk_id}'."
                        )
                    seen_content.add(content)
                for chunk in chunk_json_list:
                    local_chunk_id = chunk["chunk_id"]
                    new_chunk_id = self.hashing_func.content_hash(
                        chunk["content"]
                    )
                    local_chunk_ids_to_new_chunk_ids[local_chunk_id] = new_chunk_id
                    chunk["chunk_id"] = new_chunk_id
                    chunk["source_id"] = source_name_to_id[chunk["source_name"]]
                return local_chunk_ids_to_new_chunk_ids, chunk_json_list
        def __verify_prebuilt_entity_json(
                self,
                local_chunks_to_new_chunks_ids: dict[object, int],
                chunk_json: list[dict],
                entity_json: list[dict]
            ) -> tuple[dict, list[dict]]:
                """
                TODO: ADD TOKEN COUNTING TO ENSURE IT DOESN'T GO OVER EMBEDDING WINDOW
                        Expected Schema for each entity in the list:
                        [
                            {
                                "entity_id": "Unique identifier for the entity.",
                                "chunk_id": "Unique identifier for the chunk associated with the entity.",
                                "entity_name": "The actual name of the entity.",
                                "entity_description": "The description of the entity.",
                                "entity_types": ["List of types associated with the entity."],
                            }
                        ]
                Returns the local-to-generated entity ID mapping and the enriched
                entity JSON list.
                """
                if not isinstance(local_chunks_to_new_chunks_ids, dict):
                    raise TypeError(
                        "local_chunks_to_new_chunks_ids must be a dictionary."
                    )
                if not isinstance(chunk_json, list):
                    raise TypeError("Chunk JSON must be a list of chunk objects.")
                if not isinstance(entity_json, list):
                    raise TypeError("Entity JSON must be a list of entity objects.")
                chunk_content_by_id = {
                    chunk["chunk_id"]: chunk["content"]
                    for chunk in chunk_json
                }
                local_entity_ids_to_new_entity_ids = {}
                seen_local_entity_ids = set()
                prepared_entities = []
                entities_with_multiple_types = []
                # Validate the complete input before modifying any entity dictionaries.
                for index, entity in enumerate(entity_json):
                    if not isinstance(entity, dict):
                        raise TypeError(
                            f"Entity at index {index} must be an object (dictionary)."
                        )
                    missing_keys = [
                        key
                        for key in ("entity_id", "chunk_id", "entity_name")
                        if key not in entity
                    ]
                    if missing_keys:
                        raise ValueError(
                            f"Entity at index {index} is missing required key(s): "
                            f"{', '.join(missing_keys)}."
                        )
        
                    local_entity_id = entity["entity_id"]
                    try:
                        if local_entity_id in seen_local_entity_ids:
                            raise ValueError(
                                f"Duplicate entity_id '{local_entity_id}' found at "
                                f"index {index}."
                            )
                        seen_local_entity_ids.add(local_entity_id)
                    except TypeError as exc:
                        raise TypeError(
                            f"'entity_id' for entity at index {index} must be hashable."
                        ) from exc
        
                    local_chunk_id = entity["chunk_id"]
                    try:
                        new_chunk_id = local_chunks_to_new_chunks_ids[local_chunk_id]
                    except TypeError as exc:
                        raise TypeError(
                            f"'chunk_id' for entity '{local_entity_id}' must be hashable."
                        ) from exc
                    except KeyError as exc:
                        raise ValueError(
                            f"Entity '{local_entity_id}' references unknown chunk_id "
                            f"'{local_chunk_id}'."
                        ) from exc
        
                    entity_name = entity["entity_name"]
                    if not isinstance(entity_name, str):
                        raise TypeError(
                            f"'entity_name' for entity '{local_entity_id}' must be a string."
                        )
                    normalized_entity_name = entity_name.strip().lower()
                    if not normalized_entity_name:
                        raise ValueError(
                            f"'entity_name' for entity '{local_entity_id}' cannot be empty."
                        )
    
                    name_matches_chunk, missing_name_tokens = (
                        self.__does_entity_name_overlap_chunk(
                            normalized_entity_name,
                            chunk_content_by_id[new_chunk_id],
                        )
                    )
                    if not name_matches_chunk:
                        raise ValueError(
                            f"Entity '{local_entity_id}' claims entity_name "
                            f"'{entity_name}', but that name is not present in "
                            f"chunk '{local_chunk_id}'. Missing normalized token(s): "
                            f"{missing_name_tokens}."
                        )
        
                    entity_description = entity.get("entity_description", "")
                    if entity_description is None:
                        entity_description = ""
                    if not isinstance(entity_description, str):
                        raise TypeError(
                            f"'entity_description' for entity '{local_entity_id}' "
                            "must be a string."
                        )
        
                    entity_types = entity.get("entity_types", [])
                    if entity_types is None:
                        entity_types = []
                    if not isinstance(entity_types, list):
                        raise TypeError(
                            f"'entity_types' for entity '{local_entity_id}' must be a list."
                        )
                    if any(not isinstance(entity_type, str) for entity_type in entity_types):
                        raise TypeError(
                            f"Every entity type for entity '{local_entity_id}' must be "
                            "a string."
                        )
        
                    if len(entity_types) > 1:
                        entities_with_multiple_types.append(local_entity_id)
        
                    new_entity_id = self.hashing_func.content_hash(
                        normalized_entity_name
                    )
                    prepared_entities.append(
                        (
                            entity,
                            local_entity_id,
                            new_entity_id,
                            new_chunk_id,
                            normalized_entity_name,
                            entity_description,
                            entity_types,
                        )
                    )
        
                if entities_with_multiple_types:
                    warnings.warn(
                        "Entities with local entity_id(s) "
                        f"{entities_with_multiple_types} have more than one entity type. "
                        "They are supported by DotRAG but may be incompatible with "
                        "GraphRAG, LightRAG, PathRAG, and other methods that require "
                        "exactly one type per node.",
                        UserWarning,
                        stacklevel=2,
                    )
        
                for (
                    entity,
                    local_entity_id,
                    new_entity_id,
                    new_chunk_id,
                    normalized_entity_name,
                    entity_description,
                    entity_types,
                ) in prepared_entities:
                    local_entity_ids_to_new_entity_ids[local_entity_id] = new_entity_id
                    entity["entity_id"] = new_entity_id
                    entity["chunk_id"] = new_chunk_id
                    entity["entity_name"] = normalized_entity_name
                    entity["entity_description"] = entity_description
                    entity["entity_types"] = entity_types
                    entity["_is_entity_enriched"] = bool(entity_description.strip())
                    entity["does_entity_have_types"] = bool(entity_types)
        
                return local_entity_ids_to_new_entity_ids, entity_json
        def __verify_entity_types(
                self,
                entity_json: list[dict],
                entity_types_json: dict[str, str] = None,
            ) -> dict[str, str]:
                """Verify definitions exist for every custom entity type in use."""
                if not isinstance(entity_json, list):
                    raise TypeError("Entity JSON must be a list of entity objects.")
    
                used_entity_types = {
                    entity_type
                    for entity in entity_json
                    for entity_type in entity.get("entity_types", [])
                }
                if not used_entity_types:
                    if entity_types_json is None:
                        return {}
                    if not isinstance(entity_types_json, dict):
                        raise TypeError(
                            "entity_types_json must be a dictionary mapping entity "
                            "type names to definitions."
                        )
                    return entity_types_json
    
                if entity_types_json is None:
                    raise ValueError(
                        "entity_types_json is required because the entity JSON "
                        "contains entity types."
                    )
                if not isinstance(entity_types_json, dict):
                    raise TypeError(
                        "entity_types_json must be a dictionary mapping entity "
                        "type names to definitions."
                    )
    
                invalid_definitions = [
                    entity_type
                    for entity_type, definition in entity_types_json.items()
                    if not isinstance(entity_type, str)
                    or not entity_type.strip()
                    or not isinstance(definition, str)
                    or not definition.strip()
                ]
                if invalid_definitions:
                    raise ValueError(
                        "Every entity type and definition in entity_types_json "
                        "must be a nonempty string. Invalid type(s): "
                        f"{invalid_definitions}."
                    )
    
                missing_definitions = sorted(
                    used_entity_types - set(entity_types_json)
                )
                if missing_definitions:
                    raise ValueError(
                        "Missing definitions in entity_types_json for entity "
                        f"type(s): {missing_definitions}."
                    )
    
                return entity_types_json
    
        def __verify_prebuilt_relationship_json(
                self,
                old_chunk_id_to_new_chunk_id: dict,
                old_entity_id_to_new_entity_id: dict[str, str],
                chunk_json: list[dict],
                entity_json: list[dict],
                relationship_json: list[dict]
            ) -> list[dict]:
                """
                TODO: ADD TOKEN COUNTING TO ENSURE IT DOESN'T GO OVER EMBEDDING WINDOW
                Expected Schema for each relationship in the list:
                [
                    {
                        "chunk_id": "Unique identifier for the associated chunk.",
                        "src_entity_id": "Unique identifier for the entity.",
                        "dst_entity_id": "Unique identifier for the related entity.",
                        "relation_description": "The description of the entity.",
                        "keywords": ["List of types associated with the entity."],
                    }
                ]
        
                The chunk ID and source and destination entity IDs are required.
                Optional metadata is defaulted and marked for later enrichment.
                """
                if not isinstance(old_chunk_id_to_new_chunk_id, dict):
                    raise TypeError(
                        "old_chunk_id_to_new_chunk_id must be a dictionary."
                    )
                if not isinstance(old_entity_id_to_new_entity_id, dict):
                    raise TypeError(
                        "old_entity_id_to_new_entity_id must be a dictionary."
                    )
                if not isinstance(chunk_json, list):
                    raise TypeError("Chunk JSON must be a list of chunk objects.")
                if not isinstance(entity_json, list):
                    raise TypeError("Entity JSON must be a list of entity objects.")
                if not isinstance(relationship_json, list):
                    raise TypeError(
                        "Relationship JSON must be a list of relationship objects."
                    )
                chunk_content_by_id = {
                    chunk["chunk_id"]: chunk["content"]
                    for chunk in chunk_json
                }
                entity_name_by_id = {
                    entity["entity_id"]: entity["entity_name"]
                    for entity in entity_json
                }
        
                prepared_relationships = []
        
                # Validate the complete input before modifying any relationships.
                for index, relationship in enumerate(relationship_json):
                    if not isinstance(relationship, dict):
                        raise TypeError(
                            f"Relationship at index {index} must be an object "
                            "(dictionary)."
                        )
        
                    missing_keys = [
                        key
                        for key in ("chunk_id", "src_entity_id", "dst_entity_id")
                        if key not in relationship
                    ]
                    if missing_keys:
                        raise ValueError(
                            f"Relationship at index {index} is missing required "
                            f"key(s): {', '.join(missing_keys)}."
                        )
        
                    old_chunk_id = relationship["chunk_id"]
                    try:
                        new_chunk_id = old_chunk_id_to_new_chunk_id[old_chunk_id]
                    except TypeError as exc:
                        raise TypeError(
                            f"'chunk_id' for relationship at index {index} must "
                            "be hashable."
                        ) from exc
                    except KeyError as exc:
                        raise ValueError(
                            f"Relationship at index {index} references unknown "
                            f"chunk_id '{old_chunk_id}'."
                        ) from exc
    
                    old_src_entity_id = relationship["src_entity_id"]
                    old_dst_entity_id = relationship["dst_entity_id"]
                    try:
                        new_src_entity_id = old_entity_id_to_new_entity_id[
                            old_src_entity_id
                        ]
                    except TypeError as exc:
                        raise TypeError(
                            f"'src_entity_id' for relationship at index {index} "
                            "must be hashable."
                        ) from exc
                    except KeyError as exc:
                        raise ValueError(
                            f"Relationship at index {index} references unknown "
                            f"src_entity_id '{old_src_entity_id}'."
                        ) from exc
        
                    try:
                        new_dst_entity_id = old_entity_id_to_new_entity_id[
                            old_dst_entity_id
                        ]
                    except TypeError as exc:
                        raise TypeError(
                            f"'dst_entity_id' for relationship at index {index} "
                            "must be hashable."
                        ) from exc
                    except KeyError as exc:
                        raise ValueError(
                            f"Relationship at index {index} references unknown "
                            f"dst_entity_id '{old_dst_entity_id}'."
                        ) from exc
    
                    for endpoint_label, endpoint_id, endpoint_name in (
                        (
                            "src_entity_id",
                            old_src_entity_id,
                            entity_name_by_id[new_src_entity_id],
                        ),
                        (
                            "dst_entity_id",
                            old_dst_entity_id,
                            entity_name_by_id[new_dst_entity_id],
                        ),
                    ):
                        name_matches_chunk, missing_name_tokens = (
                            self.__does_entity_name_overlap_chunk(
                                endpoint_name,
                                chunk_content_by_id[new_chunk_id],
                            )
                        )
                        if not name_matches_chunk:
                            raise ValueError(
                                f"Relationship at index {index} references "
                                f"{endpoint_label} '{endpoint_id}' with entity_name "
                                f"'{endpoint_name}', but that name is not present in "
                                f"chunk '{old_chunk_id}'. Missing normalized "
                                f"token(s): {missing_name_tokens}."
                            )
        
                    relation_description = relationship.get("relation_description", "")
                    if relation_description is None:
                        relation_description = ""
                    if not isinstance(relation_description, str):
                        raise TypeError(
                            f"'relation_description' for relationship at index "
                            f"{index} must be a string."
                        )
        
                    keywords = relationship.get("keywords", [])
                    if keywords is None:
                        keywords = []
                    if not isinstance(keywords, list):
                        raise TypeError(
                            f"'keywords' for relationship at index {index} must be a list."
                        )
                    if any(not isinstance(keyword, str) for keyword in keywords):
                        raise TypeError(
                            f"Every keyword for relationship at index {index} must "
                            "be a string."
                        )
        
                    relation_id = self.hashing_func.content_hash(
                        f"{new_src_entity_id}_{new_dst_entity_id}"
                    )
                    prepared_relationships.append(
                        (
                            relationship,
                            relation_id,
                            new_chunk_id,
                            new_src_entity_id,
                            new_dst_entity_id,
                            relation_description,
                            keywords,
                        )
                    )
        
                for (
                    relationship,
                    relation_id,
                    new_chunk_id,
                    new_src_entity_id,
                    new_dst_entity_id,
                    relation_description,
                    keywords,
                ) in prepared_relationships:
                    relationship["relation_id"] = relation_id
                    relationship["chunk_id"] = new_chunk_id
                    relationship["src_entity_id"] = new_src_entity_id
                    relationship["dst_entity_id"] = new_dst_entity_id
                    relationship["relation_description"] = relation_description
                    relationship["keywords"] = keywords
                    relationship["_is_relationship_enriched"] = bool(
                        relation_description.strip()
                    )
                    relationship["does_relationship_have_keywords"] = bool(keywords)
                return relationship_json

        async def ingest_and_embed_custom_entities(self, entity_json, entity_types=None, async_cooldown=60, requests_per_minute=1000) -> None:
               self._require_open_connection()
               try:
                   await self.add_entity_types(entity_types)
                   registered_entity_types = await self.get_entity_types()
                   entity_indexer = EntityIndexer(
                       conn=self.conn,
                       lock=self.lock,
                       language_model=self.language_model.generate,
                       embedding_model=self.embedding_model,
                       hash_func=self.hashing_func.content_hash,
                       entity_groups=registered_entity_types,
                       add_embedding_func=self.add_embedding,
                   )
                   await entity_indexer.ingest_custom_entities_from_ingested_chunks(
                       custom_entities_json=entity_json,
                       async_cooldown=async_cooldown,
                       requests_per_minute=requests_per_minute,
                       verbose=self.verbose,
                   )
                   await entity_indexer.fix_entities(
                       async_cooldown=async_cooldown,
                       requests_per_minute=requests_per_minute,
                       verbose=self.verbose,
                   )
                   await entity_indexer.embed_entities(
                       async_cooldown=async_cooldown,
                       requests_per_minute=requests_per_minute,
                       verbose=self.verbose,
                   )
               except Exception as e:
                   traceback.print_exc()
                   raise RuntimeError(
                       "Error ingesting and embedding custom entities: "
                       f"{e}"
                   ) from e
    
        async def ingest_and_embed_custom_relationships(self, relationship_json, async_cooldown=60, requests_per_minute=1000) -> None:
               self._require_open_connection()
               try:
                   relationship_indexer = RelationshipIndexer(
                       conn=self.conn,
                       lock=self.lock,
                       language_model=self.language_model.generate,
                       hash_func=self.hashing_func.content_hash,
                       embedding_model=self.embedding_model,
                       add_embedding_func=self.add_embedding,
                   )
                   await relationship_indexer.ingest_custom_relationships_from_ingested_chunks(
                       custom_relationships_json=relationship_json,
                       async_cooldown=async_cooldown,
                       requests_per_minute=requests_per_minute,
                       verbose=self.verbose,
                   )
                   await relationship_indexer.fix_relationships(
                       async_cooldown=async_cooldown,
                       requests_per_minute=requests_per_minute,
                       verbose=self.verbose,
                   )
                   new_relationships = await self.get_new_relationships()
                   updated_relationships = await self.get_updated_relationships()
                   await relationship_indexer.embed_relationships(
                       new_relationships=new_relationships,
                       updated_relationships=updated_relationships,
                       async_cooldown=async_cooldown,
                       requests_per_minute=requests_per_minute,
                       verbose=self.verbose,
                   )
               except Exception as e:
                   traceback.print_exc()
                   raise RuntimeError(
                       "Error ingesting and embedding custom relationships: "
                       f"{e}"
                   ) from e
        async def __verify_custom_records_persisted(
            self,
            entity_json: list[dict],
            relationship_json: list[dict],
        ) -> bool:
            """Verify every supplied custom record and chunk mapping was saved."""
            expected_entity_ids = {entity["entity_id"] for entity in entity_json}
            expected_relationship_ids = {
                relationship["relation_id"] for relationship in relationship_json
            }
            expected_entity_mappings = {
                (entity["chunk_id"], entity["entity_id"])
                for entity in entity_json
            }
            expected_relationship_mappings = {
                (relationship["chunk_id"], relationship["relation_id"])
                for relationship in relationship_json
            }
    
            async with self.lock:
                entity_cursor = await self.conn.execute(
                    "SELECT entity_id FROM entity"
                )
                saved_entity_ids = {row[0] for row in await entity_cursor.fetchall()}
                await entity_cursor.close()
    
                relationship_cursor = await self.conn.execute(
                    "SELECT relation_id FROM relationship"
                )
                saved_relationship_ids = {
                    row[0] for row in await relationship_cursor.fetchall()
                }
                await relationship_cursor.close()
    
                entity_mapping_cursor = await self.conn.execute(
                    "SELECT chunk_id, entity_id FROM chunk_to_entity_mapping"
                )
                saved_entity_mappings = {
                    tuple(row) for row in await entity_mapping_cursor.fetchall()
                }
                await entity_mapping_cursor.close()
    
                relationship_mapping_cursor = await self.conn.execute(
                    """
                    SELECT chunk_id, relation_id
                    FROM chunk_to_relationship_mapping
                    """
                )
                saved_relationship_mappings = {
                    tuple(row)
                    for row in await relationship_mapping_cursor.fetchall()
                }
                await relationship_mapping_cursor.close()
    
            saved_expected_entity_ids = expected_entity_ids & saved_entity_ids
            saved_expected_relationship_ids = (
                expected_relationship_ids & saved_relationship_ids
            )
            missing_entities = expected_entity_ids - saved_entity_ids
            missing_relationships = (
                expected_relationship_ids - saved_relationship_ids
            )
            if len(saved_expected_entity_ids) != len(expected_entity_ids):
                raise RuntimeError(
                    "Custom entity persistence count mismatch: expected "
                    f"{len(expected_entity_ids)} unique supplied entities, found "
                    f"{len(saved_expected_entity_ids)}; missing IDs: "
                    f"{sorted(missing_entities)}."
                )
            if len(saved_expected_relationship_ids) != len(
                expected_relationship_ids
            ):
                raise RuntimeError(
                    "Custom relationship persistence count mismatch: expected "
                    f"{len(expected_relationship_ids)} unique supplied relationships, "
                    f"found {len(saved_expected_relationship_ids)}; missing IDs: "
                    f"{sorted(missing_relationships)}."
                )
    
            missing_entity_mappings = (
                expected_entity_mappings - saved_entity_mappings
            )
            if missing_entity_mappings:
                raise RuntimeError(
                    "Custom entity chunk mappings were not persisted: "
                    f"{sorted(missing_entity_mappings)}."
                )
            missing_relationship_mappings = (
                expected_relationship_mappings - saved_relationship_mappings
            )
            if missing_relationship_mappings:
                raise RuntimeError(
                    "Custom relationship chunk mappings were not persisted: "
                    f"{sorted(missing_relationship_mappings)}."
                )
            return True
    
        async def construct_custom_GraphRAG(self, chunks: list[dict], entities: list[dict] = None, rels_json: list[dict] = None, entity_types_json: dict[str, str] = None, debug_mode=False, async_cooldown=60, requests_per_minute=1000) -> None:
                       # First, validate the provided information before we make any llm or embedding calls.
                       try: 
                           self._graph_construction_print("STEP ONE: COLLECTING SOURCE DOCUMENTS AND VERIFYING THEIR FORMAT")
                           source_name_to_id = self.__verify_source_json(chunks)
                       except Exception as e:
                           traceback.print_exc()
                           raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the source JSON: {e}")
                       try: 
                            self._graph_construction_print("STEP TWO: VERIFYING THE FORMAT OF PROVIDED CHUNKS")
                            old_chunk_ids_to_new_chunk_ids, chunk_json_list = self.__verify_prechunked_corpus(source_name_to_id, chunks)
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the chunk JSON: {e}")
                       try: 
                            self._graph_construction_print("STEP THREE: VALIDATING PROVIDED ENTITY JSON FORMAT")
                            local_entity_ids_to_new_entity_ids, entity_json = self.__verify_prebuilt_entity_json(
                                old_chunk_ids_to_new_chunk_ids,
                                chunk_json_list,
                                entities,
                            )
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the entity JSON: {e}")
                       try:
                            self._graph_construction_print("STEP FOUR: VALIDATING PROVIDED ENTITY TYPE DEFINITIONS")
                            entity_types_json = self.__verify_entity_types(
                                entity_json,
                                entity_types_json,
                            )
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying entity type definitions: {e}")
                       try:
                            self._graph_construction_print("STEP FIVE: VALIDATING PROVIDED RELATIONSHIP JSON FORMAT")
                            relationship_json = self.__verify_prebuilt_relationship_json(
                                old_chunk_ids_to_new_chunk_ids,
                                local_entity_ids_to_new_entity_ids,
                                chunk_json_list,
                                entity_json,
                                rels_json,
                            )
                       except Exception as e:
                              traceback.print_exc()
                              raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the relationship JSON: {e}")
                       # Second, we ingest the chunks into the database.
                       try:
                            total_chunks = []
                            self._graph_construction_print("STEP SIX: INGESTING AND EMBEDDING THE CHUNKS")
                            for chunk in tqdm_asyncio(chunk_json_list, desc=f"Adding chunks from custom chunks into the DB"):
                                await self.add_chunk(
                                    content=chunk["content"],
                                    source_name=chunk["source_name"],
                                    model_name=self.embedding_model.model_name
                                )
                                total_chunks.append(chunk)
    
                            failed_chunks = await self.get_unprocessed_metadata("chunk")
                            while failed_chunks:
                                current_batch = failed_chunks[:requests_per_minute]
                                remaining_chunks = failed_chunks[requests_per_minute:]
    
                                async def embed_custom_chunk(chunk):
                                    try:
                                        await self.add_embedding(
                                            content=chunk["content"],
                                            table="chunk",
                                            pk_hash=chunk["chunk_id"],
                                        )
                                        return None
                                    except Exception as error:
                                        if self.verbose:
                                            print(
                                                "Custom chunk embedding failed; "
                                                f"retrying: {error}"
                                            )
                                        return chunk
    
                                batch_results = await tqdm_asyncio.gather(
                                    *(
                                        embed_custom_chunk(chunk)
                                        for chunk in current_batch
                                    ),
                                    desc="Embedding custom chunks",
                                    total=len(current_batch),
                                )
                                failed_chunks = [
                                    chunk
                                    for chunk in batch_results
                                    if chunk is not None
                                ]
                                failed_chunks.extend(remaining_chunks)
                                if failed_chunks:
                                    await asyncio.sleep(async_cooldown)
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while adding chunks to the database: {e}")
    
                       # Third, construct the graph from the provided entity and relationship json
                       try:
                            self._graph_construction_print("STEP SEVEN: INGESTING AND EMBEDDING CUSTOM ENTITIES")
                            await self.ingest_and_embed_custom_entities(
                                entity_json=entity_json,
                                entity_types=entity_types_json or None,
                                async_cooldown=async_cooldown,
                                requests_per_minute=requests_per_minute,
                            )
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(
                                "Error in construct_custom_GraphRAG while ingesting "
                                f"and embedding custom entities: {e}"
                            ) from e
    
                       try:
                            self._graph_construction_print("STEP EIGHT: INGESTING AND EMBEDDING CUSTOM RELATIONSHIPS")
                            await self.ingest_and_embed_custom_relationships(
                                relationship_json=relationship_json,
                                async_cooldown=async_cooldown,
                                requests_per_minute=requests_per_minute,
                            )
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(
                                "Error in construct_custom_GraphRAG while ingesting "
                                f"and embedding custom relationships: {e}"
                            ) from e
    
                       try:
                            self._graph_construction_print(
                                "STEP NINE: VERIFYING CUSTOM GRAPH PERSISTENCE"
                            )
                            await self.__verify_custom_records_persisted(
                                entity_json,
                                relationship_json,
                            )
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(
                                "Error in construct_custom_GraphRAG while verifying "
                                f"persisted custom records: {e}"
                            ) from e
    
                       # Fourth, generate and save a global description for each source document.
                       try:
                          self._graph_construction_print("STEP TEN: GENERATING A GLOBAL GRAPH DESCRIPTION")
                          await self.generate_global_descriptions(debug_mode=debug_mode)
                       except Exception as e:
                          traceback.print_exc()
                          raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while generating global descriptions: {e}")
                        
                       # Build the graph reachability sidecars from the persisted data.
                       try:
                         self._graph_construction_print("STEP ELEVEN: CONSTRUCTING THE REACHABILITY MAP")
                         await self.construct_reachability_map()
                       except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(
                                "Error in construct_custom_GraphRAG while "
                                f"constructing the reachability map: {e}"
                            ) from e
                       # Lastly, we construct the BM25 index
                       try:
                          self._graph_construction_print("STEP TWELVE: CREATING THE BM25 INDEXES")
                          await self.create_BM25_index_for_table("chunk")
                          await self.create_BM25_index_for_table("entity")
                          await self.create_BM25_index_for_table("relationship")  
                       except Exception as e:
                          traceback.print_exc()
                          raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while creating the BM25 index: {e}")
    
        def __prepare_chunk_free_graph(
            self,
            entity_json: list[dict],
            relationship_json: list[dict],
            entity_types_json: dict[str, str] = None,
        ) -> tuple[dict, list[dict], list[dict], dict[str, str]]:
            """Create the placeholder chunk and validate structured records."""
            if not isinstance(entity_json, list):
                raise TypeError("entity_json must be a list of entity objects.")
            if any(not isinstance(entity, dict) for entity in entity_json):
                raise TypeError("Every entity must be an object (dictionary).")
            if not isinstance(relationship_json, list):
                raise TypeError(
                    "relationship_json must be a list of relationship objects."
                )
            if any(
                not isinstance(relationship, dict)
                for relationship in relationship_json
            ):
                raise TypeError(
                    "Every relationship must be an object (dictionary)."
                )

            # Generated IDs and the internal chunk ID must not modify the
            # caller's structured input.
            prepared_entities = [dict(entity) for entity in entity_json]
            prepared_relationships = [
                dict(relationship) for relationship in relationship_json
            ]

            placeholder_chunk_id = "__structured_graph_placeholder__"
            placeholder_source_name = "__structured_graph__"

            for entity in prepared_entities:
                entity["chunk_id"] = placeholder_chunk_id
            for relationship in prepared_relationships:
                relationship["chunk_id"] = placeholder_chunk_id

            # Entity names satisfy the normal chunk-membership validation.
            # Descriptions provide fallback context if metadata enrichment is
            # needed, but this placeholder is never embedded or retrieved.
            placeholder_lines = [
                "Internal structured-graph placeholder chunk."
            ]
            for entity in prepared_entities:
                entity_name = entity.get("entity_name")
                if isinstance(entity_name, str) and entity_name.strip():
                    placeholder_lines.append(entity_name.strip())
                entity_description = entity.get("entity_description")
                if (
                    isinstance(entity_description, str)
                    and entity_description.strip()
                ):
                    placeholder_lines.append(entity_description.strip())
            for relationship in prepared_relationships:
                relation_description = relationship.get(
                    "relation_description"
                )
                if (
                    isinstance(relation_description, str)
                    and relation_description.strip()
                ):
                    placeholder_lines.append(relation_description.strip())

            placeholder_chunks = [
                {
                    "chunk_id": placeholder_chunk_id,
                    "source_name": placeholder_source_name,
                    "content": "\n".join(placeholder_lines),
                }
            ]

            source_name_to_id = self.__verify_source_json(
                placeholder_chunks
            )
            (
                old_chunk_ids_to_new_chunk_ids,
                placeholder_chunks,
            ) = self.__verify_prechunked_corpus(
                source_name_to_id,
                placeholder_chunks,
            )
            (
                local_entity_ids_to_new_entity_ids,
                prepared_entities,
            ) = self.__verify_prebuilt_entity_json(
                old_chunk_ids_to_new_chunk_ids,
                placeholder_chunks,
                prepared_entities,
            )
            entity_types_json = self.__verify_entity_types(
                prepared_entities,
                entity_types_json,
            )
            prepared_relationships = self.__verify_prebuilt_relationship_json(
                old_chunk_ids_to_new_chunk_ids,
                local_entity_ids_to_new_entity_ids,
                placeholder_chunks,
                prepared_entities,
                prepared_relationships,
            )

            return (
                placeholder_chunks[0],
                prepared_entities,
                prepared_relationships,
                entity_types_json,
            )

        async def __save_placeholder_chunk(
            self,
            placeholder_chunk: dict,
        ) -> None:
            """Persist the required placeholder without creating a vector."""
            await self.add_chunk(
                content=placeholder_chunk["content"],
                source_name=placeholder_chunk["source_name"],
                model_name=self.embedding_model.model_name,
                tokens=0,
            )

            # Custom entity ingestion selects chunks marked as embedded. The
            # placeholder needs that state but intentionally has no vector row.
            async with self.lock:
                await self.conn.execute(
                    """
                    UPDATE chunk
                    SET is_embedded = 1
                    WHERE chunk_id = ?
                    """,
                    (placeholder_chunk["chunk_id"],),
                )
                await self.conn.commit()

        async def construct_custom_Graph_no_chunks(
            self,
            entity_json: list[dict],
            relationship_json: list[dict],
            entity_types_json: dict[str, str] = None,
            debug_mode: bool = False,
            async_cooldown: int = 60,
            requests_per_minute: int = 1000,
        ) -> None:
            """Construct a structured graph without source-document chunks.

            The database schema requires entities and relationships to be
            associated with a chunk. This method therefore creates one internal
            placeholder chunk for the supplied graph. The placeholder is marked
            as embedded without generating or storing a vector because it exists
            only to satisfy the database mappings; retrieval remains focused on
            the real entities and relationships.
            """
            self._require_open_connection()

            try:
                self._graph_construction_print(
                    "STEP ONE: PREPARING AND VALIDATING THE STRUCTURED GRAPH"
                )
                (
                    placeholder_chunk,
                    prepared_entities,
                    prepared_relationships,
                    entity_types_json,
                ) = self.__prepare_chunk_free_graph(
                    entity_json,
                    relationship_json,
                    entity_types_json,
                )
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error validating the chunk-free structured graph: "
                    f"{error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP TWO: CREATING THE INTERNAL PLACEHOLDER CHUNK"
                )
                await self.__save_placeholder_chunk(placeholder_chunk)
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error creating the internal placeholder chunk: "
                    f"{error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP THREE: INGESTING AND EMBEDDING CUSTOM ENTITIES"
                )
                await self.ingest_and_embed_custom_entities(
                    entity_json=prepared_entities,
                    entity_types=entity_types_json or None,
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                )
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error ingesting entities for the chunk-free structured "
                    f"graph: {error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP FOUR: INGESTING AND EMBEDDING CUSTOM RELATIONSHIPS"
                )
                await self.ingest_and_embed_custom_relationships(
                    relationship_json=prepared_relationships,
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                )
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error ingesting relationships for the chunk-free "
                    f"structured graph: {error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP FIVE: VERIFYING CUSTOM GRAPH PERSISTENCE"
                )
                await self.__verify_custom_records_persisted(
                    prepared_entities,
                    prepared_relationships,
                )
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error verifying the chunk-free structured graph: "
                    f"{error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP SIX: GENERATING A GLOBAL GRAPH DESCRIPTION"
                )
                await self.generate_global_descriptions(
                    debug_mode=debug_mode
                )
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error generating the global graph description: "
                    f"{error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP SEVEN: CONSTRUCTING THE REACHABILITY MAP"
                )
                await self.construct_reachability_map()
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error constructing the reachability map: "
                    f"{error}"
                ) from error

            try:
                self._graph_construction_print(
                    "STEP EIGHT: CREATING ENTITY AND RELATIONSHIP BM25 INDEXES"
                )
                await self.create_BM25_index_for_table("entity")
                await self.create_BM25_index_for_table("relationship")
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(
                    "Error creating BM25 indexes for the chunk-free "
                    f"structured graph: {error}"
                ) from error
