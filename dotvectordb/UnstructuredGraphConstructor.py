"""
Author: Larnell Moore
Purpose: This module provides the Unstructured GraphConstructor class, which extends the DotVectorDB class to include advanced graph construction capabilities. It allows users to ingest documents, extract and embed entities and relationships, generate global descriptions, and construct a graph representation of the database. The class also supports creating a BM25 index for efficient text retrieval.
"""
from .DotVDB import DotVectorDB
import json
from itertools import chain
import shutil
from pyroaring import BitMap
import traceback
import asyncio
from dotvectordb.interfaces.EmbeddingModel import EmbeddingModel
from dotvectordb.interfaces.LanguageModel import LanguageModel
from dotvectordb.interfaces.ChunkingFunc import ChunkingFunc
from dotvectordb.interfaces.HashingFunc import HashingFunc
from dotvectordb.plugins.embedding_providers.openai_embeddings import OpenAIEmbeddingModel
from dotvectordb.data_structures.bitmap.RoaringBitMapConstruction import (
    ExactHopBitmapBuilder,
    LmdbStore,
)
from dotvectordb.plugins.llm_providers.openai_llms import OpenAiAgent
from dotvectordb.plugins.chunk_functions.FixedSizeChunker import FixedSizeChunker
from dotvectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc
from dotvectordb.plugins.loaders.TextConverter import DocumentTextExtractor
from dotvectordb.construction_prompts.entity_prompts import DEFAULT_ENTITY_TYPES, isEntityTypesFormatCorrect
from dotvectordb.graph.entity_indexer import EntityIndexer
from dotvectordb.graph.relationship_indexer import RelationshipIndexer
from dotvectordb.graph.generate_global_desc import GlobalDescriptionGenerator
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
import asyncio
import traceback
import numpy as np
class UnstructuredGraphConstructor(DotVectorDB):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        graph_path = Path(self.path).resolve().with_suffix("")
        self.reachability_output_dir = Path(f"{graph_path}__graph.pkl.memory_map")
        self.current_bitmap_path = Path(f"{graph_path}__graph.pkl.memory_map.current.json")
        previous = json.loads(self.current_bitmap_path.read_text()) if self.current_bitmap_path.exists() else {}
        self.previous_max_hop = (
            previous.get("max_hop", 1)
            if previous.get("path") == str(self.reachability_output_dir) else 1
        )
    async def get_unprocessed_metadata(self, table) -> list[dict]:
            """Returns a list of dictionaries containing metadata for all rows in the specified table where the is_embedded flag is 0 (i.e., unprocessed). Each dictionary contains the column names as keys and the corresponding row values as values. This function is useful for identifying which chunks, entities, or relationships still need to be embedded."""
            self._require_open_connection()
            try:
                EMBEDDING_PROGRESS_SQL = f"""SELECT * FROM {table} WHERE is_embedded=0"""
                async with self.conn.execute(
                    EMBEDDING_PROGRESS_SQL
                ) as cursor:
                    rows = await cursor.fetchall()
                    column_names = [description[0] for description in cursor.description]
                return [dict(zip(column_names, row)) for row in rows]
            except Exception as e:  
                traceback.print_exc()
                raise RuntimeError(f"Error in the get_unprocessed_metadata function inside DotVectorDB: {e}")
    async def create_BM25_index_for_table(self, table_name: str):
                """Creates a BM25 index for the specified table ('chunk', 'entity', or 'relationship') using SQLite's FTS5 extension."""
                try:
                    self._require_open_connection()
                    table_config = {
                        "chunk": {
                            "id_column": "chunk_id",
                            "columns": ["content"],
                        },
                        "entity": {
                            "id_column": "entity_id",
                            "columns": [
                                "entity_name",
                                "entity_description",
                            ],
                        },
                        "relationship": {
                            "id_column": "relation_id",
                            "columns": [
                                "relation_description",
                            ],
                        },
                    }
                    if table_name not in table_config:
                        raise ValueError(
                            "table_name must be 'chunk', 'entity', or 'relationship'"
                        )
                    config = table_config[table_name]
                    progress_flag = {
                        "entity": "entity_bm25_in_progress",
                        "relationship": "relationship_bm25_in_progress",
                    }.get(table_name)
                    if progress_flag:
                        await self.conn.execute(
                            f"UPDATE global_config SET {progress_flag} = TRUE WHERE id = 1"
                        )
                        await self.conn.commit()
                        self.global_config[progress_flag] = True
                    index_name = f"{table_name}_fts"
                    columns = ", ".join(config["columns"])
                    id_column = config["id_column"]
                    old_values = ", ".join(f"OLD.{column}" for column in config["columns"])
                    new_values = ", ".join(f"NEW.{column}" for column in config["columns"])
                    changed_text = " OR ".join(
                        f"OLD.{column} IS NOT NEW.{column}" for column in config["columns"]
                    )
                    async with self.lock:
                        await self.conn.execute(
                            f"""
                            CREATE VIRTUAL TABLE IF NOT EXISTS {index_name}
                            USING fts5(
                                {columns},
                                content='{table_name}',
                                content_rowid='{id_column}'
                            )
                            """
                        )
                        await self.conn.execute(
                            f"""
                            CREATE TRIGGER IF NOT EXISTS insert_{index_name}
                            AFTER INSERT ON {table_name}
                            BEGIN
                                INSERT INTO {index_name}(rowid, {columns})
                                VALUES (NEW.{id_column}, {new_values});
                            END;
                            """
                        )
                        await self.conn.execute(
                            f"""
                            CREATE TRIGGER IF NOT EXISTS delete_{index_name}
                            AFTER DELETE ON {table_name}
                            BEGIN
                                INSERT INTO {index_name}({index_name}, rowid, {columns})
                                VALUES ('delete', OLD.{id_column}, {old_values});
                            END;
                            """
                        )
                        await self.conn.execute(
                            f"""
                            CREATE TRIGGER IF NOT EXISTS update_{index_name}
                            AFTER UPDATE OF {columns}, {id_column} ON {table_name}
                            WHEN {changed_text} OR OLD.{id_column} IS NOT NEW.{id_column}
                            BEGIN
                                INSERT INTO {index_name}({index_name}, rowid, {columns})
                                VALUES ('delete', OLD.{id_column}, {old_values});
                                INSERT INTO {index_name}(rowid, {columns})
                                VALUES (NEW.{id_column}, {new_values});
                            END;
                            """
                        )
                        # Populate the index using records already in the table.
                        await self.conn.execute(
                            f"""
                            INSERT INTO {index_name}({index_name})
                            VALUES ('rebuild')
                            """
                        )
                        if progress_flag:
                            await self.conn.execute(
                                f"UPDATE global_config SET {progress_flag} = FALSE WHERE id = 1"
                            )
                        await self.conn.commit()
                        if progress_flag:
                            self.global_config[progress_flag] = False
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(
                        "Error in the create_BM25_index_for_table "
                        f"function inside DotVectorDB: {e}"
                    )
    async def ingest_documents(
                self,
                input_document_list=None,
                async_cooldown=60,
                requests_per_minute=1000,
                skip_embeddings=False,
                skip_bm25=False,
            ):
                """
                :Docstring for ingest_documents
                :input_document_list is a list of paths to the documents to be ingested into the database.
                :async_cooldown is the cooldown time in seconds between batches of requests to the embedding model to avoid rate limiting.
                :requests_per_minute is the maximum number of requests to the embedding model per minute to avoid rate limiting.
                :skip_embeddings skips generating chunk embeddings when True.
                :skip_bm25 skips building the chunk BM25 index when True.
                :verbose is a boolean flag that enables verbose logging for debugging purposes.
                :This function extracts text from the documents, chunks the text, and adds the chunks and their embeddings to the database.

                Ingest a corpus of documents into the database. This function extracts text 
                from the documents, chunks the text, and adds the chunks and their 
                embeddings to the database.
                """ 
                self._require_open_connection()
                # First, input document can either be None or a list of strings. If neither, we raise a value error.
                # If the user provided valid documents, we extract all text from them.
                try:
                    extracted_text = None
                    if input_document_list is not None:
                        if not isinstance(input_document_list, list) or not all(isinstance(item, str) for item in input_document_list):
                            raise ValueError("input_document_list must be a list of strings or None.")
                        else:
                            extracted_text = DocumentTextExtractor(directories=input_document_list)
                            results = extracted_text.controller()
                            if results == {}:
                                raise ValueError("No text could be extracted from the provided documents.")
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for extracting text: {e}")       
                # If we have extracted text, we need to chunk them and put them into the database.
                try:
                    if extracted_text:
                        total_chunks = []
                        for document, value in results.items():
                            print(f" Processing document: {document}")
                            document_chunks = self.chunking_func.chunk(value)
                            source_id = await self.add_metadata_content(table="source", name=document)
                            async with self.lock:
                                await self.conn.execute(
                                    "UPDATE source SET num_of_chunks_found = ?, is_chunk_number_known = TRUE WHERE id = ?",
                                    (len(document_chunks), source_id),
                                )
                                await self.conn.commit()
                            print(f"  Number of chunks created for {document}: {len(document_chunks)}")
                            async with self.conn.execute(
                                "SELECT num_of_chunks_added FROM source WHERE id = ?",
                                (source_id,),
                            ) as cursor:
                                num_of_chunks_added = (await cursor.fetchone())[0]
                            for chunk in tqdm_asyncio(
                                document_chunks[num_of_chunks_added:],
                                initial=num_of_chunks_added,
                                total=len(document_chunks),
                                desc=f"Adding chunks from {document} into the DB",
                            ):
                                await self.add_chunk(
                                    content=chunk["content"],
                                    source_name=document,
                                    model_name=self.embedding_model.model_name,
                                    tokens=chunk["tokens"],
                                    numbered_line_dict=chunk["numbered_lines"],
                                )
                            total_chunks.extend(document_chunks)
                        if self.verbose:
                            print(f"Total chunks created: {len(total_chunks)}")
                            print(f"Starting to add embeddings for each chunk to the database...")
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for chunking and adding chunks: {e}")
                # By this point, all chunks have been added to the database. Now,  we add the embeddings for each chunk to the database. 
                # We begin by grabbing all chunks where is_embedded flag is 0. This allows us to resume from any chunks that were never embedded yet.
                # We use async to send concurrent requests to the embedding model, but we also implement a cooldown to avoid rate limiting.
                if not skip_embeddings:
                    try:
                        failed_chunks = await self.get_unprocessed_metadata("chunk")
                        while failed_chunks:
                            current_batch = failed_chunks[:requests_per_minute]
                            remaining_chunks = failed_chunks[requests_per_minute:]
                            failed_chunks = []
                            async def embed_chunk(chunk):
                                try:
                                    content = chunk["content"]
                                    pk_hash = chunk["chunk_id"]
                                    await self.add_embedding(
                                        content=content,
                                        table="chunk",
                                        pk_hash=pk_hash
                                    )
                                    return None
                                except Exception as error:
                                    if self.verbose:
                                        print(f"Embedding failed, retrying next batch: {error}")
                                    return chunk         
                            if self.verbose:
                                print(f"Sending batch of {len(current_batch)} chunks to embedding model...")
                            batch_results = await tqdm_asyncio.gather(
                                    *(embed_chunk(chunk) for chunk in current_batch),
                                    desc="Embedding chunks and adding to the vector database",
                                )
                            failed_chunks.extend([
                                chunk for chunk in batch_results
                                if chunk is not None
                            ])
                            failed_chunks.extend(remaining_chunks)
                            if failed_chunks:
                                if self.verbose:
                                    print(
                                        f"Cooldown for {async_cooldown} seconds. "
                                        f"{len(failed_chunks)} chunks remaining."
                                    )
                                await asyncio.sleep(async_cooldown)
                        if self.verbose:
                            print("All embeddings added successfully.")
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for embedding chunks: {e}")
                if not skip_bm25:
                    await self.create_BM25_index_for_table("chunk")

    async def extract_and_embed_entities(self, entity_types=None, async_cooldown=60, requests_per_minute=1000,
                                         entity_extract_prompt=None, entity_input_example=None, entity_output_example=None) -> None:
                """Extracts, classifies, and embeds entities from the ingested chunks in the database. If entity_types is provided, it will add those entity types to the database before processing."""
                self._require_open_connection()
                status = await self.is_Database_new_or_old()
                try:
                    if status == True:
                        print("Database is new, adding default entity types...")
                        await self.add_entity_types(entity_types)
                    entity_types = await self.get_entity_types()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the extract_and_embed_entities3 function inside DotVectorDB while adding entity types: {e}")
                try:
                    entity_indexer = EntityIndexer(
                        conn=self.conn,
                        lock=self.lock,
                        language_model=self.language_model.generate,
                        embedding_model=self.embedding_model,
                        hash_func=self.hashing_func.content_hash,
                        entity_groups=entity_types,
                        add_embedding_func=self.add_embedding,
                        entity_extract_prompt=entity_extract_prompt,
                        entity_input_example=entity_input_example,
                        entity_output_example=entity_output_example,
                    )
                    await entity_indexer.extract_entities_from_ingested_chunks(
                        async_cooldown=async_cooldown,
                        requests_per_minute=requests_per_minute,
                        verbose=self.verbose
                    )
                    
                    await entity_indexer.fix_entities(
                        async_cooldown=async_cooldown,
                        requests_per_minute=requests_per_minute,
                        verbose=self.verbose         
                    )

                    new_entities = await self.get_new_entities_before_last_completed_version()          
                    updated_entities = await self.get_entities_that_need_updated_dedupe()
                    await entity_indexer.embed_entities(
                        new_entities=new_entities,
                        updated_entities=updated_entities,
                        async_cooldown=async_cooldown,
                        requests_per_minute=requests_per_minute,
                        verbose=self.verbose
                    )
                    # Note: This doesn't actually deduplicate entities at all
                    # It does something much more sophiscated, which is to find the nodes most similar to each other
                    # There's no point to potentially destroying the graph by merging nodes.
                    # So, the thought process is that at retrieval, just treat similar nodes as the same node
                    # If node a have 3 edges and nnode b has 5, treat it as if node a has 8 edges. It absorbs all edges from b, and b becomes an alias.
                    # tldr; dont deduplicate, just cluster to keep graph integrity.
                    entites_to_deduplicate = list(dict.fromkeys(
                                                entity_id for entity_id, _ in new_entities + updated_entities
                    ))
                    await entity_indexer.deduplicate_entities(
                        entities=entites_to_deduplicate,
                        async_cooldown=async_cooldown,
                        requests_per_minute=requests_per_minute,
                        verbose=self.verbose,
                    )      
                    return new_entities, updated_entities     
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the extract_and_embed_entities4 function inside DotVectorDB: {e}")
    async def extract_and_embed_relationships(self, async_cooldown=60, requests_per_minute=1000,
                                             relationship_extract_prompt=None, relationship_input_example=None,
                                             relationship_output_example=None, relationship_entity_example=None) -> None:
                    """Extracts relationships from the ingested chunks in the database and embeds them. This function uses the RelationshipIndexer class to perform the extraction and embedding of relationships."""
                    self._require_open_connection()
                    try:
                        relationship_indexer = RelationshipIndexer(
                            conn=self.conn,
                            lock=self.lock,
                            language_model=self.language_model.generate,
                            hash_func=self.hashing_func.content_hash,
                            embedding_model=self.embedding_model,
                            add_embedding_func=self.add_embedding,
                            relationship_extract_prompt=relationship_extract_prompt,
                            relationship_input_example=relationship_input_example,
                            relationship_output_example=relationship_output_example,
                            relationship_entity_example=relationship_entity_example,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the extract_and_embed_relationships function inside DotVectorDB while initializing RelationshipIndexer: {e}")
                    try:
                        await relationship_indexer.extract_relations_from_chunks(
                            async_cooldown=async_cooldown,
                            requests_per_minute=requests_per_minute,
                            verbose=self.verbose
                        )
                        await relationship_indexer.fix_relationships(
                            async_cooldown=async_cooldown,
                            requests_per_minute=requests_per_minute,
                            verbose=self.verbose
                        )
                        new_relationships = await self.get_new_relationships()
                        updated_relationships = await self.get_updated_relationships()
                        await relationship_indexer.embed_relationships(
                            new_relationships=new_relationships,
                            updated_relationships=updated_relationships,
                            async_cooldown=async_cooldown,
                            requests_per_minute=requests_per_minute,
                            verbose=self.verbose)
                        return new_relationships, updated_relationships
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the extract_and_embed_relationships function inside DotVectorDB while extracting and embedding relationships: {e}")
    async def generate_global_descriptions(self, sample_size=5, debug_mode=False, global_desc=True) -> list[dict]:
            """Generates a global description for each source document in the database. This function uses the GlobalDescriptionGenerator class to create a summary of the entity-relationship graph for each source document."""
            self._require_open_connection()
            try:
                self.global_config = await self.get_global_config()
                if not global_desc:
                    description = "No global description"
                    version = self.global_config["current_version_count"]
                    await self.update_global_description(description, version)
                    self.global_config["description"] = json.dumps(description)
                    self.global_config["description_version"] = version
                    return description
                entity_types = await self.get_entity_types(only_populated=True)
                sources = await self.get_sources()
                generator = GlobalDescriptionGenerator(
                    conn=self.conn,
                    lock=self.lock,
                    language_model=self.language_model.generate,
                    entity_types=entity_types,
                    sources=sources,
                    global_config=self.global_config,
                    save_global_description_func=self.update_global_description,
                    verbose=self.verbose,
                    debug_mode=debug_mode
                )
                return await generator.generate(
                    sample_size=sample_size,
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(
                    "Error in the generate_global_descriptions function inside DotVectorDB: "
                    f"{e}"
                ) 
        
    async def save_phase_checkpoint(self, checkpoints, phase=None):
        if phase is not None:
            checkpoints[phase] = True
        await self.conn.execute(
            "UPDATE global_config SET bitmap_phase_checkpoints = ? WHERE id = 1",
            (json.dumps(checkpoints),),
        )
        await self.conn.commit()
        self.global_config["bitmap_phase_checkpoints"] = checkpoints.copy()

    async def construct_reachability_map(
        self,
        new_entities,
        new_relationships,
        updated_entities,
        updated_relationships,
        max_hop: int = 3,
        batch_size: int = 10_000,
        workers: int = 4,
        high_degree_limit: int = 100,
    ) -> dict:
        """Update the Roaring index in four phases; max_hop=1 skips extra hops."""
        self._require_open_connection()
        self._require_size_positive(batch_size, "batch_size")
        self._require_size_positive(max_hop, "max_hop")

        builder = ExactHopBitmapBuilder(self.reachability_output_dir, max_hop=max_hop, workers=workers)
        indexed_version = int(self.global_config["current_version_count"])

        checkpoints = {
            "add_rows_complete": False,
            "add_entity_and_types_complete": False,
            **{f"add_{hop}_hop_complete": False for hop in range(1, max_hop + 1)},
        }

        checkpoints.update(self.global_config.get("bitmap_phase_checkpoints", {}))

        await self.save_phase_checkpoint(checkpoints)

        new_entity_ids = list(dict.fromkeys(entity_id for entity_id, _ in new_entities))
        entity_types = {}
        for entity_id, entity_type in new_entities + updated_entities:
            entity_types.setdefault(entity_id, set())
            if entity_type is not None:
                entity_types[entity_id].add(entity_type)
        # Keep only this run's affected nodes and added connections in memory.
        neighbors_by_entity = {entity_id: set() for entity_id in new_entity_ids}
        for entity_id, _ in updated_entities:
            neighbors_by_entity.setdefault(entity_id, set())
        for relationship in new_relationships + updated_relationships:
            source = relationship["src_entity_id"]
            destination = relationship["dst_entity_id"]
            neighbors_by_entity.setdefault(source, set()).add(destination)
            neighbors_by_entity.setdefault(destination, set()).add(source)
        affected_entity_ids = list(neighbors_by_entity)

        # Phase 1: Reserve rows only for new IDs not already assigned on disk.
        # Unassigned reservations from an interrupted call are reused on retry.
        self._graph_construction_print("REACHABILITY PHASE ONE: Adding entity rows")
        with tqdm(total=len(new_entity_ids), desc="Phase 1: Entity rows",
                  unit="entity", disable=not self.verbose) as progress:
            missing_new_ids = set(new_entity_ids)
            assigned_rows = 0
            if builder.entity_mapping_path.exists():
                with LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
                    assigned_rows = int(mapping.get(b"next_row", b"0"))
                    missing_new_ids = {entity_id for entity_id in new_entity_ids
                                       if f"id:{entity_id}".encode() not in mapping}
            unused_rows = builder.last_entity_row + 1 - assigned_rows
            rows_added = max(0, len(missing_new_ids) - unused_rows)
            if not checkpoints["add_rows_complete"]:
                builder.add_new_entity_ids(rows_added)
            progress.update(len(new_entity_ids))

        await self.save_phase_checkpoint(checkpoints, "add_rows_complete")

        # Phase 2: Use the supplied types and fetch affected degrees in batches.
        self._graph_construction_print("REACHABILITY PHASE TWO: Adding entities and types")
        degrees = await self.batch_get_entity_degrees(set(affected_entity_ids), batch_size=batch_size)
        if not checkpoints["add_entity_and_types_complete"]:
            with tqdm(total=len(affected_entity_ids), desc="Phase 2: Entities and types",
                      unit="entity", disable=not self.verbose) as progress:
                if not affected_entity_ids:
                    builder.add_entity_types([])
                for start in range(0, len(affected_entity_ids), batch_size):
                    batch = affected_entity_ids[start:start + batch_size]
                    builder.add_entity_types(
                        (entity_id, entity_types[entity_id])
                        for entity_id in batch if entity_id in entity_types
                    )
                    # New neighbors need degrees before one-hop batches begin.
                    # Existing degree changes are applied by the builder so estimates
                    # of incoming anchors also receive their degree deltas.
                    with LmdbStore(str(builder.entity_mapping_path), "w") as mapping:
                        for entity_id in batch:
                            row = int(mapping[f"id:{entity_id}".encode()])
                            degree_key = f"degree:{row}".encode()
                            if entity_id in missing_new_ids or degree_key not in mapping:
                                mapping[degree_key] = str(degrees[entity_id]).encode()
                    progress.update(len(batch))

        await self.save_phase_checkpoint(checkpoints, "add_entity_and_types_complete")

        # Phase 3: Add both directions of new and updated relationships; retain old edges.
        # Updated and isolated entities also receive their current degree.
        self._graph_construction_print("REACHABILITY PHASE THREE: Adding one-hop connections")
        one_hop_additions = 0
        if not checkpoints["add_1_hop_complete"]:
            with tqdm(total=len(affected_entity_ids), desc="Phase 3: One-hop connections",
                      unit="entity", disable=not self.verbose) as progress:
                one_hop_additions = builder.add_new_1_hop_relationships(
                    ((entity_id, neighbors_by_entity[entity_id], degrees[entity_id])
                     for entity_id in affected_entity_ids),
                    batch_size=batch_size,
                    relationships=chain(new_relationships, updated_relationships),
                )
                builder.save_entity_stats(
                    self.global_config["avg_degree"],
                    self.global_config["max_degree"],
                    self.global_config["min_degree"],
                )
                progress.update(len(affected_entity_ids))

        await self.save_phase_checkpoint(checkpoints, "add_1_hop_complete")

        # Phase 4: Save completion separately for each requested later hop.
        later_hop_rows = {}
        if max_hop > 1:
            self._graph_construction_print("REACHABILITY PHASE FOUR: Updating extra hops")
            with LmdbStore(str(builder.entity_mapping_path), "r") as mapping:
                pending_rows = BitMap.deserialize(mapping.get(b"later_hop_dirty", BitMap().serialize()))
            with tqdm(total=max_hop - 1, desc="Phase 4: Extra hops", unit="hop",
                      disable=not self.verbose) as progress:
                for hop in range(2, max_hop + 1):
                    if checkpoints[f"add_{hop}_hop_complete"]:
                        progress.update(1)
                        continue
                    with LmdbStore(str(builder.entity_mapping_path), "w") as mapping:
                        # The builder clears its pending set after each call.
                        # Reuse it for the next hop; newly introduced hops need all rows.
                        rows = pending_rows | BitMap(
                            int(mapping[f"id:{entity_id}".encode()])
                            for entity_id in affected_entity_ids
                        )
                        if hop > self.previous_max_hop and self.previous_max_hop >= 2:
                            rows = BitMap(range(int(mapping.get(b"next_row", b"0"))))
                        builder._mark_later_hop_dirty(mapping, rows)
                    builder.max_hop = hop
                    later_hop_rows.update(builder.add_later_hops(
                        high_degree_limit, batch_size=batch_size,
                        verbose=self.verbose, start_hop=hop))
                    await self.save_phase_checkpoint(checkpoints, f"add_{hop}_hop_complete")
                    progress.update(1)

        metadata = {
            "indexed_version": indexed_version,
            "max_hop": max_hop,
            "path": str(self.reachability_output_dir),
            "output_dir": str(self.reachability_output_dir),
            "entity_count": builder.last_entity_row + 1,
            "new_entity_rows": rows_added,
            "one_hop_rows": len(affected_entity_ids),
            "one_hop_additions": one_hop_additions,
            "later_hop_rows": later_hop_rows,
        }
        temporary_pointer_path = self.current_bitmap_path.with_suffix(self.current_bitmap_path.suffix + ".tmp")
        temporary_pointer_path.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_pointer_path.replace(self.current_bitmap_path)
        self.previous_max_hop = max_hop
        async with self.lock:
            try:
                await self.conn.execute("DELETE FROM new_and_updated_entity")
                await self.conn.execute("DELETE FROM new_and_updated_relationship")
                await self.conn.execute(
                    "UPDATE global_config SET bitmap_in_progress = FALSE, "
                    "bitmap_pending_inputs = '{}' WHERE id = 1"
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        self.global_config["bitmap_in_progress"] = False
        self.global_config["bitmap_pending_inputs"] = {}
        self.bitmap_in_progress = False
        # Resume state is no longer needed after completion is committed.
        for checkpoint_dir in self.reachability_output_dir.glob(".batch-checkpoint-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
        return metadata

    async def construct_GraphRAG(self, input_document_list=None, entity_types=None, debug_mode=False, async_cooldown=60, requests_per_minute=1000, global_desc=True,
                               entity_extract_prompt=None, entity_input_example=None, entity_output_example=None,
                               relationship_extract_prompt=None, relationship_input_example=None,
                               relationship_output_example=None, relationship_entity_example=None) -> None:
                    """Constructs a graph-based retrieval-augmented generation (RAG) system from the provided documents. This function orchestrates the entire process, including document ingestion, entity extraction and embedding, relationship extraction and embedding, global description generation, graph construction, and BM25 index creation."""
                    self.global_config = await self.get_global_config()
                    self.bitmap_in_progress = self.global_config.get("bitmap_in_progress", False)
                    if self.bitmap_in_progress:
                        self._graph_construction_print("Resuming unfinished bitmap construction.")
                        await self.construct_reachability_map(
                            **self.global_config["bitmap_pending_inputs"]
                        )
                        
                    for table, flag in (("entity", "entity_bm25_in_progress"),
                                        ("relationship", "relationship_bm25_in_progress")):
                        if self.global_config.get(flag, False):
                            self._graph_construction_print(f"Resuming unfinished {table} BM25 index.")
                            await self.create_BM25_index_for_table(table)

                    # First, we ingest the documents into the database, which includes extracting text, chunking it, and adding embeddings.
                    try:
                        self._graph_construction_print("STEP ONE: Chunking & Embedding a Given Corpus")
                        await self.ingest_documents(input_document_list=input_document_list,
                                                    async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while ingesting text chunks: {e}")
                    # Second, we extract, classify, and embed entities from the chunks.
                    try:
                       self._graph_construction_print("STEP TWO: Extracting, Classifying, and Embedding Entities")
                       new_entities, updated_entities = await self.extract_and_embed_entities(
                           entity_types=entity_types, async_cooldown=async_cooldown, requests_per_minute=requests_per_minute,
                           entity_extract_prompt=entity_extract_prompt,
                           entity_input_example=entity_input_example,
                           entity_output_example=entity_output_example,
                       )
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while adding in the entities: {e}")
        
                     # Third, we extract relationships from the entities and add them to the database.
                    try:
                        self._graph_construction_print("STEP THREE: Extracting Relationships from the Ingested/Unprocessed Chunks")
                        new_relationships, updated_relationships = await self.extract_and_embed_relationships(
                            async_cooldown=async_cooldown, requests_per_minute=requests_per_minute,
                            relationship_extract_prompt=relationship_extract_prompt,
                            relationship_input_example=relationship_input_example,
                            relationship_output_example=relationship_output_example,
                            relationship_entity_example=relationship_entity_example,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while adding in the relationships: {e}")
        
                    # Fourth, generate and save a global description for each source document.
                    try:
                        self._graph_construction_print("STEP FOUR: Generating a Global Description for the entity-relationship graph")
                        await self.generate_global_descriptions(debug_mode=debug_mode, global_desc=global_desc)
                        pending_inputs = {
                            "new_entities": new_entities,
                            "updated_entities": updated_entities,
                            "new_relationships": [
                                {"relation_id": relationship["relation_id"],
                                 "src_entity_id": relationship["src_entity_id"],
                                 "dst_entity_id": relationship["dst_entity_id"]}
                                for relationship in new_relationships
                            ],
                            "updated_relationships": [
                                {"relation_id": relationship["relation_id"],
                                 "src_entity_id": relationship["src_entity_id"],
                                 "dst_entity_id": relationship["dst_entity_id"]}
                                for relationship in updated_relationships
                            ],
                        }
                        await self.conn.execute(
                            "UPDATE global_config SET bitmap_in_progress = TRUE, "
                            "entity_bm25_in_progress = TRUE, relationship_bm25_in_progress = TRUE, "
                            "bitmap_phase_checkpoints = '{}', bitmap_pending_inputs = ? WHERE id = 1",
                            (json.dumps(pending_inputs),),
                        )
                        await self.conn.commit()
                        self.global_config["bitmap_in_progress"] = True
                        self.global_config["entity_bm25_in_progress"] = True
                        self.global_config["relationship_bm25_in_progress"] = True
                        self.global_config["bitmap_phase_checkpoints"] = {}
                        self.global_config["bitmap_pending_inputs"] = pending_inputs
                    except Exception as e:
                                traceback.print_exc()
                                raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while generating global descriptions: {e}")
                    
                    # Fifth, we need to generate the graph representation of the database.
                    try:
                        pass
                        self._graph_construction_print("STEP FIVE: Creating the reachability map of the Database")
                        await self.construct_reachability_map(
                            new_entities=new_entities,
                            new_relationships=new_relationships,
                            updated_entities=updated_entities,
                            updated_relationships=updated_relationships,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while clustering entities: {e}")
                    # Lastly, we construct the BM25 index
                    try:
                       self._graph_construction_print("STEP SIX: Creating the BM25 Index")
                       await self.create_BM25_index_for_table("entity")
                       await self.create_BM25_index_for_table("relationship")  
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while creating the BM25 index: {e}")
