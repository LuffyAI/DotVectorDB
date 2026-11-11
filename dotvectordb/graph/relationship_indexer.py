from dotvectordb.construction_prompts.relationship_prompts import (
    RELATIONSHIP_PROMPT,
    RELATIONSHIP_SUMMARIZATION_PROMPT,
    ENTITY_EXAMPLE,
    INPUT_EXAMPLE,
    OUTPUT_EXAMPLE,
    Relationships,
    RelationshipSummarizer,
    prepare_relationship_enrichment_prompt_sections,
)
import json
import re
import os
import traceback
import asyncio
import numpy as np
from model2vec import StaticModel
from tqdm.asyncio import tqdm_asyncio

class RelationshipIndexer:
    def __init__(
        self,
        conn,
        lock,
        language_model,
        hash_func,
        embedding_model,
        add_embedding_func,
        max_relation_keywords=None,
        max_relation_keywords_token_size=None,
        max_keyword_token_size=None,
        max_entity_relation_description_token_size=None,
        recommended_entity_relation_description_token_size=None,
        summary_language=None,
        relationship_extract_prompt=None,
        relationship_input_example=None,
        relationship_output_example=None,
        relationship_entity_example=None,
    ):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.hash_func = hash_func
        self.embedding_model = embedding_model
        self.add_embedding_func = add_embedding_func
        self.relationship_extract_prompt = RELATIONSHIP_PROMPT if relationship_extract_prompt is None else relationship_extract_prompt
        self.relationship_input_example = INPUT_EXAMPLE if relationship_input_example is None else relationship_input_example
        self.relationship_output_example = OUTPUT_EXAMPLE if relationship_output_example is None else relationship_output_example
        self.relationship_entity_example = ENTITY_EXAMPLE if relationship_entity_example is None else relationship_entity_example
        self.max_relation_keywords = (
            max_relation_keywords
            if max_relation_keywords is not None
            else int(os.getenv("MAX_RELATION_KEYWORDS", 8))
        )
        self.max_relation_keywords_token_size = (
            max_relation_keywords_token_size
            if max_relation_keywords_token_size is not None
            else int(os.getenv("MAX_RELATION_KEYWORDS_TOKEN_SIZE", 48))
        )
        self.max_keyword_token_size = (
            max_keyword_token_size
            if max_keyword_token_size is not None
            else int(os.getenv("MAX_KEYWORD_TOKEN_SIZE", 8))
        )
        self.max_entity_relation_description_token_size = (
            max_entity_relation_description_token_size
            if max_entity_relation_description_token_size is not None
            else int(
                os.getenv("MAX_ENTITY_RELATION_DESCRIPTION_TOKEN_SIZE", 1200)
            )
        )
        self.recommended_entity_relation_description_token_size = (
            recommended_entity_relation_description_token_size
            if recommended_entity_relation_description_token_size is not None
            else int(
                os.getenv(
                    "RECOMMENDED_ENTITY_RELATION_DESCRIPTION_TOKEN_SIZE",
                    600,
                )
            )
        )
        self.summary_language = (
            summary_language
            if summary_language is not None
            else os.getenv("DEFAULT_SUMMARY_LANGUAGE", "English")
        )
        self.small_llm = StaticModel.from_pretrained("minishlab/potion-base-32M")

    async def log_failure(
        self,
        phase,
        item_type,
        item_id,
        error,
        lock_held=False,
    ):
        """Persist a processing failure without masking the original error."""
        async def insert_failure():
            await self.conn.execute(
                """
                INSERT INTO failure_log (
                    phase, item_type, item_id, error_message
                ) VALUES (?, ?, ?, ?)
                """,
                (phase, item_type, str(item_id), str(error)),
            )
            await self.conn.commit()

        try:
            if lock_held:
                await insert_failure()
            else:
                async with self.lock:
                    await insert_failure()
        except Exception as logging_error:
            traceback.print_exc()
            print(f"Unable to save failure log: {logging_error}")

    async def remove_extra_keywords_from_relationship(
        self,
        keywords,
        threshold=0.7,
    ):
        """Deduplicate semantically similar relationship keywords."""
        if not keywords:
            return []
        embeddings = self.small_llm.encode(keywords)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        sorted_indices = sorted(
            range(len(keywords)),
            key=lambda index: len(keywords[index]),
        )
        kept_indices = []
        for index in sorted_indices:
            if all(
                np.dot(embeddings[index], embeddings[kept_index]) < threshold
                for kept_index in kept_indices
            ):
                kept_indices.append(index)
        kept_indices.sort()
        return [keywords[index] for index in kept_indices]

    def find_least_similar_keywords(self, keywords):
        """Keep the most semantically diverse keywords up to the limit."""
        if len(keywords) <= self.max_relation_keywords:
            return keywords
        if self.max_relation_keywords <= 0:
            return []

        embeddings = self.small_llm.encode(keywords)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings),
            where=norms != 0,
        )
        similarities = embeddings @ embeddings.T
        average_similarities = (
            similarities.sum(axis=1) - np.diag(similarities)
        ) / (len(keywords) - 1)
        selected_indices = [int(np.argmin(average_similarities))]
        remaining_indices = set(range(len(keywords))) - set(selected_indices)

        while len(selected_indices) < self.max_relation_keywords:
            next_index = min(
                remaining_indices,
                key=lambda index: (
                    max(
                        similarities[index, selected_index]
                        for selected_index in selected_indices
                    ),
                    index,
                ),
            )
            selected_indices.append(next_index)
            remaining_indices.remove(next_index)

        selected_indices.sort()
        return [keywords[index] for index in selected_indices]

    def remove_keywords_over_token_limits(
        self,
        keywords,
        enforce_total_limit=True,
    ):
        """Apply per-keyword and aggregate relationship keyword limits."""
        keywords_with_token_counts = [
            (keyword, len(self.embedding_model.tokenize(keyword)))
            for keyword in keywords
            if isinstance(keyword, str) and keyword.strip()
        ]
        keywords_with_token_counts = [
            (keyword, token_count)
            for keyword, token_count in keywords_with_token_counts
            if token_count <= self.max_keyword_token_size
        ]

        while (
            enforce_total_limit
            and sum(token_count for _, token_count in keywords_with_token_counts)
            > self.max_relation_keywords_token_size
            and keywords_with_token_counts
        ):
            longest_index = max(
                range(len(keywords_with_token_counts)),
                key=lambda index: (
                    keywords_with_token_counts[index][1],
                    index,
                ),
            )
            keywords_with_token_counts.pop(longest_index)

        return [keyword for keyword, _ in keywords_with_token_counts]
    ############################################################
    #   Token Economics Tracking                               # 
    ############################################################
    async def save_input_and_output_tokens_for_llm(self, total_tokens):
                await self.conn.execute(
                    """
                    INSERT INTO llm_token_usage (stage, total_tokens)
                    VALUES (?, ?)
                    """,
                    ("relation", total_tokens)
                )
    #################################################################
    #   Fetch Relationship-Related Data For Relationship Indexing   # 
    #################################################################
    async def get_chunks_waiting_for_relation_extraction(self):
        """Gets entity types from the database and returns them as a dictionary"""
        async with self.lock:
            cursor = await self.conn.execute(
            """SELECT content FROM chunk WHERE is_embedded = 1 AND are_entities_extracted = 1 AND are_relations_extracted = 0"""
        )
        chunks = await cursor.fetchall()
        await cursor.close()
        return chunks
    
    async def get_entities_connected_to_chunk(self, chunk_id):
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT entity_id FROM chunk_to_entity_mapping WHERE chunk_id = ?", (chunk_id,)
            )
            rows = await cursor.fetchall()
            await cursor.close()
            entity_ids = [row[0] for row in rows]
            return entity_ids

    async def get_relationships_waiting_for_embedding(self):
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM relationship WHERE is_embedded = 0"
            )
            rows = await cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            await cursor.close()

            relationships = [
                dict(zip(column_names, row))
                for row in rows
            ]
            return relationships

    async def get_relationships_requiring_summarization(self):
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT
                    relationship.*,
                    src_entity.entity_name AS src_entity_name,
                    dst_entity.entity_name AS dst_entity_name
                FROM relationship
                JOIN entity AS src_entity
                    ON relationship.src_entity_id = src_entity.entity_id
                JOIN entity AS dst_entity
                    ON relationship.dst_entity_id = dst_entity.entity_id
                WHERE relationship.is_summarized = FALSE
                """
            )
            rows = await cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            await cursor.close()

        return [dict(zip(column_names, row)) for row in rows]
    
   ##########################################################################
   #   Handles all preparation needed to prepare the relationship prompt    # 
   ##########################################################################
    async def get_relationship_prompt_for_embedding(self, relationship):
        formatted_relationship = {
        "src_entity_id": relationship["src_entity_id"],
        "dst_entity_id": relationship["dst_entity_id"],
        "relation_description": relationship["relation_description"],
        "keywords": relationship["keywords"],
        }
        return formatted_relationship
    
    async def prepare_local_entity_mapping_for_chunk(self, entity_ids):
         """Given a list of entity IDS, we create two mappings: one from a number to entity id and another from number to entity names"""
         number_to_entity_id_map = {}
         number_to_entity_name_map = {}
         start = 1
    
         for entity_id in entity_ids:
              async with self.lock:
                cursor = await self.conn.execute(
                     "SELECT entity_name FROM entity WHERE entity_id = ?", (entity_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row:
                     entity_name = row[0]
                     number_to_entity_id_map[start] = entity_id
                     number_to_entity_name_map[start] = entity_name
                     start += 1
         return number_to_entity_id_map, number_to_entity_name_map

    async def get_sentence_highlighted_version_of_chunk(self, chunk_id):
        """Return numbered sentence text and its valid sentence-number range."""
        async with self.conn.execute(
            """
            SELECT sentence_number, sentence_text
            FROM chunk_to_sentence_mapping
            WHERE chunk_id = ?
            ORDER BY sentence_number
            """,
            (chunk_id,),
        ) as cursor:
            sentences = await cursor.fetchall()
        return " ".join(f"[L{number}] {text}" for number, text in sentences), range(1, len(sentences) + 1)
    
    ##########################################################################
    #       Relationship Extraction & Ingestion Operations for Indexing      # 
    ##########################################################################
    async def extract_relation(self, chunk, cooldown=30):
        if not self.conn:
            raise RuntimeError("Database not opened")
        
        # We fill the relationship recognition prompt with the entity mapping and the chunk of text we want to analyze.
        chunk_hash = self.hash_func(chunk)
        entity_ids = await self.get_entities_connected_to_chunk(chunk_hash)
        number_to_entity_id_map, number_to_entity_name_map = await self.prepare_local_entity_mapping_for_chunk(entity_ids)
        transformed_chunk, sentence_range = await self.get_sentence_highlighted_version_of_chunk(chunk_hash)
        embedding_model_hash = self.hash_func(self.embedding_model.model_name)
        embedding_model_hash_str = f"{embedding_model_hash}"
        RELATIONSHIP_PROMPT_FILLED = self.relationship_extract_prompt.format(
            entities=json.dumps(number_to_entity_name_map),
            chunk=transformed_chunk,
            entity_example=self.relationship_entity_example,
            input_example=self.relationship_input_example,
            output_example=self.relationship_output_example,
            language=self.summary_language,
        )

        # We call the language model with the filled prompt and expect a response in the format of ExtractedEntitiesResponse.
        try:
          response = await self.language_model(RELATIONSHIP_PROMPT_FILLED, structuredInput=Relationships)
        except Exception as e:
           await self.log_failure("relation", "chunk", chunk_hash, e)
           print(f"Error extracting relationships: {e}")
           return 0

        async with self.lock:
            try:
                new_relationships_added = 0
                for relation in response[0]["relations"]:
                    # We extract key metadata from the LLM response to add to the database.
                    relation_str = json.dumps(relation, indent=2)
                    tokens = len(self.embedding_model.tokenize(relation_str))
                    src_entity_id = relation["src_entity_id"]
                    dst_entity_id = relation["dst_entity_id"]
                    description = relation["description"]
                    supporting_lines = relation.get("supporting_lines")
                    if not isinstance(supporting_lines, list):
                        raise ValueError("supporting_lines must be a list of line labels")
                    normalized_lines = set()
                    for line in supporting_lines:
                        if not isinstance(line, str) or re.fullmatch(r"l[1-9][0-9]*", line.lower()) is None:
                            raise ValueError(f"Invalid supporting line label: {line!r}")
                        line = line.lower()
                        if int(line[1:]) not in sentence_range:
                            raise ValueError(f"Supporting line {line!r} is outside the chunk's sentence range")
                        normalized_lines.add(line)
                    supporting_lines = sorted(normalized_lines, key=lambda line: int(line[1:]))
                    chunk_key = str(chunk_hash)
                    try:
                        true_src_entity_id = number_to_entity_id_map[src_entity_id]
                        true_dst_entity_id = number_to_entity_id_map[dst_entity_id]
                    except KeyError:
                        print(f"Invalid relationship entity ID returned by LLM.")
                        print(f"number_to_entity_id_map: {number_to_entity_id_map}")
                        print(f"LLM relationship response: {relation}")
                        raise

                    # To generate a unique relation ID, we concatenate the source and destination entity IDs and hash them.
                    relation_id = f"{true_src_entity_id}_{true_dst_entity_id}"
                    relation_id_hash = self.hash_func(relation_id)

                    keywords = relation["keywords"]
                    is_summarized = (
                        len(self.embedding_model.tokenize(description))
                        <= self.max_entity_relation_description_token_size
                    )
                    
                    # Duplicate check: We check if the relationship already exists in the database by comparing the source and destination entity IDs.
                    dupeCheck = await self.conn.execute(
                    """
                    SELECT relation_description, keywords, tokens, supporting_lines
                    FROM relationship
                    WHERE relation_id = ?
                    """,
                    (relation_id_hash,)
                    )
                    existing_relation = await dupeCheck.fetchone()
                    await dupeCheck.close()

                    if not existing_relation:
                        await self.conn.execute(
                            """
                            INSERT INTO relationship (
                                relation_id,
                                src_entity_id,
                                dst_entity_id,
                                relation_description,
                                embedding_model_id,
                                keywords,
                                supporting_lines,
                                tokens,
                                is_summarized
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                relation_id_hash,
                                true_src_entity_id,
                                true_dst_entity_id,
                                description,
                                embedding_model_hash_str,
                                json.dumps(keywords),
                                json.dumps({chunk_key: supporting_lines}),
                                tokens,
                                is_summarized,
                            ),
                        )
                        new_relationships_added += 1
                    else:
                        old_description = existing_relation[0]
                        old_keywords = json.loads(existing_relation[1])
                        old_tokens = existing_relation[2]
                        updated_supporting_lines = json.loads(existing_relation[3])
                        chunk_lines = {line.lower() for line in updated_supporting_lines.get(chunk_key, [])}
                        chunk_lines.update(supporting_lines)
                        updated_supporting_lines[chunk_key] = sorted(chunk_lines, key=lambda line: int(line[1:]))
                        updated_description = old_description + "<SEP>" + description
                        updated_keywords = old_keywords + keywords
                        updated_tokens = old_tokens + tokens
                        is_summarized = (
                            len(
                                self.embedding_model.tokenize(
                                    updated_description
                                )
                            )
                            <= self.max_entity_relation_description_token_size
                        )
                        await self.conn.execute(
                        """
                        UPDATE relationship
                        SET relation_description = ?,
                            keywords = ?,
                            supporting_lines = ?,
                            tokens = ?,
                            is_summarized = ?,
                            is_embedded = FALSE
                        WHERE relation_id = ?
                        """,
                        (
                            updated_description,
                            json.dumps(updated_keywords),
                            json.dumps(updated_supporting_lines),
                            updated_tokens,
                            is_summarized,
                            relation_id_hash,
                        )
                        )
                    await self.conn.execute(
                        """
                        INSERT OR IGNORE INTO chunk_to_relationship_mapping
                        (chunk_id, relation_id) VALUES (?, ?)
                        """,
                        (chunk_hash, relation_id_hash),
                    )
                await self.conn.execute("UPDATE chunk SET are_relations_extracted = 1 WHERE chunk_id = ?", (chunk_hash,))
                await self.save_input_and_output_tokens_for_llm(response[1])
                await self.conn.commit()
                return new_relationships_added
            except Exception as e:
                await self.conn.rollback()
                await self.log_failure(
                    "relation",
                    "chunk",
                    chunk_hash,
                    e,
                    lock_held=True,
                )
                traceback.print_exc()
                print(f"Error saving extracted relationships to database: {e}")
                return 0

    async def extract_relations_from_chunks(self, async_cooldown=60,requests_per_minute=1000,verbose=True):
     # Check if the database connection is established
     if not self.conn:
            raise RuntimeError("Database not opened")
        
     MAX_TRIES = 4
     TRY = 0
     while True:
        # First, we pull only the chunks that haven't been processed for relationship extraction yet.
        chunks = await self.get_chunks_waiting_for_relation_extraction()
        if not chunks:
            break
        if TRY >= MAX_TRIES:
            raise RuntimeError(
                f"Relationship extraction failed after {MAX_TRIES} attempts: "
                f"{len(chunks)} chunks still require extraction."
            )

        TRY += 1

        if verbose:
            print(f"Found {len(chunks)} chunks waiting for relationship extraction.")

        total_chunks = len(chunks)

        # We do not know how many unique entities will be added in advance.
        relation_progress = tqdm_asyncio(
        total=None,
        desc=" Extracting Relationships",
        bar_format="{desc}: {n} relationships [{elapsed}, {rate_fmt}]"
        )

        # We process the chunks in batches starting from 0 to the total number of chunks, with a step size of requests_per_minute.
        for batch_start in range(0, total_chunks, requests_per_minute):
            batch = chunks[
                batch_start:batch_start + requests_per_minute
            ]

            tasks = []
            for chunk in batch:
                tasks.append(
                    self.extract_relation(
                        chunk=chunk[0]
                    )
                )

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )
            for chunk, result in zip(batch, results):
                if isinstance(result, Exception):
                    await self.log_failure(
                        "relation",
                        "chunk",
                        self.hash_func(chunk[0]),
                        result,
                    )
            # Each successful result is the number of NEW relationships
            # inserted by extract_relation().
            for result in results:
                if not isinstance(result, Exception):
                    relation_progress.update(result)

            if verbose:
                batch_number = (
                    batch_start // requests_per_minute
                ) + 1

                print(
                    f"Finished batch {batch_number}: "
                    f"{len(batch)} chunks processed"
                )

                # We check for any exceptions that occurred during the processing of the batch and print them out.
                for index, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(
                            f"Chunk {batch_start + index} failed: "
                            f"{result}"
                        )

            # To handle rate limiting, we check if there are more chunks to process after the current batch.
            # If there are, we wait for the specified cooldown period before processing the next batch.
            has_more_chunks = (
                batch_start + requests_per_minute < total_chunks
            )

            if has_more_chunks:
                if verbose:
                    print(
                        f"Cooling down for "
                        f"{async_cooldown} seconds..."
                    )

                await asyncio.sleep(async_cooldown)
    
    async def fix_relationships(
        self,
        async_cooldown=60,
        requests_per_minute=1000,
        verbose=True,
    ):
        if not self.conn:
            raise RuntimeError("Database connection is not established")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")

        relationships = await self.get_relationships_requiring_summarization()

        async def summarize_relationship(relationship):
            try:
                descriptions = [
                    description.strip()
                    for description in relationship[
                        "relation_description"
                    ].split("<SEP>")
                    if description.strip()
                ]
                description_list = "\n".join(
                    json.dumps(
                        {"Description": description},
                        ensure_ascii=False,
                    )
                    for description in descriptions
                )
                relationship_name = (
                    f"{relationship['src_entity_name']} -> "
                    f"{relationship['dst_entity_name']}"
                )
                prompt = RELATIONSHIP_SUMMARIZATION_PROMPT.format(
                    summary_length=(
                        self.recommended_entity_relation_description_token_size
                    ),
                    description_type="Relationship",
                    description_name=relationship_name,
                    description_list=description_list,
                    language=self.summary_language,
                )
                response = await self.language_model(
                    prompt,
                    structuredInput=RelationshipSummarizer,
                )
                new_description = response[0]["new_description"].strip()
                if not new_description:
                    raise ValueError(
                        f"Relationship summarizer returned an empty description "
                        f"for '{relationship_name}'."
                    )

                description_tokens = len(
                    self.embedding_model.tokenize(new_description)
                )
                if (
                    description_tokens
                    > self.max_entity_relation_description_token_size
                ):
                    raise ValueError(
                        f"Summary for relationship '{relationship_name}' "
                        f"contains {description_tokens} tokens; the maximum is "
                        f"{self.max_entity_relation_description_token_size}."
                    )

                async with self.lock:
                    await self.conn.execute(
                        """
                        UPDATE relationship
                        SET relation_description = ?, is_summarized = TRUE, is_embedded = FALSE
                        WHERE relation_id = ?
                        """,
                        (new_description, relationship["relation_id"]),
                    )
                    await self.save_input_and_output_tokens_for_llm(response[1])
                    await self.conn.commit()
                return None
            except Exception as error:
                await self.log_failure(
                    "desc",
                    "relation",
                    relationship.get("relation_id", "unknown"),
                    error,
                )
                if verbose:
                    print(
                        f"Summarization failed for relationship "
                        f"'{relationship.get('relation_id')}': {error}"
                    )
                traceback.print_exc()
                return relationship

        failed_relationships = relationships.copy()
        while failed_relationships:
            current_batch = failed_relationships[:requests_per_minute]
            remaining_relationships = failed_relationships[
                requests_per_minute:
            ]
            batch_results = await tqdm_asyncio.gather(
                *(
                    summarize_relationship(relationship)
                    for relationship in current_batch
                ),
                desc="Summarizing relationships",
                total=len(current_batch),
            )
            failed_relationships = [
                relationship
                for relationship in batch_results
                if relationship is not None
            ]
            failed_relationships.extend(remaining_relationships)

            if failed_relationships:
                if verbose:
                    print(
                        f"Cooldown for {async_cooldown} seconds. "
                        f"{len(failed_relationships)} relationships remaining."
                    )
                await asyncio.sleep(async_cooldown)

    async def embed_relationships(self, new_relationships, updated_relationships, async_cooldown=60, requests_per_minute=1000, verbose=True):
        new_relationship_set = {relationship["relation_id"] for relationship in new_relationships}
        updated_relationship_set = {relationship["relation_id"] for relationship in updated_relationships}
        relationships = await self.get_relationships_waiting_for_embedding()
        print(f"Found {len(relationships)} relationships waiting for embedding.")

        async def embed_relationship(relationship):
            try:
                relation_id = relationship["relation_id"]
                if relation_id in new_relationship_set:
                    status = "new"
                elif relation_id in updated_relationship_set:
                    status = "updated"
                else:
                    raise ValueError(f"Relationship {relation_id} is not tracked as new or updated")
                content = await self.get_relationship_prompt_for_embedding(relationship)
                keywords = content["keywords"]
                if isinstance(keywords, str):
                    keywords = json.loads(keywords)
                keywords = self.remove_keywords_over_token_limits(
                    keywords,
                    enforce_total_limit=False,
                )
                content["keywords"] = (
                    await self.remove_extra_keywords_from_relationship(keywords)
                )
                content["keywords"] = self.find_least_similar_keywords(
                    content["keywords"]
                )
                content["keywords"] = self.remove_keywords_over_token_limits(
                    content["keywords"]
                )
                if verbose:
                    print(f"Prompt for relationship {relationship['relation_id']}: {content}")
                await self.add_embedding_func(
                    content=json.dumps(content, indent=4),
                    table="relationship",
                    pk_hash=relation_id,
                    status=status,
                )
                return None
            except Exception as error:
                await self.log_failure(
                    "relation",
                    "relation",
                    relationship.get("relation_id", "unknown"),
                    error,
                )
                if verbose:
                    print(f"Embedding failed, retrying next batch: {error}")
                traceback.print_exc()
                return relationship

        failed_relationships = relationships.copy()
        while failed_relationships:
            current_batch = failed_relationships[:requests_per_minute]
            print(f"Processing batch of {len(current_batch)} relationships for embedding.")
            remaining_relationships = failed_relationships[requests_per_minute:]
            batch_results = await tqdm_asyncio.gather(
                *(embed_relationship(relationship) for relationship in current_batch),
                desc="Embedding relationships",
                total=len(current_batch),
            )
            failed_relationships = [
                relationship for relationship in batch_results
                if relationship is not None
            ]
            failed_relationships.extend(remaining_relationships)

            if failed_relationships:
                if verbose:
                    print(
                        f"Cooldown for {async_cooldown} seconds. "
                        f"{len(failed_relationships)} relationships remaining."
                    )
                await asyncio.sleep(async_cooldown)
        return relationships

    ##########################################################################
    #       Custom Relationship Functions                                    # 
    ##########################################################################
    async def prepare_custom_relationship_entity_mapping(self, relationships):
        """Map real endpoint IDs to temporary IDs and names for one chunk."""
        entity_ids = []
        seen_entity_ids = set()
        for relationship in relationships:
            for entity_id in (
                relationship["src_entity_id"],
                relationship["dst_entity_id"],
            ):
                if entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    entity_ids.append(entity_id)

        entity_id_to_local_id = {
            entity_id: local_entity_id
            for local_entity_id, entity_id in enumerate(entity_ids, start=1)
        }
        prompt_entities = {}
        async with self.lock:
            for entity_id in entity_ids:
                cursor = await self.conn.execute(
                    "SELECT entity_name FROM entity WHERE entity_id = ?",
                    (entity_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise ValueError(
                        f"Custom relationship references entity '{entity_id}', "
                        "but that entity is not present in the database."
                    )
                prompt_entities[entity_id_to_local_id[entity_id]] = row[0]

        return entity_id_to_local_id, prompt_entities

    async def enrich_custom_relationship_group(
            self,
            relationships,
            chunk_content,
            requirements,
            entity_id_to_local_id,
            prompt_entities,
        ):
            """Enrich relationships from one chunk with identical requirements."""
            (
                needs_descriptions,
                needs_keywords,
            ) = requirements
            prompt_template, enrichment_model = (
                await prepare_relationship_enrichment_prompt_sections(
                    needs_descriptions=needs_descriptions,
                    needs_keywords=needs_keywords,
                )
            )
            local_id_to_relationship = {
                local_relationship_id: relationship
                for local_relationship_id, relationship in enumerate(
                    relationships,
                    start=1,
                )
            }
            prompt_relationships = [
                {
                    "rel_id": local_relationship_id,
                    "src_entity_id": entity_id_to_local_id[
                        relationship["src_entity_id"]
                    ],
                    "dst_entity_id": entity_id_to_local_id[
                        relationship["dst_entity_id"]
                    ],
                }
                for local_relationship_id, relationship
                in local_id_to_relationship.items()
            ]
            prompt = prompt_template.format(
                language=self.summary_language,
                entities=json.dumps(
                    prompt_entities,
                    ensure_ascii=False,
                    indent=2,
                ),
                relationships=json.dumps(
                    prompt_relationships,
                    ensure_ascii=False,
                    indent=2,
                ),
                chunk=chunk_content,
                entity_example=ENTITY_EXAMPLE,
                input_example=INPUT_EXAMPLE,
                output_example=OUTPUT_EXAMPLE,
            )
            response = await self.language_model(
                prompt,
                structuredInput=enrichment_model,
            )
            returned_relationships = response[0]["relationships"]
            returned_local_ids = [
                relationship["local_relationship_id"]
                for relationship in returned_relationships
            ]
            expected_local_ids = set(local_id_to_relationship)
            if (
                len(returned_relationships) != len(local_id_to_relationship)
                or len(returned_local_ids) != len(set(returned_local_ids))
                or set(returned_local_ids) != expected_local_ids
            ):
                raise ValueError(
                    "Relationship enrichment response IDs did not match the "
                    f"requested local IDs. Expected {sorted(expected_local_ids)}, "
                    f"received {returned_local_ids}."
                )

            enriched_relationships = []
            for returned_relationship in returned_relationships:
                local_relationship_id = returned_relationship[
                    "local_relationship_id"
                ]
                enriched_relationship = dict(
                    local_id_to_relationship[local_relationship_id]
                )
                if needs_descriptions:
                    description = returned_relationship[
                        "relation_description"
                    ].strip()
                    if not description:
                        raise ValueError(
                            "Generated relationship description cannot be empty "
                            f"for local relationship {local_relationship_id}."
                        )
                    enriched_relationship["relation_description"] = description
                if needs_keywords:
                    enriched_relationship["keywords"] = returned_relationship[
                        "keywords"
                    ]
                enriched_relationships.append(enriched_relationship)

            return enriched_relationships, response[1]

    async def save_custom_relationships_for_chunk(
            self,
            chunk_id,
            relationships,
            llm_token_totals,
        ):
            """Atomically save all custom relationships assigned to one chunk."""
            embedding_model_hash = str(
                self.hash_func(self.embedding_model.model_name)
            )
            new_relationships_added = 0

            async with self.lock:
                try:
                    for relationship in relationships:
                        relationship_record = {
                            "src_entity_id": relationship["src_entity_id"],
                            "dst_entity_id": relationship["dst_entity_id"],
                            "relation_description": relationship[
                                "relation_description"
                            ],
                            "keywords": relationship["keywords"],
                        }
                        tokens = len(
                            self.embedding_model.tokenize(
                                json.dumps(
                                    relationship_record,
                                    ensure_ascii=False,
                                )
                            )
                        )
                        cursor = await self.conn.execute(
                            """
                            SELECT relation_description, keywords, tokens
                            FROM relationship
                            WHERE relation_id = ?
                            """,
                            (relationship["relation_id"],),
                        )
                        existing_relationship = await cursor.fetchone()
                        await cursor.close()

                        if existing_relationship is None:
                            is_summarized = (
                                len(
                                    self.embedding_model.tokenize(
                                        relationship["relation_description"]
                                    )
                                )
                                <= self.max_entity_relation_description_token_size
                            )
                            await self.conn.execute(
                                """
                                INSERT INTO relationship (
                                    relation_id,
                                    src_entity_id,
                                    dst_entity_id,
                                    relation_description,
                                    keywords,
                                    embedding_model_id,
                                    tokens,
                                    is_summarized
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    relationship["relation_id"],
                                    relationship["src_entity_id"],
                                    relationship["dst_entity_id"],
                                    relationship["relation_description"],
                                    json.dumps(relationship["keywords"]),
                                    embedding_model_hash,
                                    tokens,
                                    is_summarized,
                                ),
                            )
                            new_relationships_added += 1
                        else:
                            (
                                old_description,
                                old_keywords_json,
                                old_tokens,
                            ) = existing_relationship
                            description_segments = [
                                description.strip()
                                for description in (
                                    old_description.split("<SEP>")
                                    + relationship[
                                        "relation_description"
                                    ].split("<SEP>")
                                )
                                if description.strip()
                            ]
                            deduplicated_segments = []
                            seen_description_hashes = set()
                            for description in description_segments:
                                normalized_description = " ".join(
                                    description.split()
                                ).casefold()
                                description_hash = self.hash_func(
                                    normalized_description
                                )
                                if description_hash in seen_description_hashes:
                                    continue
                                seen_description_hashes.add(description_hash)
                                deduplicated_segments.append(description)
                            combined_description = "<SEP>".join(
                                deduplicated_segments
                            )

                            if self.hash_func(
                                old_description
                            ) != self.hash_func(combined_description):
                                combined_keywords = (
                                    json.loads(old_keywords_json)
                                    + relationship["keywords"]
                                )
                                is_summarized = (
                                    len(
                                        self.embedding_model.tokenize(
                                            combined_description
                                        )
                                    )
                                    <= self.max_entity_relation_description_token_size
                                )
                                await self.conn.execute(
                                    """
                                    UPDATE relationship
                                    SET relation_description = ?,
                                        keywords = ?,
                                        tokens = ?,
                                        is_summarized = ?,
                                        is_embedded = FALSE
                                    WHERE relation_id = ?
                                    """,
                                    (
                                        combined_description,
                                        json.dumps(combined_keywords),
                                        old_tokens + tokens,
                                        is_summarized,
                                        relationship["relation_id"],
                                    ),
                                )

                        await self.conn.execute(
                            """
                            INSERT OR IGNORE INTO chunk_to_relationship_mapping
                            (chunk_id, relation_id) VALUES (?, ?)
                            """,
                            (chunk_id, relationship["relation_id"]),
                        )

                    for llm_tokens in llm_token_totals:
                        await self.save_input_and_output_tokens_for_llm(llm_tokens)
                    await self.conn.execute(
                        """
                        UPDATE chunk
                        SET are_relations_extracted = 1
                        WHERE chunk_id = ?
                        """,
                        (chunk_id,),
                    )
                    await self.conn.commit()
                    return new_relationships_added
                except Exception:
                    await self.conn.rollback()
                    raise

    async def ingest_custom_relationships_from_ingested_chunks(
        self,
        custom_relationships_json,
        async_cooldown=60,
        requests_per_minute=1000,
        verbose=True,
    ):
        if not self.conn:
            raise RuntimeError("Database connection is not established")
        if not isinstance(custom_relationships_json, list):
            raise TypeError("custom_relationships_json must be a list")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")

        relationships_by_chunk = {}
        for relationship in custom_relationships_json:
            chunk_id = relationship["chunk_id"]
            if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
                raise TypeError(
                    "Custom relationship chunk_id values must be "
                    "SQLite-compatible integers; got "
                    f"{chunk_id!r} ({type(chunk_id).__name__})."
                )
            relationships_by_chunk.setdefault(
                chunk_id,
                [],
            ).append(relationship)

        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT chunk_id, content FROM chunk"
            )
            persisted_chunks = {
                row[0]: row[1] for row in await cursor.fetchall()
            }
            await cursor.close()
        persisted_chunk_ids = set(persisted_chunks)
        unmatched_chunk_ids = set(relationships_by_chunk) - persisted_chunk_ids
        if unmatched_chunk_ids:
            raise ValueError(
                "Custom relationships reference chunk IDs that are not present "
                f"in SQLite: {sorted(unmatched_chunk_ids)}. This usually "
                "indicates a generated chunk-ID type or hash mismatch."
            )
        hash_mismatches = {
            chunk_id: self.hash_func(persisted_chunks[chunk_id])
            for chunk_id in relationships_by_chunk
            if self.hash_func(persisted_chunks[chunk_id]) != chunk_id
        }
        if hash_mismatches:
            raise ValueError(
                "Custom relationship chunk IDs do not match the hashes "
                "recomputed from SQLite chunk content: "
                f"{hash_mismatches}."
            )

        max_tries = 4
        current_try = 0
        while current_try < max_tries:
            chunks = await self.get_chunks_waiting_for_relation_extraction()
            if not chunks:
                break
            current_try += 1

            if verbose:
                print(
                    f"Found {len(chunks)} chunks waiting for custom "
                    "relationship ingestion."
                )

            progress = tqdm_asyncio(
                total=None,
                desc="Adding Pre-Defined Relationships",
                bar_format=(
                    "{desc}: {n} relationships [{elapsed}, {rate_fmt}]"
                ),
            )

            async def ingest_chunk(chunk):
                chunk_content = chunk[0]
                chunk_id = self.hash_func(chunk_content)
                custom_relationships = relationships_by_chunk.get(chunk_id, [])
                if chunk_id in relationships_by_chunk and not custom_relationships:
                    raise RuntimeError(
                        f"Chunk '{chunk_id}' was expected to contain supplied "
                        "custom relationships, but none matched."
                    )
                relationships_by_requirements = {}
                completed_relationships = []

                for relationship in custom_relationships:
                    cursor = await self.conn.execute(
                        """
                        SELECT relation_description
                        FROM relationship
                        WHERE relation_id = ?
                        """,
                        (relationship["relation_id"],),
                    )
                    existing_relationship = await cursor.fetchone()
                    await cursor.close()

                    if (
                        existing_relationship is not None
                        and self.hash_func(existing_relationship[0])
                        == self.hash_func(
                            relationship["relation_description"]
                        )
                    ):
                        completed_relationships.append(dict(relationship))
                        continue

                    requirements = (
                        not relationship["_is_relationship_enriched"],
                        not relationship["does_relationship_have_keywords"],
                    )
                    if any(requirements):
                        relationships_by_requirements.setdefault(
                            requirements,
                            [],
                        ).append(relationship)
                    else:
                        completed_relationships.append(dict(relationship))

                entity_id_to_local_id, prompt_entities = (
                    await self.prepare_custom_relationship_entity_mapping(
                        custom_relationships
                    )
                )
                llm_token_totals = []
                for requirements, relationship_group in (
                    relationships_by_requirements.items()
                ):
                    enriched_relationships, llm_tokens = (
                        await self.enrich_custom_relationship_group(
                            relationships=relationship_group,
                            chunk_content=chunk_content,
                            requirements=requirements,
                            entity_id_to_local_id=entity_id_to_local_id,
                            prompt_entities=prompt_entities,
                        )
                    )
                    completed_relationships.extend(enriched_relationships)
                    llm_token_totals.append(llm_tokens)

                if len(completed_relationships) != len(custom_relationships):
                    raise ValueError(
                        f"Expected {len(custom_relationships)} relationships "
                        f"for chunk '{chunk_id}', but prepared "
                        f"{len(completed_relationships)}."
                    )

                return await self.save_custom_relationships_for_chunk(
                    chunk_id=chunk_id,
                    relationships=completed_relationships,
                    llm_token_totals=llm_token_totals,
                )

            for batch_start in range(0, len(chunks), requests_per_minute):
                batch = chunks[
                    batch_start:batch_start + requests_per_minute
                ]
                results = await asyncio.gather(
                    *(ingest_chunk(chunk) for chunk in batch),
                    return_exceptions=True,
                )
                for chunk, result in zip(batch, results):
                    if isinstance(result, Exception):
                        await self.log_failure(
                            "relation",
                            "chunk",
                            self.hash_func(chunk[0]),
                            result,
                        )
                for result in results:
                    if not isinstance(result, Exception):
                        progress.update(result)

                if verbose:
                    batch_number = batch_start // requests_per_minute + 1
                    print(
                        f"Finished batch {batch_number}: "
                        f"{len(batch)} chunks processed"
                    )
                    for index, result in enumerate(results):
                        if isinstance(result, Exception):
                            print(
                                f"Chunk {batch_start + index} failed: {result}"
                            )

                if batch_start + requests_per_minute < len(chunks):
                    if verbose:
                        print(f"Cooling down for {async_cooldown} seconds...")
                    await asyncio.sleep(async_cooldown)

            progress.close()
            if current_try < max_tries:
                remaining_chunks = (
                    await self.get_chunks_waiting_for_relation_extraction()
                )
                if remaining_chunks:
                    if verbose:
                        print(
                            f"Retrying {len(remaining_chunks)} incomplete "
                            f"chunks after {async_cooldown} seconds..."
                        )
                    await asyncio.sleep(async_cooldown)

        remaining_chunks = await self.get_chunks_waiting_for_relation_extraction()
        if remaining_chunks:
            remaining_ids = [self.hash_func(chunk[0]) for chunk in remaining_chunks]
            raise RuntimeError(
                "Custom relationship ingestion did not complete for chunk IDs "
                f"{remaining_ids}; their extraction flags were not marked complete."
            )
    
 
