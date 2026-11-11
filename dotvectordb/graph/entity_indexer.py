import asyncio
import json
import os
import traceback
from tqdm.asyncio import tqdm_asyncio
from dotvectordb.construction_prompts.entity_prompts import (
    ENTITY_RECOGNITION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
    EntitySummarizer,
    ExtractedEntitiesResponse,
    INPUT_EXAMPLE,
    OUTPUT_EXAMPLE,
    prepare_entity_enrichment_prompt_sections,
)
class EntityIndexer:
    def __init__(
        self,
        conn,
        lock,
        language_model,
        embedding_model,
        hash_func,
        entity_groups,
        add_embedding_func,
        max_entity_relation_description_token_size=None,
        recommended_entity_relation_description_token_size=None,
        summary_language=None,
        entity_extract_prompt=None,
        entity_input_example=None,
        entity_output_example=None,
    ):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.hash_func = hash_func
        self.entity_groups = entity_groups
        self.add_embedding_func = add_embedding_func
        self.entity_extract_prompt = ENTITY_RECOGNITION_PROMPT if entity_extract_prompt is None else entity_extract_prompt
        self.entity_input_example = INPUT_EXAMPLE if entity_input_example is None else entity_input_example
        self.entity_output_example = OUTPUT_EXAMPLE if entity_output_example is None else entity_output_example
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
    ############################################################
    #   Token Economics Tracking                               #
    ############################################################
    async def save_input_and_output_tokens_for_llm(self, total_tokens):
        await self.conn.execute(
            """
            INSERT INTO llm_token_usage (stage, total_tokens)
            VALUES (?, ?)
            """,
            ("entity", total_tokens),
        )
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
    ############################################################
    #   Fetch Entity-Related Data For Entity Indexing          #
    ############################################################
    async def get_chunks_waiting_for_entity_extraction(self):
        cursor = await self.conn.execute(
            """
            SELECT content
            FROM chunk
            WHERE is_embedded = 1 AND are_entities_extracted = 0
            """
        )
        chunks = await cursor.fetchall()
        await cursor.close()
        return chunks

    async def get_entities_waiting_for_embedding(self):
        cursor = await self.conn.execute(
            "SELECT * FROM entity WHERE is_embedded = 0"
        )
        rows = await cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        await cursor.close()

        return [dict(zip(column_names, row)) for row in rows]

    async def get_entities_requiring_summarization(self):
        cursor = await self.conn.execute(
            "SELECT * FROM entity WHERE is_summarized = FALSE"
        )
        rows = await cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        await cursor.close()

        return [dict(zip(column_names, row)) for row in rows]

    async def get_entities_that_need_updated_dedupe(self):
        """Return tracked updated entity IDs and their types."""
        async with self.conn.execute(
            """
            SELECT tracked.entity_id, mapping.entity_type
            FROM new_and_updated_entity AS tracked
            LEFT JOIN entity_to_entity_type_mapping AS mapping
                ON mapping.entity_id = tracked.entity_id
            WHERE tracked.is_new = FALSE
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return rows

    async def get_new_entities_before_last_completed_version(self):
        """Return tracked new entity IDs and their types."""
        async with self.conn.execute(
            """
            SELECT tracked.entity_id, mapping.entity_type
            FROM new_and_updated_entity AS tracked
            LEFT JOIN entity_to_entity_type_mapping AS mapping
                ON mapping.entity_id = tracked.entity_id
            WHERE tracked.is_new = TRUE
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return rows

    async def get_and_save_top_k_similar_entities(self, entity_id, k=10):
        """Return {entity_id: similarity_score} for the nearest other entities."""
        async with self.conn.execute(
            "SELECT embedding FROM entity_embeddings WHERE rowid = ?",
            (entity_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return {}

        async with self.conn.execute(
            """
            SELECT
                v.rowid,
                1.0 / (
                    1.0 + vec1_l2_distance(:query_embedding, v.embedding)
                ) AS similarity_score
            FROM entity_embeddings(:query_embedding, :search_parameters) AS v
            JOIN entity AS m ON m.rowid = v.rowid
            WHERE v.rowid <> :entity_id
            ORDER BY similarity_score DESC
            LIMIT :top_k
            """,
            {
                "query_embedding": row[0],
                "search_parameters": json.dumps({"k": k + 1}),
                "entity_id": entity_id,
                "top_k": k,
            },
        ) as cursor:
            rows = await cursor.fetchall()
        weight_dict = {row[0]: row[1] for row in rows}
        await self.save_entity_weights(entity_id, weight_dict)
        return weight_dict
    async def save_entity_weights(self, entity_id, weight_dict):
        """Commit similarity scores and the entity dedupe checkpoint together."""
        async with self.lock:
            try:
                await self.conn.executemany(
                    """
                    INSERT INTO entity_to_entity_weights (
                        entity_id, similar_entity_id, similarity_score
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(entity_id, similar_entity_id) DO UPDATE SET
                        similarity_score = excluded.similarity_score
                    """,
                    [
                        (min(entity_id, similar_entity_id),
                         max(entity_id, similar_entity_id), score)
                        for similar_entity_id, score in weight_dict.items()
                    ],
                )
                await self.conn.execute(
                    """
                    UPDATE entity
                    SET last_dedupe_update = (
                        SELECT current_version_count FROM global_config WHERE id = 1
                    )
                    WHERE entity_id = ?
                    """,
                    (entity_id,),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
    ##########################################################################
    #   Handles all preparation needed to prepare the entity prompt          #
    ##########################################################################
    def prepare_entity_group_mapping(self):
        group_to_number_map = {}
        number_to_group_map = {}

        for number, (group, description) in enumerate(
            self.entity_groups.items(), start=1
        ):
            group_to_number_map[group] = number
            number_to_group_map[number] = group

        header = f"""
        There are {len(self.entity_groups)} possible entity groups. An entity
        can belong to one or more groups. The groups are for routing and must
        not restrict which entities are extracted. Do not invent new groups.
        """

        body = ""
        for group, description in self.entity_groups.items():
            number = group_to_number_map[group]
            body += f"{number}: {group} - {description}\n"

        return header + body, group_to_number_map, number_to_group_map

    async def get_entity_prompt_for_embedding(self, entity):
        async with self.lock:
            type_cursor = await self.conn.execute(
                """
                SELECT entity_type
                FROM entity_to_entity_type_mapping
                WHERE entity_id = ?
                """,
                (entity["entity_id"],),
            )
            type_rows = await type_cursor.fetchall()
            await type_cursor.close()

        formatted_entity = {
            "entity_name": entity["entity_name"],
            "entity_description": entity["entity_description"],
            "entity_groups": [row[0] for row in type_rows],
        }
        if entity.get("alias") is not None:
            formatted_entity["alias"] = entity["alias"]

        return formatted_entity

    ##########################################################################
    #       Entity Extraction & Ingestion Operations for Indexing            #
    ##########################################################################
    async def extract_entity(
        self,
        entity_groups_prompt,
        number_to_group_map,
        chunk,
    ):
        if not self.conn:
            raise RuntimeError("Database not opened")

        try:
            chunk_hash = self.hash_func(chunk)
            embedding_model_hash = self.hash_func(
                self.embedding_model.model_name
            )
            embedding_model_hash_str = f"{embedding_model_hash}"
            prompt = self.entity_extract_prompt.format(
                entity_types=entity_groups_prompt,
                chunk=chunk,
                input_example=self.entity_input_example,
                output_example=self.entity_output_example,
                language=self.summary_language,
            )
        except Exception as error:
            await self.log_failure(
                "entity",
                "chunk",
                locals().get("chunk_hash", "unknown"),
                error,
            )
            traceback.print_exc()
            raise RuntimeError(
                f"Error preparing entity recognition prompt: {error}"
            )

        try:
            response = await self.language_model(
                prompt,
                structuredInput=ExtractedEntitiesResponse,
            )
        except Exception as error:
            await self.log_failure(
                "entity",
                "chunk",
                chunk_hash,
                error,
            )
            traceback.print_exc()
            print(f"Error extracting entities: {error}")
            return 0

        async with self.lock:
            try:
                new_entities_added = 0
                seen_entity_ids = set()

                for entity in response[0]["entities"]:
                    entity_name = entity["entity_name"].strip().lower()
                    entity_id = self.hash_func(entity_name)

                    if entity_id in seen_entity_ids:
                        continue
                    seen_entity_ids.add(entity_id)

                    entity_description = entity["entity_description"]
                    entity_groups = sorted(
                        {
                            number_to_group_map.get(group, group)
                            for group in entity["entity_groups"]
                        }
                    )
                    entity_str = json.dumps(entity, indent=2)
                    tokens = len(
                        self.embedding_model.tokenize(entity_str)
                    )
                    is_summarized = (
                        len(
                            self.embedding_model.tokenize(
                                entity_description
                            )
                        )
                        <= self.max_entity_relation_description_token_size
                    )

                    dupe_cursor = await self.conn.execute(
                        """
                        SELECT entity_description, tokens
                        FROM entity
                        WHERE entity_id = ?
                        """,
                        (entity_id,),
                    )
                    existing_entity = await dupe_cursor.fetchone()
                    await dupe_cursor.close()

                    if not existing_entity:
                        await self.conn.execute(
                            """
                            INSERT INTO entity (
                                entity_id,
                                entity_name,
                                entity_description,
                                embedding_model_id,
                                tokens,
                                is_summarized
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entity_id,
                                entity_name,
                                entity_description,
                                embedding_model_hash_str,
                                tokens,
                                is_summarized,
                            ),
                        )
                        new_entities_added += 1
                    else:
                        old_description = existing_entity[0]
                        old_tokens = existing_entity[1]

                        old_description_hash = self.hash_func(
                            old_description
                        )
                        new_description_hash = self.hash_func(
                            entity_description
                        )

                        if (
                            old_description_hash
                            != new_description_hash
                        ):
                            combined_description = (
                                f"{old_description}<SEP>"
                                f"{entity_description}"
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
                                UPDATE entity
                                SET entity_description = ?,
                                    tokens = ?,
                                    is_summarized = ?,
                                    is_embedded = FALSE
                                WHERE entity_id = ?
                                """,
                                (
                                    combined_description,
                                    old_tokens + tokens,
                                    is_summarized,
                                    entity_id,
                                ),
                            )

                    for group in entity_groups:
                        await self.conn.execute(
                            """
                            INSERT OR IGNORE INTO
                                entity_to_entity_type_mapping
                                (entity_id, entity_type)
                            VALUES (?, ?)
                            """,
                            (entity_id, group),
                        )

                    await self.conn.execute(
                        """
                        INSERT OR IGNORE INTO
                            chunk_to_entity_mapping
                            (chunk_id, entity_id)
                        VALUES (?, ?)
                        """,
                        (chunk_hash, entity_id),
                    )

                await self.conn.execute(
                    """
                    UPDATE chunk
                    SET are_entities_extracted = 1
                    WHERE chunk_id = ?
                    """,
                    (chunk_hash,),
                )
                await self.save_input_and_output_tokens_for_llm(
                    response[1]
                )
                await self.conn.commit()
                return new_entities_added
            except Exception as error:
                await self.conn.rollback()
                await self.log_failure(
                    "entity",
                    "chunk",
                    chunk_hash,
                    error,
                    lock_held=True,
                )
                traceback.print_exc()
                print(
                    "Error saving extracted entities to database: "
                    f"{error}"
                )
                return 0

    async def extract_entities_from_ingested_chunks(
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

        max_tries = 4
        current_try = 0

        while current_try < max_tries:
            chunks = await self.get_chunks_waiting_for_entity_extraction()
            if not chunks:
                break
            current_try += 1

            if verbose:
                print(f"Found {len(chunks)} chunks waiting for entity extraction.")

            group_prompt, _, number_to_group_map = (
                self.prepare_entity_group_mapping()
            )
            progress = tqdm_asyncio(
                total=None,
                desc="Extracting Entities",
                bar_format="{desc}: {n} entities [{elapsed}, {rate_fmt}]",
            )

            for batch_start in range(0, len(chunks), requests_per_minute):
                batch = chunks[batch_start:batch_start + requests_per_minute]
                results = await asyncio.gather(
                    *(
                        self.extract_entity(
                            entity_groups_prompt=group_prompt,
                            number_to_group_map=number_to_group_map,
                            chunk=chunk[0],
                        )
                        for chunk in batch
                    ),
                    return_exceptions=True,
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
                            print(f"Chunk {batch_start + index} failed: {result}")

                if batch_start + requests_per_minute < len(chunks):
                    if verbose:
                        print(f"Cooling down for {async_cooldown} seconds...")
                    await asyncio.sleep(async_cooldown)

        if current_try == max_tries:
            remaining_chunks = await self.get_chunks_waiting_for_entity_extraction()
            if remaining_chunks:
                raise RuntimeError(
                    f"Entity extraction failed after {max_tries} attempts: "
                    f"{len(remaining_chunks)} chunks still require extraction."
                )

    async def fix_entities(
        self,
        async_cooldown=60,
        requests_per_minute=1000,
        verbose=True,
    ):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")

        entities = await self.get_entities_requiring_summarization()

        async def summarize_entity(entity):
            try:
                descriptions = [
                    description.strip()
                    for description in entity["entity_description"].split("<SEP>")
                    if description.strip()
                ]
                description_list = "\n".join(
                    json.dumps(
                        {"Description": description},
                        ensure_ascii=False,
                    )
                    for description in descriptions
                )
                prompt = ENTITY_SUMMARIZATION_PROMPT.format(
                    summary_length=(
                        self.recommended_entity_relation_description_token_size
                    ),
                    description_type="Entity",
                    description_name=entity["entity_name"],
                    description_list=description_list,
                    language=self.summary_language,
                )
                response = await self.language_model(
                    prompt,
                    structuredInput=EntitySummarizer,
                )
                new_description = response[0]["new_description"].strip()
                if not new_description:
                    raise ValueError(
                        f"Entity summarizer returned an empty description for "
                        f"'{entity['entity_name']}'."
                    )

                description_tokens = len(
                    self.embedding_model.tokenize(new_description)
                )
                if (
                    description_tokens
                    > self.max_entity_relation_description_token_size
                ):
                    raise ValueError(
                        f"Summary for entity '{entity['entity_name']}' contains "
                        f"{description_tokens} tokens; the maximum is "
                        f"{self.max_entity_relation_description_token_size}."
                    )

                async with self.lock:
                    await self.conn.execute(
                        """
                        UPDATE entity
                        SET entity_description = ?, is_summarized = TRUE, is_embedded = FALSE
                        WHERE entity_id = ?
                        """,
                        (new_description, entity["entity_id"]),
                    )
                    await self.save_input_and_output_tokens_for_llm(response[1])
                    await self.conn.commit()
                return None
            except Exception as error:
                await self.log_failure(
                    "desc",
                    "entity",
                    entity.get("entity_id", "unknown"),
                    error,
                )
                if verbose:
                    print(
                        f"Summarization failed for entity "
                        f"'{entity.get('entity_name')}': {error}"
                    )
                traceback.print_exc()
                return entity

        failed_entities = entities.copy()
        while failed_entities:
            current_batch = failed_entities[:requests_per_minute]
            remaining_entities = failed_entities[requests_per_minute:]
            batch_results = await tqdm_asyncio.gather(
                *(summarize_entity(entity) for entity in current_batch),
                desc="Summarizing entities",
                total=len(current_batch),
            )
            failed_entities = [
                entity for entity in batch_results if entity is not None
            ]
            failed_entities.extend(remaining_entities)

            if failed_entities:
                if verbose:
                    print(
                        f"Cooldown for {async_cooldown} seconds. "
                        f"{len(failed_entities)} entities remaining."
                    )
                await asyncio.sleep(async_cooldown)

    async def deduplicate_entities(self, entities, async_cooldown=60,requests_per_minute=1000,verbose=True):    
        if not self.conn:
            raise RuntimeError("Database connection is not established")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
       
        async def get_entity_weights(entity_id):
            try:
                await self.get_and_save_top_k_similar_entities(entity_id, k=40)
                return None
            except Exception as error:
                await self.log_failure("entity", "entity", entity_id, error)
                if verbose:
                    print(f"Weight lookup failed for entity {entity_id}, retrying next batch: {error}")
                traceback.print_exc()
                return entity_id

        failed_entities = entities.copy()
        while failed_entities:
            current_batch = failed_entities[:requests_per_minute]
            remaining_entities = failed_entities[requests_per_minute:]
            batch_results = await tqdm_asyncio.gather(
                *(get_entity_weights(entity_id) for entity_id in current_batch),
                desc="Saving entity weights",
                total=len(current_batch),
                disable=not verbose,
            )
            failed_entities = [
                entity_id for entity_id in batch_results if entity_id is not None
            ]
            failed_entities.extend(remaining_entities)

            if failed_entities:
                if verbose:
                    print(
                        f"Cooldown for {async_cooldown} seconds. "
                        f"{len(failed_entities)} entities remaining."
                    )
                await asyncio.sleep(async_cooldown)

    async def embed_entities(
        self,
        new_entities,
        updated_entities,
        async_cooldown=60,
        requests_per_minute=1000,
        verbose=True,
    ):
        new_entity_set = {entity_id for entity_id, _ in new_entities}
        updated_entity_set = {entity_id for entity_id, _ in updated_entities}
        entities = await self.get_entities_waiting_for_embedding()
        async def embed_entity(entity):
            try:
                entity_id = entity["entity_id"]
                if entity_id in new_entity_set:
                    status = "new"
                elif entity_id in updated_entity_set:
                    status = "updated"
                else:
                    raise ValueError(f"Entity {entity_id} is not tracked as new or updated")
                content = await self.get_entity_prompt_for_embedding(entity)
                if verbose:
                    print(f"Prompt for entity {entity['entity_name']}: {content}")
                await self.add_embedding_func(
                    content=json.dumps(content, indent=4),
                    table="entity",
                    pk_hash=entity_id,
                    status=status,
                )
                return None
            except Exception as error:
                await self.log_failure(
                    "entity",
                    "entity",
                    entity.get("entity_id", "unknown"),
                    error,
                )
                if verbose:
                    print(f"Embedding failed, retrying next batch: {error}")
                traceback.print_exc()
                return entity

        failed_entities = entities.copy()
        while failed_entities:
            current_batch = failed_entities[:requests_per_minute]
            remaining_entities = failed_entities[requests_per_minute:]
            batch_results = await tqdm_asyncio.gather(
                *(embed_entity(entity) for entity in current_batch),
                desc="Embedding entities",
                total=len(current_batch),
            )
            failed_entities = [
                entity for entity in batch_results if entity is not None
            ]
            failed_entities.extend(remaining_entities)

            if failed_entities:
                if verbose:
                    print(
                        f"Cooldown for {async_cooldown} seconds. "
                        f"{len(failed_entities)} entities remaining."
                    )
                await asyncio.sleep(async_cooldown)
    ##########################################################################
    #                   Custom Entities Provided Handler                     #
    ##########################################################################
    async def enrich_custom_entity_group(
        self,
        entities,
        chunk_content,
        requirements,
    ):
        """Enrich entities from one chunk that share the same requirements."""
        needs_descriptions, needs_types = requirements
        group_prompt, _, number_to_group_map = self.prepare_entity_group_mapping()
        prompt_template, enrichment_model = (
            await prepare_entity_enrichment_prompt_sections(
                needs_descriptions=needs_descriptions,
                needs_types=needs_types,
            )
        )

        local_id_to_entity = {
            local_entity_id: entity
            for local_entity_id, entity in enumerate(entities, start=1)
        }
        prompt_entities = [
            {
                "id": local_entity_id,
                "entity_name": entity["entity_name"],
            }
            for local_entity_id, entity in local_id_to_entity.items()
        ]
        prompt = prompt_template.format(
            language=self.summary_language,
            entity_types=group_prompt,
            chunk=chunk_content,
            entities=json.dumps(
                prompt_entities,
                ensure_ascii=False,
                indent=2,
            ),
            input_example=INPUT_EXAMPLE,
            output_example=OUTPUT_EXAMPLE,
        )
        response = await self.language_model(
            prompt,
            structuredInput=enrichment_model,
        )
        returned_entities = response[0]["entities"]
        returned_local_ids = [
            returned_entity["local_entity_id"]
            for returned_entity in returned_entities
        ]
        expected_local_ids = set(local_id_to_entity)

        if (
            len(returned_entities) != len(local_id_to_entity)
            or len(returned_local_ids) != len(set(returned_local_ids))
            or set(returned_local_ids) != expected_local_ids
        ):
            raise ValueError(
                "Entity enrichment response IDs did not match the requested "
                f"local IDs. Expected {sorted(expected_local_ids)}, received "
                f"{returned_local_ids}."
            )

        enriched_entities = []
        for returned_entity in returned_entities:
            local_entity_id = returned_entity["local_entity_id"]
            enriched_entity = dict(local_id_to_entity[local_entity_id])

            if needs_descriptions:
                description = returned_entity["entity_description"].strip()
                if not description:
                    raise ValueError(
                        f"Generated description for "
                        f"'{enriched_entity['entity_name']}' cannot be empty."
                    )
                enriched_entity["entity_description"] = description
            if needs_types:
                enriched_entity["entity_types"] = sorted(
                    {
                        number_to_group_map.get(group, group)
                        for group in returned_entity["entity_groups"]
                    }
                )

            enriched_entities.append(enriched_entity)

        return enriched_entities, response[1]

    async def save_custom_entities_for_chunk(
            self,
            chunk_id,
            entities,
            llm_token_totals,
        ):
            """Atomically save every custom entity associated with one chunk."""
            embedding_model_hash = str(
                self.hash_func(self.embedding_model.model_name)
            )
            new_entities_added = 0

            async with self.lock:
                try:
                    for entity in entities:
                        mapping_cursor = await self.conn.execute(
                            """
                            SELECT 1
                            FROM chunk_to_entity_mapping
                            WHERE chunk_id = ? AND entity_id = ?
                            """,
                            (chunk_id, entity["entity_id"]),
                        )
                        mapping_exists = await mapping_cursor.fetchone()
                        await mapping_cursor.close()
                        if mapping_exists is not None:
                            continue

                        entity_record = {
                            "entity_name": entity["entity_name"],
                            "entity_description": entity["entity_description"],
                            "entity_groups": entity["entity_types"],
                        }
                        tokens = len(
                            self.embedding_model.tokenize(
                                json.dumps(entity_record, ensure_ascii=False)
                            )
                        )
                        cursor = await self.conn.execute(
                            """
                            SELECT entity_description, tokens
                            FROM entity
                            WHERE entity_id = ?
                            """,
                            (entity["entity_id"],),
                        )
                        existing_entity = await cursor.fetchone()
                        await cursor.close()

                        if existing_entity is None:
                            is_summarized = (
                                len(
                                    self.embedding_model.tokenize(
                                        entity["entity_description"]
                                    )
                                )
                                <= self.max_entity_relation_description_token_size
                            )
                            await self.conn.execute(
                                """
                                INSERT INTO entity (
                                    entity_id,
                                    entity_name,
                                    entity_description,
                                    embedding_model_id,
                                    tokens,
                                    is_summarized
                                ) VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    entity["entity_id"],
                                    entity["entity_name"],
                                    entity["entity_description"],
                                    embedding_model_hash,
                                    tokens,
                                    is_summarized,
                                ),
                            )
                            new_entities_added += 1
                        else:
                            old_description, old_tokens = (
                                existing_entity
                            )
                            old_description_hash = self.hash_func(old_description)
                            new_description_hash = self.hash_func(
                                entity["entity_description"]
                            )

                            if old_description_hash != new_description_hash:
                                combined_description = (
                                    f"{old_description}<SEP>"
                                    f"{entity['entity_description']}"
                                )
                                is_summarized = (
                                    len(
                                        self.embedding_model.tokenize(
                                            combined_description
                                        )
                                    )
                                    <= self.max_entity_relation_description_token_size
                                )
                                # We need to set is_embedded to FALSE because the entity description has changed, and we need to re-embed it.
                                await self.conn.execute(
                                    """
                                    UPDATE entity
                                    SET entity_description = ?,
                                        tokens = ?,
                                        is_summarized = ?,
                                        is_embedded = FALSE
                                    WHERE entity_id = ?
                                    """,
                                    (
                                        combined_description,
                                        old_tokens + tokens,
                                        is_summarized,
                                        entity["entity_id"],
                                    ),
                                )

                        for entity_type in entity["entity_types"]:
                            await self.conn.execute(
                                """
                                INSERT OR IGNORE INTO entity_to_entity_type_mapping
                                (entity_id, entity_type) VALUES (?, ?)
                                """,
                                (entity["entity_id"], entity_type),
                            )
                        await self.conn.execute(
                            """
                            INSERT OR IGNORE INTO chunk_to_entity_mapping
                            (chunk_id, entity_id) VALUES (?, ?)
                            """,
                            (chunk_id, entity["entity_id"]),
                        )

                    for llm_tokens in llm_token_totals:
                        await self.save_input_and_output_tokens_for_llm(llm_tokens)
                    await self.conn.execute(
                        """
                        UPDATE chunk
                        SET are_entities_extracted = 1
                        WHERE chunk_id = ?
                        """,
                        (chunk_id,),
                    )
                    await self.conn.commit()
                    return new_entities_added
                except Exception:
                    await self.conn.rollback()
                    raise

    async def ingest_custom_entities_from_ingested_chunks(
        self,
        custom_entities_json,
        async_cooldown=60,
        requests_per_minute=1000,
        verbose=True,
    ):
        if not self.conn:
            raise RuntimeError("Database connection is not established")
        if not isinstance(custom_entities_json, list):
            raise TypeError("custom_entities_json must be a list of entities")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")

        entities_by_chunk = {}
        for entity in custom_entities_json:
            chunk_id = entity["chunk_id"]
            if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
                raise TypeError(
                    "Custom entity chunk_id values must be SQLite-compatible "
                    f"integers; got {chunk_id!r} ({type(chunk_id).__name__})."
                )
            entities_by_chunk.setdefault(chunk_id, []).append(entity)

        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT chunk_id, content FROM chunk"
            )
            persisted_chunks = {
                row[0]: row[1] for row in await cursor.fetchall()
            }
            await cursor.close()
        persisted_chunk_ids = set(persisted_chunks)
        unmatched_chunk_ids = set(entities_by_chunk) - persisted_chunk_ids
        if unmatched_chunk_ids:
            raise ValueError(
                "Custom entities reference chunk IDs that are not present in "
                f"SQLite: {sorted(unmatched_chunk_ids)}. This usually indicates "
                "a generated chunk-ID type or hash mismatch."
            )
        hash_mismatches = {
            chunk_id: self.hash_func(persisted_chunks[chunk_id])
            for chunk_id in entities_by_chunk
            if self.hash_func(persisted_chunks[chunk_id]) != chunk_id
        }
        if hash_mismatches:
            raise ValueError(
                "Custom entity chunk IDs do not match the hashes recomputed "
                f"from SQLite chunk content: {hash_mismatches}."
            )

        max_tries = 4
        current_try = 0
        while current_try < max_tries:
            chunks = await self.get_chunks_waiting_for_entity_extraction()
            if not chunks:
                break
            current_try += 1

            if verbose:
                print(f"Found {len(chunks)} chunks waiting for entity extraction.")

            progress = tqdm_asyncio(
                total=None,
                desc="Adding Pre-Defined Entities",
                bar_format="{desc}: {n} entities [{elapsed}, {rate_fmt}]",
            )

            async def ingest_chunk(chunk):
                chunk_content = chunk[0]
                chunk_id = self.hash_func(chunk_content)
                custom_entities = entities_by_chunk.get(chunk_id, [])
                if chunk_id in entities_by_chunk and not custom_entities:
                    raise RuntimeError(
                        f"Chunk '{chunk_id}' was expected to contain supplied "
                        "custom entities, but none matched."
                    )
                entities_by_requirements = {}
                completed_entities = []

                for entity in custom_entities:
                    cursor = await self.conn.execute(
                        """
                        SELECT entity_description
                        FROM entity
                        WHERE entity_id = ?
                        """,
                        (entity["entity_id"],),
                    )
                    existing_entity = await cursor.fetchone()
                    await cursor.close()

                    if (
                        existing_entity is not None
                        and self.hash_func(existing_entity[0])
                        == self.hash_func(entity["entity_description"])
                    ):
                        completed_entities.append(dict(entity))
                        continue

                    requirements = (
                        not entity["_is_entity_enriched"],
                        not entity["does_entity_have_types"],
                    )
                    if any(requirements):
                        entities_by_requirements.setdefault(
                            requirements,
                            [],
                        ).append(entity)
                    else:
                        completed_entities.append(dict(entity))

                llm_token_totals = []
                for requirements, entity_group in (
                    entities_by_requirements.items()
                ):
                    enriched_entities, llm_tokens = (
                        await self.enrich_custom_entity_group(
                            entities=entity_group,
                            chunk_content=chunk_content,
                            requirements=requirements,
                        )
                    )
                    completed_entities.extend(enriched_entities)
                    llm_token_totals.append(llm_tokens)

                if len(completed_entities) != len(custom_entities):
                    raise ValueError(
                        f"Expected {len(custom_entities)} entities for chunk "
                        f"'{chunk_id}', but prepared {len(completed_entities)}."
                    )

                return await self.save_custom_entities_for_chunk(
                    chunk_id=chunk_id,
                    entities=completed_entities,
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
                            "entity",
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
                    await self.get_chunks_waiting_for_entity_extraction()
                )
                if remaining_chunks:
                    if verbose:
                        print(
                            f"Retrying {len(remaining_chunks)} incomplete "
                            f"chunks after {async_cooldown} seconds..."
                        )
                    await asyncio.sleep(async_cooldown)

        remaining_chunks = await self.get_chunks_waiting_for_entity_extraction()
        if remaining_chunks:
            remaining_ids = [self.hash_func(chunk[0]) for chunk in remaining_chunks]
            raise RuntimeError(
                "Custom entity ingestion did not complete for chunk IDs "
                f"{remaining_ids}; their extraction flags were not marked complete."
            )
