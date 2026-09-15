import asyncio
import json
import traceback
import numpy as np
from model2vec import StaticModel
from tqdm.asyncio import tqdm_asyncio
from dotvectordb.prompts.entity_prompts import (
    ENTITY_RECOGNITION_PROMPT,
    ExtractedEntitiesResponse,
    INPUT_EXAMPLE,
    OUTPUT_EXAMPLE,
    
)


class EntityIndexer3:
    def __init__(
        self,
        conn,
        lock,
        language_model,
        embedding_model,
        hash_func,
        entity_groups,
        add_embedding_func,
    ):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.hash_func = hash_func
        self.entity_groups = entity_groups
        self.add_embedding_func = add_embedding_func
        self.small_llm = StaticModel.from_pretrained("minishlab/potion-base-32M")

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
    async def remove_extra_questions_from_entity(self, questions, threshold=0.8):
            """Deduplicate queries based on cosine similarity of their embeddings."""
            embeddings = self.small_llm.encode(questions)
            embeddings /= np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True
            )
            sorted_indices = sorted(
                range(len(questions)),
                key=lambda i: len(questions[i])
            )
            kept_indices = []
            for idx in sorted_indices:
                if all(
                    np.dot(embeddings[idx], embeddings[j]) < threshold
                    for j in kept_indices
                ):
                    kept_indices.append(idx)
            kept_indices.sort()
            return [questions[i] for i in kept_indices]


    ############################################################
    #   Fetch Entity-Related Data For Entity Indexing          #
    ############################################################
    async def get_chunks_waiting_for_entity_extraction(self):
        async with self.lock:
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
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM entity WHERE is_embedded = 0"
            )
            rows = await cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            await cursor.close()

        return [dict(zip(column_names, row)) for row in rows]

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
            "questions": entity["questions"],
        }
        if entity.get("alias") is not None:
            formatted_entity["alias"] = entity["alias"]

        return formatted_entity

    ##########################################################################
    #       Entity Extraction & Ingestion Operations for Indexing            #
    ##########################################################################
    async def extract_entity(self, entity_groups_prompt, number_to_group_map, chunk):
        if not self.conn:
            raise RuntimeError("Database not opened")

        try:
            chunk_hash = self.hash_func(chunk)
            embedding_model_hash = self.hash_func(self.embedding_model.model_name)
            embedding_model_hash_str = f"{embedding_model_hash}"
            prompt = ENTITY_RECOGNITION_PROMPT.format(
                entity_types=entity_groups_prompt,
                chunk=chunk,
                input_example=INPUT_EXAMPLE,
                output_example=OUTPUT_EXAMPLE,
            )
        except Exception as error:
            traceback.print_exc()
            raise RuntimeError(f"Error preparing entity recognition prompt: {error}")

        try:
            response = await self.language_model(
                prompt,
                structuredInput=ExtractedEntitiesResponse,
            )
        except Exception as error:
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
                    tokens = len(self.embedding_model.tokenize(entity_str))
                    questions = entity["entity_questions"]

                    dupe_cursor = await self.conn.execute(
                        """
                        SELECT entity_description, questions, tokens
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
                                questions,
                                embedding_model_id,
                                tokens
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entity_id,
                                entity_name,
                                entity_description,
                                json.dumps(questions),
                                embedding_model_hash_str,
                                tokens,
                            ),
                        )
                        new_entities_added += 1
                    else:
                        old_description = existing_entity[0]
                        old_questions = json.loads(existing_entity[1])
                        old_tokens = existing_entity[2]
                        await self.conn.execute(
                            """
                            UPDATE entity
                            SET entity_description = ?, questions = ?, tokens = ?
                            WHERE entity_id = ?
                            """,
                            (
                                old_description + "<SEP>" + entity_description,
                                json.dumps(old_questions + questions),
                                old_tokens + tokens,
                                entity_id,
                            ),
                        )

                    for group in entity_groups:
                        await self.conn.execute(
                            """
                            INSERT OR IGNORE INTO entity_to_entity_type_mapping
                            (entity_id, entity_type) VALUES (?, ?)
                            """,
                            (entity_id, group),
                        )

                    await self.conn.execute(
                        """
                        INSERT OR IGNORE INTO chunk_to_entity_mapping
                        (chunk_id, entity_id) VALUES (?, ?)
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
                await self.save_input_and_output_tokens_for_llm(response[1])
                await self.conn.commit()
                return new_entities_added
            except Exception as error:
                await self.conn.rollback()
                traceback.print_exc()
                print(f"Error saving extracted entities to database: {error}")
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

    async def embed_entities(
        self,
        async_cooldown=60,
        requests_per_minute=1000,
        verbose=True,
    ):

        entities = await self.get_entities_waiting_for_embedding()

        async def embed_entity(entity):
            try:
                content = await self.get_entity_prompt_for_embedding(entity)
                questions = content["questions"]
                if isinstance(questions, str):
                    questions = json.loads(questions)
                content["questions"] = await self.remove_extra_questions_from_entity(questions)
                if verbose:
                    print(f"Prompt for entity {entity['entity_name']}: {content}")
                await self.add_embedding_func(
                    content=json.dumps(content, indent=4),
                    table="entity",
                    pk_hash=entity["entity_id"],
                )
                return None
            except Exception as error:
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
