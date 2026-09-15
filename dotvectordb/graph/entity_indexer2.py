from dotvectordb.prompts.entity_prompts import ENTITY_RECOGNITION_PROMPT, ExtractedEntitiesResponse, INPUT_EXAMPLE, OUTPUT_EXAMPLE
import json
import traceback
import asyncio
from tqdm.asyncio import tqdm_asyncio


class EntityIndexer2:
    def __init__(self, conn, lock, language_model, embedding_model, hash_func, entity_types, add_embedding_func):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.hash_func = hash_func
        self.add_embedding_func = add_embedding_func
        self.entity_types = entity_types

    ############################################################
    #   Token Economics Tracking                               #
    ############################################################
    async def save_input_and_output_tokens_for_llm(self, total_tokens):
            await self.conn.execute(
                """
                INSERT INTO llm_token_usage (stage, total_tokens)
                VALUES (?, ?)
                """,
                ("entity", total_tokens)
            )

    ############################################################
    #   Fetch Entity-Related Data For Entity Indexing          #
    ############################################################
    async def get_chunks_waiting_for_entity_extraction(self):
        """Gives us all the chunks ready for the entity extraction process"""
        async with self.lock:
            cursor = await self.conn.execute(
            """SELECT content FROM chunk WHERE are_relations_extracted = 1 AND are_entities_extracted = 0"""
        )
        chunks = await cursor.fetchall()
        await cursor.close()
        return chunks

    async def get_temp_entities_connected_to_chunk(self, chunk_id):
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT temp_entity.temp_entity_id, temp_entity.entity_name, temp_entity.entity_tags
                FROM chunk_to_temp_entity_mapping
                JOIN temp_entity
                ON temp_entity.temp_entity_id = chunk_to_temp_entity_mapping.temp_entity_id
                WHERE chunk_to_temp_entity_mapping.chunk_id = ?
                """,
                (chunk_id,)
            )
            rows = await cursor.fetchall()
            await cursor.close()

            entities = [
                {
                    "temp_entity_id": row[0],
                    "entity_name": row[1],
                    "entity_tags": json.loads(row[2]),
                }
                for row in rows
            ]
            return entities

    async def prepare_local_entity_mapping_for_chunk(self, temp_entities):
         """Given temporary entities, create mappings from a short number to the temporary entity ID and entity metadata"""
         number_to_temp_entity_id_map = {}
         entities_for_prompt = []
         start = 1

         for entity in temp_entities:
              number_to_temp_entity_id_map[start] = entity["temp_entity_id"]
              entities_for_prompt.append({
                  "id": start,
                  "entity_name": entity["entity_name"],
              })
              start += 1

         return number_to_temp_entity_id_map, entities_for_prompt

    async def get_entities_waiting_for_embedding(self):
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM entity WHERE is_embedded = 0"
            )
            rows = await cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            await cursor.close()

            entities = [
                dict(zip(column_names, row))
                for row in rows
            ]
            return entities

    ##########################################################################
    #   Handles all preparation needed to prepare the entity prompt          #
    ##########################################################################
    async def get_entity_prompt_for_embedding(self, entity):
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT entity_type FROM entity_to_entity_type_mapping WHERE entity_id = ?",
                (entity["entity_id"],)
            )
            rows = await cursor.fetchall()
            await cursor.close()

        formatted_entity = {
        "entity_name": entity["entity_name"],
        "routing_types": [row[0] for row in rows],
        "entity_description": entity["entity_description"],
        "questions": entity["questions"],
        }
        if entity.get("alias") is not None:
            formatted_entity["alias"] = entity["alias"]

        return formatted_entity

    def prepare_entity_type_mapping(self):
        type_to_number_map = {}
        number_to_type_map = {}
        number_of_keys = len(self.entity_types)
        start = 1

        # Creates a mapping from numbers to entity type keys for easier reference in prompts
        for k,v in self.entity_types.items():
            type_to_number_map[k] = start
            number_to_type_map[start] = k
            start += 1

        # Creates a string that can be used in prompts to describe the entity types and their corresponding numbers
        HEADER = f"""
        There exist {number_of_keys} possible routing types an entity can take on. An entity can belong to one or more routing types. Keep this in mind when you describe named entities.
        You are not allowed to invent your own routing types. Read the following routing types and select all routing types that apply to each entity.
        """

        BODY = ""
        for key, description in self.entity_types.items():
            number = type_to_number_map[key]
            BODY += f"{number}: {key} - {description}\n"

        return HEADER + BODY, type_to_number_map, number_to_type_map

    ##########################################################################
    #       Entity Extraction & Ingestion Operations for Indexing            #
    ##########################################################################
    async def extract_entity(self, entity_types_prompt, number_to_type_map, entities_for_prompt, chunk, cooldown=30):
        if not self.conn:
            raise RuntimeError("Database not opened")

        # A chunk can have no temporary entities when every relationship proposal
        # was rejected. Do not send an empty entity list to the language model.
        if not entities_for_prompt:
            chunk_hash = self.hash_func(chunk)
            async with self.lock:
                await self.conn.execute(
                    "UPDATE chunk SET are_entities_extracted = 1 WHERE chunk_id = ?",
                    (chunk_hash,)
                )
                await self.conn.commit()
            return 0

        # We fill the entity recognition prompt with the expected entities and the chunk.
        try:
            chunk_hash = self.hash_func(chunk)
            embedding_model_hash = self.hash_func(self.embedding_model.model_name)
            embedding_model_hash_str = f"{embedding_model_hash}"
            ENTITY_RECOGNITION_PROMPT_FILLED = ENTITY_RECOGNITION_PROMPT.format(
                routing_types=entity_types_prompt,
                entities=json.dumps(entities_for_prompt),
                chunk=chunk,
                input_example=INPUT_EXAMPLE,
                output_example=OUTPUT_EXAMPLE
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error preparing entity recognition prompt: {e}")

        # We call the language model with the filled prompt and expect a response in the format of ExtractedEntitiesResponse.
        try:
          response = await self.language_model(ENTITY_RECOGNITION_PROMPT_FILLED, structuredInput=ExtractedEntitiesResponse)
        except Exception as e:
           traceback.print_exc()
           print(f"Error extracting entities: {e}")
           return 0

        returned_entities = response[0]["entities"]
        expected_entity_count = len(entities_for_prompt)
        returned_entity_count = len(returned_entities)

        if returned_entity_count != expected_entity_count:
            print(
                f"Invalid number of entities returned by LLM. "
                f"Expected {expected_entity_count}, received {returned_entity_count}."
            )
            return 0

        entity_input_by_id = {
            entity["id"]: entity
            for entity in entities_for_prompt
        }
        expected_entity_ids = set(entity_input_by_id)
        returned_entity_ids = {
            entity["entity_id"]
            for entity in returned_entities
        }

        if returned_entity_ids != expected_entity_ids:
            print(
                f"Invalid entity IDs returned by LLM. "
                f"Expected {sorted(expected_entity_ids)}, received {sorted(returned_entity_ids)}."
            )
            return 0

        async with self.lock:
            try:
                new_entities_added = 0
                seen_entity_ids = set()
                for entity in returned_entities:
                    # We extract key metadata from the LLM response to add to the database.
                    entity_str = json.dumps(entity, indent=2)
                    tokens = len(self.embedding_model.tokenize(entity_str))
                    local_entity_id = entity["entity_id"]
                    entity_name = entity_input_by_id[local_entity_id]["entity_name"].lower()
                    entity_types = [
                        number_to_type_map.get(entity_type, entity_type)
                        for entity_type in entity["routing_type"]
                    ]
                    entity_types = sorted(set(entity_types))

                    # To generate a unique entity ID, we hash the normalized entity name.
                    entity_id = self.hash_func(entity_name)

                    if entity_id in seen_entity_ids:
                        continue
                    seen_entity_ids.add(entity_id)
                    entity_description = entity["entity_description"]

                    questions = []

                    dupeCheck = await self.conn.execute(
                    "SELECT entity_description, questions, tokens FROM entity WHERE entity_id = ?",
                    (entity_id,)
                    )
                    existing_entity = await dupeCheck.fetchone()
                    await dupeCheck.close()

                    if not existing_entity:
                        await self.conn.execute(
                            "INSERT INTO entity (entity_id, entity_name, entity_description, questions, embedding_model_id, tokens) VALUES (?, ?, ?, ?, ?, ?)",
                            (entity_id, entity_name, entity_description, json.dumps(questions), embedding_model_hash_str, tokens)
                        )
                        new_entities_added += 1
                    else:
                        old_description = existing_entity[0]
                        old_questions = json.loads(existing_entity[1])
                        old_tokens = existing_entity[2]
                        updated_description = old_description + "<SEP>" + entity_description
                        updated_questions = old_questions + questions
                        updated_tokens = old_tokens + tokens
                        await self.conn.execute(
                        """
                        UPDATE entity
                        SET entity_description = ?, questions = ?, tokens = ?
                        WHERE entity_id = ?
                        """,
                        (
                            updated_description,
                            json.dumps(updated_questions),
                            updated_tokens,
                            entity_id,
                        )
                        )

                    for entity_type in entity_types:
                        await self.conn.execute(
                            "INSERT OR IGNORE INTO entity_to_entity_type_mapping (entity_id, entity_type) VALUES (?, ?)",
                            (entity_id, entity_type)
                        )

                    await self.conn.execute(
                        "INSERT OR IGNORE INTO chunk_to_entity_mapping (chunk_id, entity_id) VALUES (?, ?)",
                        (chunk_hash, entity_id)
                    )

                await self.conn.execute(
                    "UPDATE chunk SET are_entities_extracted = 1 WHERE chunk_id = ?",
                    (chunk_hash,)
                )
                await self.save_input_and_output_tokens_for_llm(response[1])
                await self.conn.commit()
                return new_entities_added
            except Exception as e:
                await self.conn.rollback()
                traceback.print_exc()
                print(f"Error saving extracted entities to database: {e}")
                return 0

    async def extract_entities_from_ingested_chunks(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
      if not self.conn:
       raise RuntimeError("Database connection is not established")

      MAX_TRIES = 4
      TRY = 0

      while True:
        chunks = await self.get_chunks_waiting_for_entity_extraction()
        if not chunks or TRY >= MAX_TRIES:
            break
        TRY += 1
        if verbose:
            print(f"Found {len(chunks)} chunks waiting for entity extraction.")

        prompt_string, type_to_number_map, number_to_type_map = self.prepare_entity_type_mapping()

        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")

        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")

        total_chunks = len(chunks)

        entity_progress = tqdm_asyncio(
        total=None,
        desc="Extracting Entities",
        bar_format="{desc}: {n} entities [{elapsed}, {rate_fmt}]"
        )

        # We process the chunks in batches starting from 0 to the total number of chunks, with a step size of requests_per_minute.
        for batch_start in range(0, total_chunks, requests_per_minute):
            batch = chunks[
                batch_start:batch_start + requests_per_minute
            ]

            tasks = []
            for chunk in batch:
                chunk_hash = self.hash_func(chunk[0])
                temp_entities = await self.get_temp_entities_connected_to_chunk(chunk_hash)
                number_to_temp_entity_id_map, entities_for_prompt = await self.prepare_local_entity_mapping_for_chunk(temp_entities)
                tasks.append(
                    self.extract_entity(
                        entity_types_prompt=prompt_string,
                        number_to_type_map=number_to_type_map,
                        entities_for_prompt=entities_for_prompt,
                        chunk=chunk[0]
                    )
                )

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

            # Each successful result is the number of NEW entities inserted by extract_entity().
            for result in results:
                if not isinstance(result, Exception):
                    entity_progress.update(result)

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

    async def embed_entities(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
        entities = await self.get_entities_waiting_for_embedding()
        async def embed_entity(entity):
                    try:
                        content = await self.get_entity_prompt_for_embedding(entity=entity)
                        if verbose:
                            print(f"Prompt for entity {entity['entity_name']}: {content}")
                        await self.add_embedding_func(
                            content=json.dumps(content, indent=4),
                            table="entity",
                            pk_hash=entity["entity_id"]
                        )
                        return None
                    except Exception as error:
                        if verbose:
                            print(f"Embedding failed, retrying next batch: {error}")
                        traceback.format_exc()
                        return entity
        if entities:
            if verbose:
                print(f"Found {len(entities)} entities waiting for embedding.")
            failed_entities = entities.copy()

            while failed_entities:
                current_batch = failed_entities[:requests_per_minute]
                remaining_entities = failed_entities[requests_per_minute:]
                failed_entities = []

                if verbose:
                    print(f"Sending batch of {len(current_batch)} entities to embedding model...")

                batch_results = await tqdm_asyncio.gather(
                        *(embed_entity(entity) for entity in current_batch),
                        desc="Embedding entities",
                        total=len(current_batch)
                    )

                failed_entities.extend([
                    entity for entity in batch_results
                    if entity is not None
                ])

                failed_entities.extend(remaining_entities)

                if failed_entities:
                    if verbose:
                        print(
                            f"Cooldown for {async_cooldown} seconds. "
                            f"{len(failed_entities)} entities remaining."
                        )

                    await asyncio.sleep(async_cooldown)
