from dotvectordb.prompts.entity_prompts import ENTITY_RECOGNITION_PROMPT, ExtractedEntitiesResponse, INPUT_EXAMPLE, OUTPUT_EXAMPLE
import json
import traceback
import asyncio
from tqdm.asyncio import tqdm_asyncio

class EntityIndexer:
    def __init__(self, conn, lock, language_model, embedding_model, hash_func, entity_types, add_embedding_func):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.hash_func = hash_func
        self.add_embedding_func = add_embedding_func
        self.entity_types = entity_types

    ############################################################
    #   Fetch Entity-Related Data For Entity Indexing          # 
    ############################################################
    async def get_chunks_waiting_for_entity_extraction(self):
        """Gives us all the chunks ready for the entity extraction process"""
        async with self.lock:
            cursor = await self.conn.execute(
            """SELECT content FROM chunk WHERE is_embedded = 1 AND are_entities_extracted = 0"""
        )
        chunks = await cursor.fetchall()
        await cursor.close()
        return chunks
    
    async def get_entities_waiting_for_embedding(self):
        """After all entity metadata has been filled, we can get all the entities that are ready for embedding"""
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
        formatted_entity = {
        "entity_name": entity["entity_name"],
        "entity_type": entity["entity_type"],
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
        There exist {number_of_keys} possible types an entity can take on. An entity can belong to one type. Keep this in mind when you extract named entities.
        You are not allowed to invent your own types. Read the following types and select the type that best fits the extracted entity.
        """

        # Create a string that starts with the number, then the entity, and then the description
        # 1: PERSON - A named fictional or non-fictional human. 
        BODY = ""
        for key, description in self.entity_types.items():
            number = type_to_number_map[key]
            BODY += f"{number}: {key} - {description}\n"

        return HEADER + BODY, type_to_number_map, number_to_type_map
    
    ##########################################################################
    #       Entity Extraction & Ingestion Operations for Indexing            # 
    ##########################################################################
    async def extract_entity(self, entity_types_prompt, number_to_type_map, chunk, cooldown=30):
        if not self.conn:
            raise RuntimeError("Database not opened")
        
        # We fill the entity recognition prompt with the entity types and the chunk of text we want to analyze.
        try:
            chunk_hash = self.hash_func(chunk)
            embedding_model_hash = self.hash_func(self.embedding_model.model_name)
            embedding_model_hash_str = f"{embedding_model_hash}"
            ENTITY_RECOGNITION_PROMPT_FILLED = ENTITY_RECOGNITION_PROMPT.format(entity_types=entity_types_prompt, chunk=chunk, input_example=INPUT_EXAMPLE, output_example=OUTPUT_EXAMPLE)
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

        async with self.lock:
            try:
                new_entities_added = 0
                for entity in response[0]["entities"]:
                    # We extract key metadata from the LLM response to add to the database.
                    # We normalize the enrity name to lowercase and map the entity type number back to its string representation.
                    entity_str = json.dumps(entity, indent=2)
                    tokens = len(self.embedding_model.tokenize(entity_str))
                    entity_name = entity["entity_name"].lower()
                    entity_type = number_to_type_map.get(entity["entity_type"], "Unknown")

                    # To generate unique entity IDs, we hash the entity name and type.
                    entity_id_original = f"{entity_name}_{entity_type}"
                    entity_id = self.hash_func(entity_id_original)
                    entity_description = entity["entity_description"]

                    # We retrieve all predicted questions for the entity and store them in a list.
                    questions = []
                    for question in entity["entity_questions"]["Questions"]:
                        questions.append(question)

                    # Duplicate check: We check if the entity already exists in the database by comparing the entity name and type.
                    # If it does, we update the existing record to append the description, questions, and extra tokens.
                    dupeCheck = await self.conn.execute(
                    "SELECT entity_description, questions, tokens FROM entity WHERE entity_id = ?",
                    (entity_id,)
                    )
                    existing_entity = await dupeCheck.fetchone()

                    if not existing_entity:
                        await self.conn.execute("INSERT INTO entity (entity_id, entity_name, entity_type, entity_description, questions, embedding_model_id, tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                                (entity_id, entity_name, entity_type, entity_description, json.dumps(questions), embedding_model_hash_str, tokens))
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
                    await self.conn.execute("UPDATE chunk SET are_entities_extracted = 1 WHERE chunk_id = ?", (chunk_hash,))
                    await self.conn.execute("INSERT INTO chunk_to_entity_mapping (chunk_id, entity_id) VALUES (?, ?)", (chunk_hash, entity_id))
                # IMPORTANT: We only commit the transaction after all entities from the chunk have been processed.
                await self.conn.commit()
                # Tells caller how many were extracted
                return new_entities_added
            except Exception as e:
                await self.conn.rollback()
                traceback.print_exc()
                print(f"Error saving extracted entities to database: {e}")
                return 0

    async def extract_entities_from_ingested_chunks(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
        if not self.conn:
            raise RuntimeError("Database connection is not established")
        
        # First, we retrieve all the chunks that haven't been processed for entity extraction yet
        chunks = await self.get_chunks_waiting_for_entity_extraction()

        if verbose:
            print(f"Found {len(chunks)} chunks waiting for entity extraction.")

        # Second, we also get thge entity type mapping and the prompt string that will be used for entity extraction
        prompt_string, type_to_number_map, number_to_type_map = self.prepare_entity_type_mapping()

        # Third, we validate the requests per minute and cool down.
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
        
        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")
        
        total_chunks = len(chunks)

        # We do not know how many unique entities will be added in advance.
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

            tasks = [
                self.extract_entity(
                    entity_types_prompt=prompt_string,
                    number_to_type_map=number_to_type_map,
                    chunk=chunk[0]
                )
                for chunk in batch
            ]

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

            # Each successful result is the number of NEW entities
            # inserted by extract_entity().
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


    async def embed_entities(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
        entities = await self.get_entities_waiting_for_embedding()
        async def embed_entity(entity):
                    try:
                        content = await self.get_entity_prompt_for_embedding(entity=entity)
                        entity_name = entity["entity_name"].lower()
                        entity_type = entity["entity_type"]
                        entity_id_original = self.hash_func(f"{entity_name}_{entity_type}")
                        if verbose:
                            print(f"Prompt for entity {entity['entity_name']}: {content}")
                        await self.add_embedding_func(
                            content=json.dumps(content, indent=4),
                            table="entity",
                            pk_hash=entity_id_original
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

    ##########################################################################
    #                Random Sampling For Entity Types                        # 
    ########################################################################## 
    async def get_random_sample_of_entities_by_type(self, entity_type, sample_size=5):
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM entity WHERE entity_type = ? ORDER BY RANDOM() LIMIT ?",
                (entity_type, sample_size)
            )
            rows = await cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            await cursor.close()
            entities = [
                            dict(zip(column_names, row))
                            for row in rows
                        ]
            return entities

        
    
    
    
    


        