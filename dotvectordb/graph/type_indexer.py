from dotvectordb.prompts.type_discovery_prompts import TYPE_DISCOVERY_PROMPT, TypeProposals, OUTPUT_EXAMPLE
import json
import traceback
import asyncio
from tqdm.asyncio import tqdm_asyncio

class TypeIndexer:
    def __init__(self, conn, lock, language_model, embedding_model, hash_func, tag_func):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.hash_func = hash_func
        self.tag_func = tag_func

    ############################################################
    #   Token Economics Tracking                               # 
    ############################################################
    async def save_input_and_output_tokens_for_llm(self, total_tokens):
            await self.conn.execute(
                """
                INSERT INTO llm_token_usage (stage, total_tokens)
                VALUES (?, ?)
                """,
                ("type_discovery", total_tokens)
            )
    ############################################################
    #   Fetch Type-Related Data For Entity Type Indexing          # 
    ############################################################
    async def get_chunks_waiting_for_type_discovery(self):
        """Gives us all the chunks ready for the type discovery process"""
        async with self.lock:
            cursor = await self.conn.execute(
            """SELECT content FROM chunk WHERE are_types_discovered=0"""
        )
        chunks = await cursor.fetchall()
        await cursor.close()
        return chunks
    async def check_if_tag_exists(self, tag):
            """Checks if a tag already exists in the database"""
            cursor = await self.conn.execute(
                "SELECT 1 FROM entity_tags WHERE entity_tag = ?",
                (tag,)
            )
            exists = await cursor.fetchone() is not None
            await cursor.close()
            return exists


    ##########################################################################
    #       Type Discovery & Ingestion Operations for Indexing            # 
    ##########################################################################
    async def extract_entity_tags(self, chunk, cooldown=30):
        if not self.conn:
            raise RuntimeError("Database not opened")
        
        # We fill the entity recognition prompt with the entity types and the chunk of text we want to analyze.
        try:
            chunk_hash = self.hash_func(chunk)
            embedding_model_hash = self.hash_func(self.embedding_model.model_name)
            embedding_model_hash_str = f"{embedding_model_hash}"
            TYPE_DISCOVERY_PROMPT_FILLED = TYPE_DISCOVERY_PROMPT.format(chunk=chunk, output_example=OUTPUT_EXAMPLE)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error preparing type discovery prompt: {e}")
            
        # We call the language model with the filled prompt and expect a response in the format of ExtractedEntitiesResponse.
        try:
          response = await self.language_model(TYPE_DISCOVERY_PROMPT_FILLED, structuredInput=TypeProposals)
          print(f"LLM response for chunk hash {chunk_hash}: {response}")
        except Exception as e:
           traceback.print_exc()
           print(f"Error extracting entities: {e}")
           return 0

        async with self.lock:
            try:
                new_tags_added = 0
                # Set to deduplicate entities in the case the LLM returns the same entity multiple times in the same response.
                seen_entity_ids = set()
                for proposal in response[0]["Proposals"]:
                    # We extract key metadata from the LLM response to add to the database.
                    # We normalize the enrity name to lowercase and map the entity type number back to its string representation.
                    proposal_str = json.dumps(proposal, indent=2)
                    tokens = len(self.embedding_model.tokenize(proposal_str))
                    src_entity = proposal["src_entity"].lower()
                    dst_entity = proposal["dst_entity"].lower()
                    src_entity_types = [tag.lower() for tag in proposal["src_entity_types"]]
                    dst_entity_types = [tag.lower() for tag in proposal["dst_entity_types"]]

                    for type in src_entity_types:
                        if await self.check_if_tag_exists(type):
                            continue
                        await self.tag_func(type)
                        new_tags_added += 1

                    for type in dst_entity_types:
                        if await self.check_if_tag_exists(type):
                            continue
                        await self.tag_func(type)
                        new_tags_added += 1

                    await self.conn.execute("UPDATE chunk SET are_types_discovered = 1 WHERE chunk_id = ?", (chunk_hash,))
                # Save LLM token usage for this entity extraction call.
                await self.save_input_and_output_tokens_for_llm(response[1])
                # IMPORTANT: We only commit the transaction after all entities from the chunk have been processed.
                await self.conn.commit()
                # Tells caller how many were extracted
                return new_tags_added
            except Exception as e:
                await self.conn.rollback()
                traceback.print_exc()
                print(f"Error saving extracted entities to database: {e}")
                return 0

    async def extract_tags_from_ingested_chunks(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
      if not self.conn:
       raise RuntimeError("Database connection is not established")
        
      # First, we retrieve all the chunks that haven't been processed for entity extraction yet
      

      MAX_TRIES = 4
      TRY = 0

      while True:
        chunks = await self.get_chunks_waiting_for_type_discovery()
        if not chunks or TRY >= MAX_TRIES:
            break
        TRY += 1
        if verbose:
            print(f"Found {len(chunks)} chunks waiting for type discovery.")


        # Third, we validate the requests per minute and cool down.
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")
        
        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")
        
        total_chunks = len(chunks)

        # We do not know how many unique entities will be added in advance.
        tag_progress = tqdm_asyncio(
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
                self.extract_entity_tags(
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
                    tag_progress.update(result)

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


    # async def embed_entities(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
    #     entities = await self.get_entities_waiting_for_embedding()
    #     async def embed_entity(entity):
    #                 try:
    #                     content = await self.get_entity_prompt_for_embedding(entity=entity)
    #                     entity_name = entity["entity_name"].lower()
    #                     entity_type = entity["entity_type"]
    #                     entity_id_original = self.hash_func(f"{entity_name}_{entity_type}")
    #                     if verbose:
    #                         print(f"Prompt for entity {entity['entity_name']}: {content}")
    #                     await self.add_embedding_func(
    #                         content=json.dumps(content, indent=4),
    #                         table="entity",
    #                         pk_hash=entity_id_original
    #                     )
    #                     return None
    #                 except Exception as error:
    #                     if verbose:
    #                         print(f"Embedding failed, retrying next batch: {error}")
    #                     traceback.format_exc()
    #                     return entity
    #     if entities:
    #         if verbose:
    #             print(f"Found {len(entities)} entities waiting for embedding.")
    #         failed_entities = entities.copy()

    #         while failed_entities:
    #             current_batch = failed_entities[:requests_per_minute]
    #             remaining_entities = failed_entities[requests_per_minute:]
    #             failed_entities = []

    #             if verbose:
    #                 print(f"Sending batch of {len(current_batch)} entities to embedding model...")

    #             batch_results = await tqdm_asyncio.gather(
    #                     *(embed_entity(entity) for entity in current_batch),
    #                     desc="Embedding entities",
    #                     total=len(current_batch)
    #                 )
               
    #             failed_entities.extend([
    #                 entity for entity in batch_results
    #                 if entity is not None
    #             ])

    #             failed_entities.extend(remaining_entities)

    #             if failed_entities:
    #                 if verbose:
    #                     print(
    #                         f"Cooldown for {async_cooldown} seconds. "
    #                         f"{len(failed_entities)} entities remaining."
    #                     )

    #                 await asyncio.sleep(async_cooldown)
