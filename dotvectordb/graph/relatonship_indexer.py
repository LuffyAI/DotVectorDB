from dotvectordb.prompts.relationship_prompts import RELATIONSHIP_PROMPT, ENTITY_EXAMPLE, INPUT_EXAMPLE, OUTPUT_EXAMPLE, Relationship
import json
import traceback
import asyncio
from tqdm.asyncio import tqdm_asyncio

class RelationshipIndexer:
    def __init__(self, conn, lock, language_model, hash_func, embedding_model, add_embedding_func):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.hash_func = hash_func
        self.embedding_model = embedding_model
        self.add_embedding_func = add_embedding_func 
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
    
   ##########################################################################
   #   Handles all preparation needed to prepare the relationship prompt    # 
   ##########################################################################
    async def get_relationship_prompt_for_embedding(self, relationship):
        formatted_relationship = {
        "src_entity_id": relationship["src_entity_id"],
        "dst_entity_id": relationship["dst_entity_id"],
        "relation_description": relationship["relation_description"],
        "questions": relationship["questions"],
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
    
    ##########################################################################
    #       Relationship Extraction & Ingestion Operations for Indexing      # 
    ##########################################################################
    async def extract_relation(self, number_to_entity_id_map, number_to_entity_name_map, chunk, cooldown=30):
        if not self.conn:
            raise RuntimeError("Database not opened")
        
        # We fill the relationship recognition prompt with the entity mapping and the chunk of text we want to analyze.
        chunk_hash = self.hash_func(chunk)
        embedding_model_hash = self.hash_func(self.embedding_model.model_name)
        embedding_model_hash_str = f"{embedding_model_hash}"
        RELATIONSHIP_PROMPT_FILLED = RELATIONSHIP_PROMPT.format(entities=json.dumps(number_to_entity_name_map), chunk=chunk, entity_example=ENTITY_EXAMPLE, input_example=INPUT_EXAMPLE, output_example=OUTPUT_EXAMPLE)

        # We call the language model with the filled prompt and expect a response in the format of ExtractedEntitiesResponse.
        try:
          response = await self.language_model(RELATIONSHIP_PROMPT_FILLED, structuredInput=Relationship)
        except Exception as e:
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

                    # We retrieve all predicted questions for the relationship and store them in a list.
                    questions = []
                    for question in relation["relation_questions"]["Questions"]:
                        questions.append(question)
                    
                    # Duplicate check: We check if the relationship already exists in the database by comparing the source and destination entity IDs.
                    dupeCheck = await self.conn.execute(
                    "SELECT relation_description, questions, tokens FROM relationship WHERE relation_id = ?",
                    (relation_id_hash,)
                    )
                    existing_relation = await dupeCheck.fetchone()

                    if not existing_relation:
                        await self.conn.execute("INSERT INTO relationship (relation_id, src_entity_id, dst_entity_id, relation_description, embedding_model_id, questions, tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                                (relation_id_hash, true_src_entity_id, true_dst_entity_id, description, embedding_model_hash_str, json.dumps(questions), tokens))
                        new_relationships_added += 1
                    else:
                        old_description = existing_relation[0]
                        old_questions = json.loads(existing_relation[1])
                        old_tokens = existing_relation[2]
                        updated_description = old_description + "<SEP>" + description
                        updated_questions = old_questions + questions
                        updated_tokens = old_tokens + tokens
                        await self.conn.execute(
                        """
                        UPDATE relationship
                        SET relation_description = ?, questions = ?, tokens = ?
                        WHERE relation_id = ?
                        """,
                        (
                            updated_description,
                            json.dumps(updated_questions),
                            updated_tokens,
                            relation_id_hash,
                        )
                        )
                await self.conn.execute("UPDATE chunk SET are_relations_extracted = 1 WHERE chunk_id = ?", (chunk_hash,))
                await self.save_input_and_output_tokens_for_llm(response[1])
                await self.conn.commit()
                return new_relationships_added
            except Exception as e:
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
        if not chunks or TRY >= MAX_TRIES:
            break

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
                chunk_id = self.hash_func(chunk[0])
                entity_ids = await self.get_entities_connected_to_chunk(chunk_id)
                number_to_entity_id_map, number_to_entity_name_map = await self.prepare_local_entity_mapping_for_chunk(entity_ids)
                tasks.append(
                    self.extract_relation(
                        number_to_entity_id_map=number_to_entity_id_map,
                        number_to_entity_name_map=number_to_entity_name_map,
                        chunk=chunk[0]
                    )
                )

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
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
    
    async def embed_relationships(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
        relationships = await self.get_relationships_waiting_for_embedding()
        async def embed_relationship(relationship, verbose=verbose):
            pass
            try:
                 content = await self.get_relationship_prompt_for_embedding(relationship=relationship)
                 src_entity_id = relationship["src_entity_id"]
                 dst_entity_id = relationship["dst_entity_id"]
                 hash_input = self.hash_func(f"{src_entity_id}_{dst_entity_id}")
                 if verbose:
                     print(f"Prompt for relationship {relationship['relation_id']}: {content}")
                 await self.add_embedding_func(
                     content=json.dumps(content, indent=4),
                     table="relationship",
                     pk_hash=hash_input
                 )
                 return None
            except Exception as error:
                if verbose:
                    print(f"Embedding failed, retrying next batch: {error}")
                traceback.format_exc()
                return relationship
            
        if relationships:
            if verbose:
                    print(f"Found {len(relationships)} unprocessed relationships. Starting to add embeddings...")
            failed_relationships = relationships.copy()

            while failed_relationships:
                current_batch = failed_relationships[:requests_per_minute]
                remaining_relationships = failed_relationships[requests_per_minute:]
                failed_relationships = []

            
                batch_results = await tqdm_asyncio.gather(
                        *(embed_relationship(relationship) for relationship in current_batch),
                        desc="Embedding relationships",
                        total=len(current_batch)
                    )
              

                failed_relationships.extend([
                    relationship for relationship in batch_results
                    if relationship is not None
                ])

                failed_relationships.extend(remaining_relationships)

                if failed_relationships:
                    if verbose:
                        print(
                            f"Cooldown for {async_cooldown} seconds. "
                            f"{len(failed_relationships)} relationships remaining."
                        )

                    await asyncio.sleep(async_cooldown)

    
 