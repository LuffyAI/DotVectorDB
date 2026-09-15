from dotvectordb.prompts.type_discovery_prompts import TYPE_DISCOVERY_PROMPT, TypeProposals, OUTPUT_EXAMPLE
import json
import traceback
import asyncio
from tqdm.asyncio import tqdm_asyncio


class RelationshipIndexer2:
    def __init__(self, conn, lock, language_model, hash_func):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.hash_func = hash_func

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
        """Gets all chunks waiting for relationship extraction"""
        async with self.lock:
            cursor = await self.conn.execute(
            """SELECT content FROM chunk WHERE are_relations_extracted = 0"""
        )
        chunks = await cursor.fetchall()
        await cursor.close()
        return chunks

    ##########################################################################
    #       Relationship Extraction & Ingestion Operations for Indexing      #
    ##########################################################################
    async def extract_relation(self, chunk, cooldown=30):
        if not self.conn:
            raise RuntimeError("Database not opened")

        # We fill the relationship discovery prompt with the chunk of text we want to analyze.
        try:
            chunk_hash = self.hash_func(chunk)
            TYPE_DISCOVERY_PROMPT_FILLED = TYPE_DISCOVERY_PROMPT.format(
                chunk=chunk,
                output_example=OUTPUT_EXAMPLE
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error preparing relationship discovery prompt: {e}")

        # We call the language model with the filled prompt and expect a response in the format of TypeProposals.
        try:
          response = await self.language_model(TYPE_DISCOVERY_PROMPT_FILLED, structuredInput=TypeProposals)
        except Exception as e:
           traceback.print_exc()
           print(f"Error extracting relationships: {e}")
           return 0

        async with self.lock:
            try:
                new_relationships_added = 0
                for proposal in response[0]["Proposals"]:
                    if not proposal["follows_named_entity_rules"]:
                        continue

                    # We normalize the entity names and tags returned by the LLM.
                    src_entity = proposal["src_entity"].lower()
                    dst_entity = proposal["dst_entity"].lower()
                    src_entity_types = [tag.lower() for tag in proposal["src_entity_types"]]
                    dst_entity_types = [tag.lower() for tag in proposal["dst_entity_types"]]
                    relationship_description = proposal["relationship_description"]

                    # Temporary entity IDs are based on normalized entity names because final types have not been assigned yet.
                    src_temp_entity_id = self.hash_func(src_entity)
                    dst_temp_entity_id = self.hash_func(dst_entity)

                    # Add or update the temporary source entity and its discovered tags.
                    srcDupeCheck = await self.conn.execute(
                        "SELECT entity_tags FROM temp_entity WHERE temp_entity_id = ?",
                        (src_temp_entity_id,)
                    )
                    existing_src_entity = await srcDupeCheck.fetchone()
                    await srcDupeCheck.close()

                    if not existing_src_entity:
                        await self.conn.execute(
                            "INSERT INTO temp_entity (temp_entity_id, entity_name, entity_tags) VALUES (?, ?, ?)",
                            (src_temp_entity_id, src_entity, json.dumps(src_entity_types))
                        )
                    else:
                        old_src_tags = json.loads(existing_src_entity[0])
                        updated_src_tags = list(dict.fromkeys(old_src_tags + src_entity_types))
                        await self.conn.execute(
                            "UPDATE temp_entity SET entity_tags = ? WHERE temp_entity_id = ?",
                            (json.dumps(updated_src_tags), src_temp_entity_id)
                        )

                    # Add or update the temporary destination entity and its discovered tags.
                    dstDupeCheck = await self.conn.execute(
                        "SELECT entity_tags FROM temp_entity WHERE temp_entity_id = ?",
                        (dst_temp_entity_id,)
                    )
                    existing_dst_entity = await dstDupeCheck.fetchone()
                    await dstDupeCheck.close()

                    if not existing_dst_entity:
                        await self.conn.execute(
                            "INSERT INTO temp_entity (temp_entity_id, entity_name, entity_tags) VALUES (?, ?, ?)",
                            (dst_temp_entity_id, dst_entity, json.dumps(dst_entity_types))
                        )
                    else:
                        old_dst_tags = json.loads(existing_dst_entity[0])
                        updated_dst_tags = list(dict.fromkeys(old_dst_tags + dst_entity_types))
                        await self.conn.execute(
                            "UPDATE temp_entity SET entity_tags = ? WHERE temp_entity_id = ?",
                            (json.dumps(updated_dst_tags), dst_temp_entity_id)
                        )

                    # Map both temporary entities back to the chunk where they were discovered.
                    await self.conn.execute(
                        "INSERT OR IGNORE INTO chunk_to_temp_entity_mapping (chunk_id, temp_entity_id) VALUES (?, ?)",
                        (chunk_hash, src_temp_entity_id)
                    )
                    await self.conn.execute(
                        "INSERT OR IGNORE INTO chunk_to_temp_entity_mapping (chunk_id, temp_entity_id) VALUES (?, ?)",
                        (chunk_hash, dst_temp_entity_id)
                    )

                    # Temporary relationships use the same directed source-destination hash pattern as RelationshipIndexer.
                    relation_id = f"{src_temp_entity_id}_{dst_temp_entity_id}"
                    temp_relation_id = self.hash_func(relation_id)

                    dupeCheck = await self.conn.execute(
                        "SELECT relationship_description FROM temp_relationship WHERE temp_relation_id = ?",
                        (temp_relation_id,)
                    )
                    existing_relation = await dupeCheck.fetchone()
                    await dupeCheck.close()

                    if not existing_relation:
                        await self.conn.execute(
                            "INSERT INTO temp_relationship (temp_relation_id, src_temp_entity_id, dst_temp_entity_id, relationship_description) VALUES (?, ?, ?, ?)",
                            (temp_relation_id, src_temp_entity_id, dst_temp_entity_id, relationship_description)
                        )
                        new_relationships_added += 1
                    else:
                        old_description = existing_relation[0]
                        updated_description = old_description + "<SEP>" + relationship_description
                        await self.conn.execute(
                            """
                            UPDATE temp_relationship
                            SET relationship_description = ?
                            WHERE temp_relation_id = ?
                            """,
                            (
                                updated_description,
                                temp_relation_id,
                            )
                        )

                await self.conn.execute(
                    "UPDATE chunk SET are_relations_extracted = 1 WHERE chunk_id = ?",
                    (chunk_hash,)
                )
                await self.save_input_and_output_tokens_for_llm(response[1])
                await self.conn.commit()
                return new_relationships_added
            except Exception as e:
                await self.conn.rollback()
                traceback.print_exc()
                print(f"Error saving extracted relationships to database: {e}")
                return 0

    async def extract_relations_from_chunks(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
     # Check if the database connection is established
     if not self.conn:
            raise RuntimeError("Database not opened")

     MAX_TRIES = 4
     TRY = 0
     while True:
        # We pull only the chunks that haven't been processed for relationship extraction yet.
        chunks = await self.get_chunks_waiting_for_relation_extraction()
        if not chunks or TRY >= MAX_TRIES:
            break

        TRY += 1

        if verbose:
            print(f"Found {len(chunks)} chunks waiting for relationship extraction.")

        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")

        if async_cooldown <= 0:
            raise ValueError("async_cooldown must be greater than 0")

        total_chunks = len(chunks)

        # We do not know how many unique relationships will be added in advance.
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

            tasks = [
                self.extract_relation(
                    chunk=chunk[0]
                )
                for chunk in batch
            ]

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

            # Each successful result is the number of NEW relationships inserted by extract_relation().
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
