from dotvectordb.prompts.relationship_prompts import (
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
        max_relation_questions=None,
        max_relation_questions_token_size=None,
        max_relation_keywords=None,
        max_relation_keywords_token_size=None,
        max_question_token_size=None,
        max_keyword_token_size=None,
        max_entity_relation_description_token_size=None,
        recommended_entity_relation_description_token_size=None,
        summary_language=None,
    ):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.hash_func = hash_func
        self.embedding_model = embedding_model
        self.add_embedding_func = add_embedding_func
        self.max_relation_questions = (
            max_relation_questions
            if max_relation_questions is not None
            else int(os.getenv("MAX_RELATION_QUESTIONS", 8))
        )
        self.max_relation_questions_token_size = (
            max_relation_questions_token_size
            if max_relation_questions_token_size is not None
            else int(os.getenv("MAX_RELATION_QUESTIONS_TOKEN_SIZE", 160))
        )
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
        self.max_question_token_size = (
            max_question_token_size
            if max_question_token_size is not None
            else int(os.getenv("MAX_QUESTION_TOKEN_SIZE", 32))
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

    async def remove_extra_questions_from_relationship(self, questions, threshold=0.7):
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

    def find_least_similar_questions(self, questions):
        """Keep the most semantically diverse questions up to the limit."""
        if len(questions) <= self.max_relation_questions:
            return questions
        if self.max_relation_questions <= 0:
            return []

        embeddings = self.small_llm.encode(questions)
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
        ) / (len(questions) - 1)
        selected_indices = [int(np.argmin(average_similarities))]
        remaining_indices = set(range(len(questions))) - set(selected_indices)

        while len(selected_indices) < self.max_relation_questions:
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
        return [questions[index] for index in selected_indices]

    def remove_questions_over_token_limits(
        self,
        questions,
        src_entity_id,
        dst_entity_id,
        enforce_total_limit=True,
    ):
        """Apply per-question and aggregate token limits with a fallback."""
        fallback_question = (
            "What can you tell me about the relationship between "
            f"{src_entity_id} and {dst_entity_id}?"
        )
        questions_with_token_counts = [
            (question, len(self.embedding_model.tokenize(question)))
            for question in questions
            if isinstance(question, str) and question.strip()
        ]
        questions_with_token_counts = [
            (question, token_count)
            for question, token_count in questions_with_token_counts
            if token_count <= self.max_question_token_size
        ]

        if not questions_with_token_counts:
            return [fallback_question]

        while (
            enforce_total_limit
            and sum(token_count for _, token_count in questions_with_token_counts)
            > self.max_relation_questions_token_size
            and questions_with_token_counts
        ):
            longest_index = max(
                range(len(questions_with_token_counts)),
                key=lambda index: (
                    questions_with_token_counts[index][1],
                    index,
                ),
            )
            questions_with_token_counts.pop(longest_index)

        if not questions_with_token_counts:
            return [fallback_question]
        return [question for question, _ in questions_with_token_counts]

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
        "questions": relationship["questions"],
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
        embedding_model_hash = self.hash_func(self.embedding_model.model_name)
        embedding_model_hash_str = f"{embedding_model_hash}"
        RELATIONSHIP_PROMPT_FILLED = RELATIONSHIP_PROMPT.format(entities=json.dumps(number_to_entity_name_map), chunk=chunk, entity_example=ENTITY_EXAMPLE, input_example=INPUT_EXAMPLE, output_example=OUTPUT_EXAMPLE)

        # We call the language model with the filled prompt and expect a response in the format of ExtractedEntitiesResponse.
        try:
          response = await self.language_model(RELATIONSHIP_PROMPT_FILLED, structuredInput=Relationships)
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

                    questions = relation["relation_questions"]
                    keywords = relation["keywords"]
                    weight = relation["weight"]
                    is_summarized = (
                        len(self.embedding_model.tokenize(description))
                        <= self.max_entity_relation_description_token_size
                    )
                    
                    # Duplicate check: We check if the relationship already exists in the database by comparing the source and destination entity IDs.
                    dupeCheck = await self.conn.execute(
                    """
                    SELECT relation_description, questions, keywords, weight, tokens
                    FROM relationship
                    WHERE relation_id = ?
                    """,
                    (relation_id_hash,)
                    )
                    existing_relation = await dupeCheck.fetchone()

                    if not existing_relation:
                        await self.conn.execute(
                            """
                            INSERT INTO relationship (
                                relation_id,
                                src_entity_id,
                                dst_entity_id,
                                relation_description,
                                embedding_model_id,
                                questions,
                                keywords,
                                weight,
                                tokens,
                                is_summarized
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                relation_id_hash,
                                true_src_entity_id,
                                true_dst_entity_id,
                                description,
                                embedding_model_hash_str,
                                json.dumps(questions),
                                json.dumps(keywords),
                                weight,
                                tokens,
                                is_summarized,
                            ),
                        )
                        new_relationships_added += 1
                    else:
                        old_description = existing_relation[0]
                        old_questions = json.loads(existing_relation[1])
                        old_keywords = json.loads(existing_relation[2])
                        old_weight = existing_relation[3]
                        old_tokens = existing_relation[4]
                        updated_description = old_description + "<SEP>" + description
                        updated_questions = old_questions + questions
                        updated_keywords = old_keywords + keywords
                        updated_weight = old_weight + weight
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
                            questions = ?,
                            keywords = ?,
                            weight = ?,
                            tokens = ?,
                            is_summarized = ?
                        WHERE relation_id = ?
                        """,
                        (
                            updated_description,
                            json.dumps(updated_questions),
                            json.dumps(updated_keywords),
                            updated_weight,
                            updated_tokens,
                            is_summarized,
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
                tasks.append(
                    self.extract_relation(
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
                        SET relation_description = ?, is_summarized = TRUE
                        WHERE relation_id = ?
                        """,
                        (new_description, relationship["relation_id"]),
                    )
                    await self.save_input_and_output_tokens_for_llm(response[1])
                    await self.conn.commit()
                return None
            except Exception as error:
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

    async def embed_relationships(self, async_cooldown=60, requests_per_minute=1000, verbose=True):
        relationships = await self.get_relationships_waiting_for_embedding()
        print(f"Found {len(relationships)} relationships waiting for embedding.")

        async def embed_relationship(relationship):
            try:
                content = await self.get_relationship_prompt_for_embedding(relationship)
                questions = content["questions"]
                if isinstance(questions, str):
                    questions = json.loads(questions)
                questions = self.remove_questions_over_token_limits(
                    questions,
                    relationship["src_entity_id"],
                    relationship["dst_entity_id"],
                    enforce_total_limit=False,
                )
                content["questions"] = (
                    await self.remove_extra_questions_from_relationship(questions)
                )
                content["questions"] = self.find_least_similar_questions(
                    content["questions"]
                )
                content["questions"] = self.remove_questions_over_token_limits(
                    content["questions"],
                    relationship["src_entity_id"],
                    relationship["dst_entity_id"],
                )
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
                    pk_hash=relationship["relation_id"],
                )
                return None
            except Exception as error:
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
            needs_questions,
            needs_keywords,
            needs_weights,
        ) = requirements
        prompt_template, enrichment_model = (
            await prepare_relationship_enrichment_prompt_sections(
                needs_descriptions=needs_descriptions,
                needs_questions=needs_questions,
                needs_keywords=needs_keywords,
                needs_weights=needs_weights,
            )
        )
        local_id_to_relationship = {
            local_relationship_id: relationship
            for local_relationship_id, relationship in enumerate(
                relationships,
                start=1,
            )
        }
        prompt_relationships = {
            local_relationship_id: {
                "src_entity_id": entity_id_to_local_id[
                    relationship["src_entity_id"]
                ],
                "dst_entity_id": entity_id_to_local_id[
                    relationship["dst_entity_id"]
                ],
            }
            for local_relationship_id, relationship
            in local_id_to_relationship.items()
        }
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
            if needs_questions:
                enriched_relationship["questions"] = returned_relationship[
                    "questions"
                ]
            if needs_keywords:
                enriched_relationship["keywords"] = returned_relationship[
                    "keywords"
                ]
            if needs_weights:
                enriched_relationship["weight"] = returned_relationship[
                    "weight"
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
                        "questions": relationship["questions"],
                        "keywords": relationship["keywords"],
                        "weight": relationship["weight"],
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
                        SELECT relation_description, questions, keywords,
                               weight, tokens
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
                                questions,
                                keywords,
                                weight,
                                embedding_model_id,
                                tokens,
                                is_summarized
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                relationship["relation_id"],
                                relationship["src_entity_id"],
                                relationship["dst_entity_id"],
                                relationship["relation_description"],
                                json.dumps(relationship["questions"]),
                                json.dumps(relationship["keywords"]),
                                relationship["weight"],
                                embedding_model_hash,
                                tokens,
                                is_summarized,
                            ),
                        )
                        new_relationships_added += 1
                    else:
                        (
                            old_description,
                            old_questions_json,
                            old_keywords_json,
                            old_weight,
                            old_tokens,
                        ) = existing_relationship
                        combined_description = (
                            f"{old_description}<SEP>"
                            f"{relationship['relation_description']}"
                        )
                        combined_questions = (
                            json.loads(old_questions_json)
                            + relationship["questions"]
                        )
                        combined_keywords = (
                            json.loads(old_keywords_json)
                            + relationship["keywords"]
                        )
                        combined_weight = old_weight + relationship["weight"]
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
                                questions = ?,
                                keywords = ?,
                                weight = ?,
                                tokens = ?,
                                is_summarized = ?
                            WHERE relation_id = ?
                            """,
                            (
                                combined_description,
                                json.dumps(combined_questions),
                                json.dumps(combined_keywords),
                                combined_weight,
                                old_tokens + tokens,
                                is_summarized,
                                relationship["relation_id"],
                            ),
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
            relationships_by_chunk.setdefault(
                relationship["chunk_id"],
                [],
            ).append(relationship)

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
                relationships_by_requirements = {}
                completed_relationships = []

                for relationship in custom_relationships:
                    requirements = (
                        not relationship["_is_relationship_enriched"],
                        not relationship["does_relationship_have_questions"],
                        not relationship["does_relationship_have_keywords"],
                        not relationship["does_relationship_have_weight"],
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

    
 
