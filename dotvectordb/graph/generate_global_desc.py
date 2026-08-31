import asyncio
import json
from tqdm.asyncio import tqdm_asyncio
from dotvectordb.prompts.global_desc_prompt import (
    GLOBAL_CHUNK_PROMPT,
    GLOBAL_DESCRIPTION_PROMPT,
    GLOBAL_TYPE_PROMPT,
)

class GlobalDescriptionGenerator:
    def __init__(self, conn, lock, language_model, entity_types, sources,
                 global_config, save_global_description_func, verbose=False,
                 debug_mode=False):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.entity_types = entity_types
        self.sources = sources
        self.global_config = global_config
        self.save_global_description_func = save_global_description_func
        self.verbose = verbose
        self.debug_mode = debug_mode

    ##########################################################################
    #       Fetch Entity Data For Global Description Generation             #
    ##########################################################################
    async def get_random_entity_sample(self, entity_type, sample_size=5):
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT *
                FROM entity
                WHERE entity_type = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (entity_type, sample_size)
            )
            rows = await cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            await cursor.close()
            entities = [
                dict(zip(column_names, row))
                for row in rows
            ]
        if self.verbose:
            print(f"Sampled {len(entities)} {entity_type} entity/entities.")
        return entities

    ##########################################################################
    #       Fetch Chunk Data For Global Description Generation             #
    ##########################################################################
    async def get_random_chunk_sample(self, sample_size=5):
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT source.name, chunk.content
                FROM chunk
                JOIN source ON source.id = chunk.source_id
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (sample_size,)
            )
            rows = await cursor.fetchall()
            await cursor.close()
            chunks = [
                {"source_name": row[0], "content": row[1]}
                for row in rows
            ]
        if self.verbose:
            print(f"Sampled {len(chunks)} text chunk(s).")
        return chunks

    ##########################################################################
    #        Prepare and Generate Entity-Type Descriptions                  #
    ##########################################################################
    @staticmethod
    def get_answer(response):
        result = response[0] if isinstance(response, tuple) else response
        if isinstance(result, dict):
            answer = result.get("answer")
            if isinstance(answer, str):
                return answer
        if isinstance(result, str):
            return result
        raise ValueError("The language model did not return a description")

    async def generate_type_description(self, entity_type, type_description,
                                        source_names, sample_size=5):
        sample = await self.get_random_entity_sample(
            entity_type=entity_type,
            sample_size=sample_size,
        )
        if not sample:
            return None
        type_name = entity_type
        if type_description:
            type_name = f"{entity_type}: {type_description}"
        prompt = GLOBAL_TYPE_PROMPT.format(
            source_names=json.dumps(source_names, ensure_ascii=False),
            type_name=type_name,
            sample=json.dumps(sample, ensure_ascii=False, indent=2),
        )
        if self.verbose:
            print(f"Prompt sent to the LLM for entity type '{entity_type}':\n{prompt}")
        response = await self.language_model(prompt)
        description = self.get_answer(response)
        if self.verbose:
            print(f"Generated description for entity type '{entity_type}'.")
        return description

    async def generate_chunk_description(self, source_names, sample_size=5):
        sample = await self.get_random_chunk_sample(sample_size=sample_size)
        if not sample:
            return None
        prompt = GLOBAL_CHUNK_PROMPT.format(
            source_names=json.dumps(source_names, ensure_ascii=False),
            sample=json.dumps(sample, ensure_ascii=False, indent=2),
        )
        if self.verbose:
            print(f"Prompt sent to the LLM for sampled text chunks:\n{prompt}")
        response = await self.language_model(prompt)
        description = self.get_answer(response)
        if self.verbose:
            print("Generated description for sampled text chunks.")
        return description

    ##########################################################################
    #          Generate and Save the Global Graph Description               #
    ##########################################################################
    async def generate(self, sample_size=5):
        current_version = self.global_config["current_version_count"]
        description_version = self.global_config["description_version"]
        description = self.global_config["description"]
        description_is_empty = not description
        if description and isinstance(description, str):
            try:
                description_is_empty = not bool(json.loads(description))
            except json.JSONDecodeError:
                description_is_empty = False
        if (not self.debug_mode and not description_is_empty
                and current_version == description_version):
            if self.verbose:
                print(
                    f"Global description is current at version {current_version}. "
                    "Skipping generation."
                )
            try:
                return json.loads(description)
            except (TypeError, json.JSONDecodeError):
                return description
        source_names = [source["name"] for source in self.sources]
        if self.verbose:
            if self.debug_mode:
                reason = "debug mode is enabled"
            elif description_is_empty:
                reason = "the stored description is empty"
            else:
                reason = (
                    f"description version {description_version} does not match "
                    f"current version {current_version}"
                )
            print(f"Generating a global description because {reason}.")
            print(
                f"Sending {len(self.entity_types)} entity-type description "
                "request(s) and 1 chunk-description request concurrently."
            )
        descriptions = await tqdm_asyncio.gather(
            *(
                self.generate_type_description(
                    entity_type=entity_type,
                    type_description=type_description,
                    source_names=source_names,
                    sample_size=sample_size,
                )
                for entity_type, type_description in self.entity_types.items()
            ),
            self.generate_chunk_description(
                source_names=source_names,
                sample_size=sample_size,
            ),
            desc="Sampling entity types & chunks to infer a global description",
        )
        type_results = descriptions[:-1]
        chunk_description = descriptions[-1]
        type_descriptions = {
            entity_type: description
            for entity_type, description in zip(self.entity_types, type_results)
            if description
        }
        global_descriptions = dict(type_descriptions)
        if chunk_description:
            global_descriptions["sampled_chunks"] = chunk_description
        if not global_descriptions:
            if self.verbose:
                print("No entity-type or chunk descriptions were generated.")
            return None
        prompt = GLOBAL_DESCRIPTION_PROMPT.format(
            source_names=json.dumps(source_names, ensure_ascii=False),
            global_descriptions=json.dumps(
                global_descriptions, ensure_ascii=False, indent=2
            )
        )
        if self.verbose:
            print(f"Final global-description prompt sent to the LLM:\n{prompt}")
        response = await self.language_model(prompt)
        result = {
            "global_description": self.get_answer(response),
            "entity_type_descriptions": type_descriptions,
            "chunk_description": chunk_description,
        }
        await self.save_global_description_func(
            description=result,
            description_version=current_version,
        )
        print(f"Global description generated and saved at version {current_version}.")
        if self.verbose:
            print(
                "Saved the global description to global_config at version "
                f"{current_version}."
            )
        return result
