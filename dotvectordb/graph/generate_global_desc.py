import asyncio
import json
from dotvectordb.prompts.global_desc_prompt import (
    GLOBAL_DESCRIPTION_PROMPT,
    GLOBAL_TYPE_PROMPT,
)


class GlobalDescriptionGenerator:
    """Generate and persist a corpus-wide description for each source."""

    def __init__(self, conn, lock, language_model, verbose=False):
        self.conn = conn
        self.lock = lock
        self.language_model = language_model
        self.verbose = verbose

    ##########################################################################
    #       Fetch Source and Entity Data For Global Descriptions             #
    ##########################################################################
    async def get_sources(self):
        async with self.lock:
            cursor = await self.conn.execute("SELECT id, name FROM source")
            rows = await cursor.fetchall()
            await cursor.close()
        sources = [{"source_id": row[0], "source_name": row[1]} for row in rows]
        if self.verbose:
            print(f"Found {len(sources)} source(s) for global description generation.")
        return sources

    async def get_entity_types_for_source(self, source_id):
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT DISTINCT e.entity_type, et.type_description
                FROM entity AS e
                JOIN entity_types AS et
                    ON et.entity_type = e.entity_type
                JOIN chunk_to_entity_mapping AS cem
                    ON cem.entity_id = e.entity_id
                JOIN chunk AS c
                    ON c.chunk_id = cem.chunk_id
                WHERE c.source_id = ?
                ORDER BY e.entity_type
                """,
                (source_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        entity_types = [
            {"entity_type": row[0], "type_description": row[1]}
            for row in rows
        ]
        if self.verbose:
            print(
                f"Source {source_id} has {len(entity_types)} entity type(s): "
                f"{', '.join(item['entity_type'] for item in entity_types)}"
            )
        return entity_types

    async def get_random_entity_sample(self, source_id, entity_type, sample_size=5):
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")

        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT DISTINCT
                    e.entity_id,
                    e.entity_name,
                    e.entity_type,
                    e.entity_description
                FROM entity AS e
                JOIN chunk_to_entity_mapping AS cem
                    ON cem.entity_id = e.entity_id
                JOIN chunk AS c
                    ON c.chunk_id = cem.chunk_id
                WHERE c.source_id = ? AND e.entity_type = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (source_id, entity_type, sample_size),
            )
            rows = await cursor.fetchall()
            await cursor.close()

        sample = [
            {
                "entity_name": row[1],
                "entity_type": row[2],
                "entity_description": row[3],
            }
            for row in rows
        ]
        if self.verbose:
            print(
                f"Sampled {len(sample)} {entity_type} entity/entities from source "
                f"{source_id}."
            )
        return sample

    ##########################################################################
    #        Prepare and Generate Entity-Type Descriptions                   #
    ##########################################################################
    @staticmethod
    def _get_answer(response):
        result = response[0] if isinstance(response, tuple) else response
        if isinstance(result, dict):
            answer = result.get("answer")
            if isinstance(answer, str):
                return answer
        if isinstance(result, str):
            return result
        raise ValueError("The language model did not return a description")

    async def generate_type_description(
        self, source_id, source_name, entity_type, type_description, sample_size=5
    ):
        sample = await self.get_random_entity_sample(
            source_id=source_id,
            entity_type=entity_type,
            sample_size=sample_size,
        )
        if not sample:
            return None

        type_name = entity_type
        if type_description:
            type_name = f"{entity_type}: {type_description}"

        prompt = GLOBAL_TYPE_PROMPT.format(
            source_name=source_name,
            type_name=type_name,
            sample=json.dumps(sample, ensure_ascii=False, indent=2),
        )
        if self.verbose:
            print(f"Prompt sent to the LLM for entity type '{entity_type}':\n{prompt}")
        response = await self.language_model(prompt)
        description = self._get_answer(response)
        if self.verbose:
            print(f"Generated description for entity type '{entity_type}'.")
        return description

    ##########################################################################
    #              Save Global Descriptions to Source                        #
    ##########################################################################
    async def save_source_description(self, source_id, description):
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT description FROM source WHERE id = ?", (source_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                raise ValueError(f"Source '{source_id}' does not exist")

            source_description = {}
            if row[0]:
                try:
                    existing_description = json.loads(row[0])
                    if isinstance(existing_description, dict):
                        source_description.update(existing_description)
                except (TypeError, json.JSONDecodeError):
                    source_description["metadata"] = row[0]

            source_description.update(description)
            await self.conn.execute(
                "UPDATE source SET description = ? WHERE id = ?",
                (json.dumps(source_description, ensure_ascii=False), source_id),
            )
            await self.conn.commit()
        if self.verbose:
            print(f"Saved global description to source {source_id}.")

    ##########################################################################
    #       Global Description Generation Operations For Each Source         #
    ##########################################################################
    async def generate_for_source(self, source_id, source_name, sample_size=5):
        entity_types = await self.get_entity_types_for_source(source_id)
        if not entity_types:
            if self.verbose:
                print(f"Skipping source {source_id}: no mapped entities were found.")
            return None

        if self.verbose:
            print(
                f"Sending {len(entity_types)} entity-type description request(s) "
                "concurrently."
            )
        descriptions = await asyncio.gather(
            *(
                self.generate_type_description(
                    source_id=source_id,
                    source_name=source_name,
                    entity_type=entity_type["entity_type"],
                    type_description=entity_type["type_description"],
                    sample_size=sample_size,
                )
                for entity_type in entity_types
            )
        )

        type_descriptions = {}
        for entity_type, description in zip(entity_types, descriptions):
            if description:
                type_descriptions[entity_type["entity_type"]] = description

        if not type_descriptions:
            if self.verbose:
                print(f"Skipping source {source_id}: no type descriptions were generated.")
            return None

        if self.verbose:
            print(
                f"All entity-type descriptions finished for source {source_id}. "
                "Generating the final global description."
            )
        prompt = GLOBAL_DESCRIPTION_PROMPT.format(
            source_name=source_name,
            global_descriptions=json.dumps(
                type_descriptions, ensure_ascii=False, indent=2
            )
        )
        if self.verbose:
            print(
                f"Final global-description prompt sent to the LLM for "
                f"'{source_name}':\n{prompt}"
            )
        response = await self.language_model(prompt)
        global_description = self._get_answer(response)
        result = {
            "global_description": global_description,
            "entity_type_descriptions": type_descriptions,
        }
        await self.save_source_description(source_id, result)
        return result

    ##########################################################################
    #          Generate Global Descriptions For All Sources                  #
    ##########################################################################
    async def generate(self, sample_size=5, verbose=None):
        if verbose is not None:
            self.verbose = verbose
        results = {}
        for source in await self.get_sources():
            description = await self.generate_for_source(
                source_id=source["source_id"],
                source_name=source["source_name"],
                sample_size=sample_size,
            )
            if description is not None:
                results[source["source_id"]] = description
                if self.verbose:
                    print(
                        f"Generated global description for {source['source_name']}."
                    )
        return results
