"""
Author: Larnell Moore
Purpose: This file contains the prompt templates for extracting relationships between entities from each text chunk in the database. The prompts are designed to guide the LLM during this extraction during graph construction. 
"""
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from pydantic import create_model
############################################
# Relationship Extraction Prompt Model     #
############################################
class Relationship(BaseModel):
    src_entity_id: int
    dst_entity_id: int
    description: str
    keywords: List[str]
    supporting_lines: List[str]
class Relationships(BaseModel):
    relations: List[Relationship]
#############################################
# Relationship Summarization Prompt Model   #
#############################################
class RelationshipSummarizer(BaseModel):
    new_description: str
#############################################
# Relationship Helper Functions             #
#############################################
async def create_relationship_enrichment_model(
    needs_descriptions: bool,
    needs_keywords: bool,
) -> type[BaseModel]:
    """Create the response model required by one enrichment group."""
    fields: Dict[str, Tuple[Any, Any]] = {
        "local_relationship_id": (
            int,
            Field(
                ...,
                description=(
                    "The temporary integer ID supplied for this relationship. "
                    "Return it unchanged."
                ),
            ),
        )
    }
    if needs_descriptions:
        fields["relation_description"] = (
            str,
            Field(
                ...,
                description=(
                    "A concise yet comprehensive description of the "
                    "relationship."
                ),
            ),
        )
    if needs_keywords:
        fields["keywords"] = (
            List[str],
            Field(
                ...,
                description="Keywords that characterize the relationship.",
            ),
        )
    relationship_model = create_model(
        "RelationshipEnrichmentItem",
        **fields,
    )
    return create_model(
        "RelationshipEnrichmentResponse",
        relationships=(List[relationship_model], Field(...)),
    )


async def prepare_relationship_enrichment_prompt_sections(
    needs_descriptions: bool,
    needs_keywords: bool,
) -> Tuple[str, type[BaseModel]]:
    """Build a prompt and matching response model for an enrichment group."""
    requested_sections = []
    if needs_descriptions:
        requested_sections.append(
            RELATIONSHIP_ENRICHMENT_DESCRIPTION_SECTION
        )
    if needs_keywords:
        requested_sections.append(
            RELATIONSHIP_ENRICHMENT_KEYWORDS_SECTION
        )
    if not requested_sections:
        raise ValueError(
            "At least one relationship enrichment field must be requested."
        )

    numbered_sections = [
        section.format(task_number=task_number)
        for task_number, section in enumerate(requested_sections, start=1)
    ]
    enrichment_prompt = "\n".join(
        [
            RELATIONSHIP_ENRICHMENT_HEADER_SECTION,
            *numbered_sections,
            RELATIONSHIP_ENRICHMENT_FOOTER_SECTION,
        ]
    )
    enrichment_model = await create_relationship_enrichment_model(
        needs_descriptions=needs_descriptions,
        needs_keywords=needs_keywords,
    )
    return enrichment_prompt, enrichment_model


###########################################
# Relationship Prompt Templates           #
###########################################
RELATIONSHIP_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for reading a list of entities from the '<Entities> section and describing the relationships observed between each entity based on the '<Input>' section.
</Role>

<Instructions>
**Relationship Extraction:**
   - Carefully read the '<Entities>' section to understand entities to focus on. Do not extract any additional entities from '<Input>'.
   - Identify all relationships clearly defined between the identified entities in the `<Input>` section. Extract as many as you can. Do not invent any relationships between entities that are not supported by the information in '<Input>'.
    - For each relationship, extract:
        - `src_entity_id`: The ID of the source entity as seen in the '<Entities>' section.
        - `dst_entity_id`: The ID of the destination entity as seen in the '<Entities>' section.
        - `supporting_lines:` A list of line numbers from the `<Input>` section you believe support the relationship.  If no evidence is present, return an empty list.
        - 'description': A concise yet comprehensive description of the relationship, based *solely* on the information present in the input text.
        - 'keywords': Generate a list of keywords that are relevant to the relationship and its context within the input text. The keywords should be concise and capture the essence of the relationship. For example, if the relationship is "Alice is Bob's sister", relevant keywords could be "sibling", "family".
Language & Style:
* Write all descriptions in the third person.
* Write all text in the following language: {language}. If a proper, widely accepted translation is not available or would cause ambiguity, retain proper nouns (e.g., personal names, place names, organization names) in their original language.
<Entities>
{entities}
</Entities>

<Input>
{chunk}
</Input>

<Examples>
<Entity_Example>
{entity_example}
</Entity_Example>

<Input_Example>
{input_example}
</Input_Example>

<Output_Example>
{output_example}
</Output_Example>
"""

RELATIONSHIP_SUMMARIZATION_PROMPT = """
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given relationship into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full names of the two entities involved in the relationship to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities that share the same name.
  - If distinct entities are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint: The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:
{description_list}
"""

RELATIONSHIP_ENRICHMENT_HEADER_SECTION = """
<Role>
You are a Knowledge Graph Specialist responsible for enriching the specified
relationships using only the information in the `<Input>` section.
</Role>

<Tasks>
For every relationship, return its supplied `local_relationship_id` unchanged
so the generated information can be matched to the correct relationship.
Return exactly one result for every supplied relationship; do not omit or
duplicate relationships.
"""

RELATIONSHIP_ENRICHMENT_DESCRIPTION_SECTION = """
**Task {task_number} - Relationship Description Generation:**
- Carefully read the `<Entities>` section to understand the source and
  destination entities referenced by each relationship. Do not introduce any
  additional entities that are not listed there.
- For each relationship, generate a concise yet comprehensive description of
  the relationship based solely on information explicitly present in the
  `<Input>` section.
- Describe how the source entity relates to the destination entity clearly and
  objectively. Write the description in the third person and do not invent any
  unsupported facts or relationships.
- Never omit any of the specified relationships from the response.
"""

RELATIONSHIP_ENRICHMENT_KEYWORDS_SECTION = """
**Task {task_number} - Relationship Keyword Generation:**
- For each relationship, generate a list of keywords that are relevant to the
  relationship and its context in the `<Input>` section.
- Keywords must be concise and capture the essential nature of the
  relationship without introducing information that is not supported by the
  `<Input>` section.
- For example, if the relationship is "Alice is Bob's sister," relevant
  keywords include "sibling" and "family."
- Never omit any of the specified relationships from the response.
"""

RELATIONSHIP_ENRICHMENT_FOOTER_SECTION = """
</Tasks>

Language & Style:
* Write descriptions in the third person.
* Write all generated text in {language}. Retain proper nouns in their original
  language when translation would introduce ambiguity.

<Entities>
{entities}
</Entities>

<Relationships>
{relationships}
</Relationships>

<Input>
{chunk}
</Input>

<Examples>
<Entity_Example>
{entity_example}
</Entity_Example>

<Input_Example>
{input_example}
</Input_Example>

<Output_Example>
{output_example}
</Output_Example>
"""


ENTITY_EXAMPLE = """
1: Elon Musk
2: Tesla, Inc.
3: SpaceX
4: OpenAI
5: Sam Altman
"""

INPUT_EXAMPLE = """
[L1] Elon Musk and Sam Altman are both influential figures in the AI industry. [L2] Elon Musk is the CEO of Tesla, Inc., a company known for its self-driving electric vehicles and clean energy solutions. [L3] He is also the founder and CEO of SpaceX, a private aerospace manufacturer and space transportation company. [L4] Sam Altman is the CEO of OpenAI, an artificial intelligence research lab that works on large language models to benefit all of humanity.
"""

OUTPUT_EXAMPLE = """
[{
src_entity_id: 1,
dst_entity_id: 2,
description: "Elon Musk is the CEO of Tesla, Inc., a company known for its electric vehicles and clean energy solutions.",
supporting_lines: ["L2"],
keywords: ["chief executive officer", "electric vehicles", "clean energy"]
},
{
src_entity_id: 1,
dst_entity_id: 3,
description: "Elon Musk is the founder and CEO of SpaceX, a private aerospace manufacturer and space transportation company.",
supporting_lines: ["L3"],
keywords: ["founder", "aerospace", "space transportation"]
},
{
src_entity_id: 5,
dst_entity_id: 4,
supporting_lines: ["L4"],
description: "Sam Altman is the CEO of OpenAI, an artificial intelligence research lab that aims to ensure that artificial general intelligence benefits all of humanity working on large language models.",
keywords: ["chief executive officer", "artificial intelligence", "research"]
}]
src_entity_id: 1,
dst_entity_id: 5,
description: "Elon Musk and Sam Altman are both influential figures in the AI industry with Elon focusing on areas such as self-driving cars and Sam Altman on large language models",
supporting_lines: ["L1", "L2", "L4"],
"""
