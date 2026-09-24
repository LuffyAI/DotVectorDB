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
    relation_questions: List[str]
    keywords: List[str]
    weight: int
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
    needs_questions: bool,
    needs_keywords: bool,
    needs_weights: bool,
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
    if needs_questions:
        fields["questions"] = (
            List[str],
            Field(
                ...,
                description=(
                    "Questions that can be answered using information about "
                    "the relationship."
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
    if needs_weights:
        fields["weight"] = (
            int,
            Field(
                ...,
                ge=1,
                le=10,
                description=(
                    "The relationship's significance from 1 to 10."
                ),
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
    needs_questions: bool,
    needs_keywords: bool,
    needs_weights: bool,
) -> Tuple[str, type[BaseModel]]:
    """Build a prompt and matching response model for an enrichment group."""
    requested_sections = []
    if needs_descriptions:
        requested_sections.append(
            RELATIONSHIP_ENRICHMENT_DESCRIPTION_SECTION
        )
    if needs_questions:
        requested_sections.append(
            RELATIONSHIP_ENRICHMENT_QUESTIONS_SECTION
        )
    if needs_keywords:
        requested_sections.append(
            RELATIONSHIP_ENRICHMENT_KEYWORDS_SECTION
        )
    if needs_weights:
        requested_sections.append(
            RELATIONSHIP_ENRICHMENT_WEIGHT_SECTION
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
        needs_questions=needs_questions,
        needs_keywords=needs_keywords,
        needs_weights=needs_weights,
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
        - 'description': A concise yet comprehensive description of the relationship, based *solely* on the information present in the input text.
        - 'relation_questions': Generate a list of at least one question that could be asked and answered about the relationship based on the information in the input text. The question should be relevant to the relationship and its context within the input text. For example, if the relationship is "Alice is Bob's sister", a relevant question could be "Who is Bob's sister?"
        - 'keywords': Generate a list of keywords that are relevant to the relationship and its context within the input text. The keywords should be concise and capture the essence of the relationship. For example, if the relationship is "Alice is Bob's sister", relevant keywords could be "sibling", "family".
        - 'weight': Assign a numeric score to the relationship based on its significance and relevance within the context of the input text. The weight should be an integer value between 1-10, with higher values indicating greater significance. For example, if the relationship is central to the main topic of the input text, it may receive a higher weight, while a less significant relationship may receive a lower weight.
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

RELATIONSHIP_ENRICHMENT_QUESTIONS_SECTION = """
**Task {task_number} - Relationship Question Generation:**
- For each relationship, generate a list containing at least one question that
  could reasonably be asked and answered about the relationship based on the
  information in the `<Input>` section.
- Each question must be relevant to the relationship and its context in the
  `<Input>` section. Do not include questions that cannot already be answered
  using the supplied information.
- For example, if the relationship is "Alice is Bob's sister," a relevant
  question is "Who is Bob's sister?"
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

RELATIONSHIP_ENRICHMENT_WEIGHT_SECTION = """
**Task {task_number} - Relationship Weight Assignment:**
- For each relationship, assign an integer weight from 1 to 10 based on its
  significance and relevance within the context of the `<Input>` section.
- Higher values indicate greater significance. A relationship that is central
  to the main topic should receive a higher weight, while a less significant
  relationship should receive a lower weight.
- Base the weight only on the supplied `<Input>` and do not infer significance
  from outside knowledge.
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
Elon Musk and Sam Altman are both influential figures in the AI industry. Elon Musk is the CEO of Tesla, Inc., a company known for its electric vehicles and clean energy solutions. He is also the founder and CEO of SpaceX, a private aerospace manufacturer and space transportation company. Sam Altman is the CEO of OpenAI, an artificial intelligence research lab that aims to ensure that artificial general intelligence benefits all of humanity.
"""

OUTPUT_EXAMPLE = """
[{
src_entity_id: 1,
dst_entity_id: 2,
description: "Elon Musk is the CEO of Tesla, Inc., a company known for its electric vehicles and clean energy solutions.",
Question: ["What is the CEO of Tesla?", "How does Elon Musk promote sustainability?"]
},
{
src_entity_id: 1,
dst_entity_id: 3,
description: "Elon Musk is the founder and CEO of SpaceX, a private aerospace manufacturer and space transportation company.",
Question: ["Who is the founder and CEO at SpaceX?", "What company owned by Elon Musk helps with space transportation?"]
},
{
src_entity_id: 5,
dst_entity_id: 4,
description: "Sam Altman is the CEO of OpenAI, an artificial intelligence research lab that aims to ensure that artificial general intelligence benefits all of humanity.",
Question: ["Who is the CEO of OpenAI?", "What is the goal of OpenAI?"]
}]
"""
