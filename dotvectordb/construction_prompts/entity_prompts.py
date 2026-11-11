"""
Author: Larnell Moore
Purpose: This file contains the prompt templates for extracting entities from each text chunk in the database. The prompts are designed to guide the LLM during this extraction during graph construction. 
Notes: Some prompt templates were inspired/derived from the LightRAG project
"""
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, create_model
######################################
# Entity Extraction Prompt Model     #
######################################
class Entity(BaseModel):
    entity_name: str
    entity_groups: List[int]
    entity_description: str
class ExtractedEntitiesResponse(BaseModel):
    entities: List[Entity]
########################################
# Entity Summarization Prompt Model    #
########################################
class EntitySummarizer(BaseModel):
    new_description: str
###########################################
# Custom Entity Enrichment Prompt Model   #
###########################################
async def create_entity_enrichment_model(
    needs_descriptions: bool,
    needs_types: bool,
) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic model for entity enrichment based on the specified needs.

    Args:
        needs_descriptions (bool): Whether entity descriptions are needed.
        needs_types (bool): Whether entity types are needed.

    Returns:
        BaseModel: A dynamically created Pydantic model for entity enrichment.
    """
    fields: Dict[str, Tuple[Any, Any]] = {
        "local_entity_id": (
            int,
            Field(
                ...,
                description=(
                    "The temporary integer ID supplied for this entity. "
                    "Return it unchanged."
                ),
            ),
        )
    }
    if needs_descriptions:
        fields['entity_description'] = (str, Field(..., description="A concise yet comprehensive description of the entity's attributes and activities."))
    if needs_types:
        fields['entity_groups'] = (List[int], Field(..., description="A list of integers representing the types or categories the entity belongs to."))
    entity_model = create_model("EntityEnrichmentItem", **fields)
    return create_model(
        "EntityEnrichmentResponse",
        entities=(List[entity_model], Field(...)),
    )
###########################################
# Entity Prompt Graph Helper Functions    #
###########################################
async def isEntityTypesFormatCorrect(entity_types):
    """
    Validates the format of the entity_types dictionary.

    Args:
        entity_types (dict): A dictionary where keys are entity type names and values are their descriptions.

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    if not isinstance(entity_types, dict):
        return False

    for key, value in entity_types.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return False
    return True 

async def prepare_entity_enrichment_prompt_sections(
    needs_descriptions: bool,
    needs_types: bool,
) -> Tuple[str, type[BaseModel]]:
    """
    Prepares the sections for the entity enrichment prompt based on the specified needs.

    Args:
        needs_descriptions (bool): Whether entity descriptions are needed.
        needs_types (bool): Whether entity types are needed.

    Returns:
        Tuple[str, type[BaseModel]]: The assembled enrichment prompt and the
        matching Pydantic response model.
    """
    requested_sections = []
    if needs_descriptions:
        requested_sections.append(ENTITY_ENRICHMENT_DESCRIPTION_SECTION)
    if needs_types:
        requested_sections.append(ENTITY_ENRICHMENT_ENTITY_TYPES_SECTION)

    if not requested_sections:
        raise ValueError("At least one entity enrichment field must be requested.")

    numbered_sections = [
        section.format(task_number=task_number)
        for task_number, section in enumerate(requested_sections, start=1)
    ]
    enrichment_prompt = "\n".join(
        [ENTITY_ENRICHMENT_HEADER_SECTION, *numbered_sections, ENTITY_ENRICHMENT_FOOTER_SECTION]
    )
    enrichment_model = await create_entity_enrichment_model(
        needs_descriptions=needs_descriptions,
        needs_types=needs_types,
    )

    return enrichment_prompt, enrichment_model
###########################################
# Entity Prompt Templates                 #
###########################################
ENTITY_RECOGNITION_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for extracting meaningful named entities from the `<Input>` section and then assigning each entity one or more types from the `<Entity_Type>` section.
</Role>

<Instructions>
**Named Entity Extraction:**
   - Entity discovery and entity grouping are separate decisions. First, identify all meaningful entities in the input without using the available entity types to restrict extraction. After identifying an entity, assign it one or more entity types. If none of the specific entity types apply, assign `other`. Never omit an otherwise meaningful entity because its type is uncertain or because it does not fit a specific type.
   - Carefully read the '<Entity_Type>' section only after discovering the entities so you can group them correctly. Do not invent any entity types.
   - Identify all clearly defined and meaningful entities in the `<Input>` section. Extract as many as you can. Each entity can belong to multiple entity types, and you can combine types to make up for the limited number of types. Do not extract the same entity multiple times. Ensure that the extracted entities are **unique** and **relevant** to the context of the input text.
    - For each entity, extract:
        - `entity_name`: The exact name of the entity. Ensure **consistent naming** across the entire extraction process.
        - 'entity_type': Categorize the entity using the type guidance provided in the `<Entity_Type>` section. Use any number of types needed to describe the entity fully. Also include the number associated with each type. 
        - 'entity_description': Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    - Look at the `<Examples>` section for examples of how to extract entities. Use these examples to guide your extraction process.
Language & Style:
* Write all entity names and descriptions in the third person.
* Write all text in the following language: {language}. If a proper, widely accepted translation is not available or would cause ambiguity, retain proper nouns (e.g., personal names, place names, organization names) in their original language.

<Entity_Types>
{entity_types}
</Entity_Types>

<Input>
{chunk}
</Input>

<Examples>
<Input_Example>
{input_example}
</Input_Example>

<Output_Example>
{output_example}
</Output_Example>
"""

ENTITY_SUMMARIZATION_PROMPT = """
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity at the beginning of the summary to ensure immediate clarity and context.
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

# If a custom knowledge graph is being used and the user did not provide a description for one or more of the following, the required values will be filled in by the language model.
ENTITY_ENRICHMENT_HEADER_SECTION = """
<Role>
You are a Knowledge Graph Specialist responsible for carefully reading the information in the `<Input>` and enriching each of the specified entities in the '<Entities>' section by completing the following tasks:
</Role>

<Tasks>
For every entity, return its supplied `local_entity_id` unchanged so the
generated information can be matched to the correct entity. Return exactly
one result for every supplied entity; do not omit or duplicate entities.
"""
ENTITY_ENRICHMENT_DESCRIPTION_SECTION= """ 
**Task {task_number} - Entity Description Generation:**
- Carefully read the `<Input>` section to understand the context of the entities listed in the '<Entities>' section.
- For each entity, generate a concise yet comprehensive description of the entity's attributes and activities based solely on the information present in the `<Input>` section. The description should be written in the third person and should not include any information that is not explicitly stated in the `<Input>` section. Never omit any of the specified entities from your response. 
"""
ENTITY_ENRICHMENT_ENTITY_TYPES_SECTION = """
**Task {task_number} - Entity Type Assignment:**
 - For each entity, assign one or more types from the '<Entity_Types>' section. If none of the specific entity types apply, assign `other`.
 """

ENTITY_ENRICHMENT_FOOTER_SECTION = """
</Tasks>
- Look at the `<Examples>` section for reference. Use these examples as guidance.
Language & Style:
* Write all entity names and descriptions in the third person.
* Write all text in the following language: {language}. If a proper, widely accepted translation is not available or would cause ambiguity, retain proper nouns (e.g., personal names, place names, organization names) in their original language.

<Entity_Types>
{entity_types}
</Entity_Types>

<Input>
{chunk}
</Input>

<Entities>
{entities}
</Entities>

<Examples>
<Input_Example>
{input_example}
</Input_Example>

<Output_Example>
{output_example}
</Output_Example>
"""


###########################################
# Entity Prompt Defaults & Constants      #
###########################################
DEFAULT_ENTITY_TYPES = {
    "individual": "A named individual being, fictional or non-fictional, including humans, animals, creatures, monsters, and aliens, regardless of whether they are alive, deceased, historical, or imaginary.",
    "organization": "A group or institution of fictional or non-fictional people.",
    "location": "A fictional or non-fictional named place or geographic area, such as a city, country, or landmark",
    "substance": "A named edible substance, drug, element, or chemical compound",
    "event": "A planned, organized occasion or occurrence, such as a festival, war, or historical event",
    "WORK_OF_ART": "A named non-academic focused artistic or literary creation, such as a book, movie, album, game, song, article, or painting",
    "SCHOLARLY_WORK": "A named academic or scientific publication, such as a research paper, thesis, or journal article",
    "LEGAL_WORK": "A named legal document, such as a law, regulation, or court case",
    "time": "A named date or year",
    "medical_condition": "A named disease, disorder, syndrome, injury, or symptom",
    "tech": "A named AI model, non-videogame computer application, device, or operating system such as GPT4o, Photoshop, PS5, and Android",
    "object": "A nontechology named inanimate object or item that can be held or used by an individual to assist in a task, such as a hammer or gun",
    "concept": "A named idea, theory, framework, or principle, such as relativity, democracy, RAG, or the butterfly effect",
    "language": "A named system of communication or formal notation, such as human or programming language",
    "currency": "A named fictional or non-fictional system of money or medium of exchange, such as the US dollar, Bitcoin, or the Galactic Credit Standard",
    "other": "A named entity that does not fit into any of the other categories.",
}

INPUT_EXAMPLE = """
Elon Musk and Sam Altman are both influential figures in the AI industry. Elon Musk is the CEO of Tesla, Inc., a company known for its electric vehicles and clean energy solutions. He is also the founder and CEO of SpaceX, a private aerospace manufacturer and space transportation company. Sam Altman is the CEO of OpenAI, an artificial intelligence research lab that aims to ensure that artificial general intelligence benefits all of humanity. Bitcoin is a decentralized digital currency implemented using blockchain technology. Project Aurora is the internal codename for an initiative coordinating this work.
"""

OUTPUT_EXAMPLE = """
{
  "entities": [
    {
      "entity_name": "Elon Musk",
      "entity_groups": [1, 11],
      "entity_description": "Elon Musk is an influential figure in the AI industry, the CEO of Tesla, Inc., and the founder and CEO of SpaceX."
    },
    {
      "entity_name": "Sam Altman",
      "entity_groups": [1, 11],
      "entity_description": "Sam Altman is an influential figure in the AI industry and the CEO of OpenAI."
    },
    {
      "entity_name": "SpaceX",
      "entity_groups": [2, 11],
      "entity_description": "SpaceX is a private aerospace manufacturer and space transportation company founded and led by Elon Musk."
    },
    {
      "entity_name": "OpenAI",
      "entity_groups": [2, 11],
      "entity_description": "OpenAI is an artificial intelligence research lab led by Sam Altman that aims to ensure artificial general intelligence benefits humanity."
    },
    {
      "entity_name": "Tesla, Inc.",
      "entity_groups": [2, 11],
      "entity_description": "Tesla, Inc. is a company led by Elon Musk and known for electric vehicles and clean energy solutions."
    },
    {
      "entity_name": "Bitcoin",
      "entity_groups": [11, 15],
      "entity_description": "Bitcoin is a decentralized digital currency implemented using blockchain technology."
    },
    {
      "entity_name": "Project Aurora",
      "entity_groups": [16],
      "entity_description": "Project Aurora is the internal codename for an initiative coordinating the work described in the text."
    }
  ]
}
"""
