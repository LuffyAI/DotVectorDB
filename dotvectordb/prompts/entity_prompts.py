"""
Author: Larnell Moore
Purpose: This file contains the prompt templates for extracting entities from each text chunk in the database. The prompts are designed to guide the LLM during this extraction during graph construction. 
"""
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

class Psuedoquery(BaseModel):
    Questions: List[str]

class Entity(BaseModel):
    entity_name: str
    entity_groups: List[int]
    entity_description: str
    entity_questions: List[str]

class ExtractedEntitiesResponse(BaseModel):
    entities: List[Entity]

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

ENTITY_RECOGNITION_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for extracting meaningful named entities from the `<Input>` section and then assigning each entity one or more types from the `<Entity_Type>` section.
</Role>

<Instructions>
**Named Entity Extraction:**
   - Entity discovery and entity grouping are separate decisions. First identify all meaningful entities in the input without using the available entity types to restrict extraction. After an entity has been identified, assign it one or more entity types. If none of the specific entity types apply, assign `other`. Never omit an otherwise meaningful entity because its type is uncertain or because it does not fit a specific type.
   - Carefully read the '<Entity_Type>' section only after discovering the entities so you can group them correctly. Do not invent any entity types.
   - Identify all clearly defined and meaningful entities in the `<Input>` section. Extract as many as you can. Each entity can belong to any number of entity types, and you can use a combination of types to make up for the limited number of types. Do not extract the same entity multiple times. Ensure that the extracted entities are **unique** and **relevant** to the context of the input text.
    - For each entity, extract:
        - `entity_name`: The exact name of the entity. Ensure **consistent naming** across the entire extraction process.
        - 'entity_type': Categorize the entity using the type guidance provided in the `<Entity_Type>` section. Use any numbers of types needed to fully describe the entity. Additionally, use the number associated with the type. 
        - 'entity_description': Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
        - 'entity_questions': Generate a list of at least one question that could be asked and answered about the entity based on the information in the input text. The question should be relevant to the entity and its context within the input text. Do not include questions that are not already answerable based on the information in the input text.
    - Look at the `<Examples>` section for examples of how to extract entities. Use these examples as a guide for your extraction process.

**Language & Style:**
   - Ensure all entity names and descriptions are written in the **third person**. 
</Instructions>

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
</Examples>
"""

ALLOWED_ENTITY_TYPE_PAIRS = {
    frozenset(("currency", "tech")),
    frozenset(("currency", "concept")),
    frozenset(("tech", "concept")),
    frozenset(("tech", "language")),
    frozenset(("SCHOLARLY_WORK", "tech")),
    frozenset(("language", "concept")),
}

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
      "entity_description": "Elon Musk is an influential figure in the AI industry, the CEO of Tesla, Inc., and the founder and CEO of SpaceX.",
      "entity_questions": ["Who is the CEO of Tesla, Inc.?", "Who founded SpaceX?"]
    },
    {
      "entity_name": "Sam Altman",
      "entity_groups": [1, 11],
      "entity_description": "Sam Altman is an influential figure in the AI industry and the CEO of OpenAI.",
      "entity_questions": ["Who is the CEO of OpenAI?"]
    },
    {
      "entity_name": "SpaceX",
      "entity_groups": [2, 11],
      "entity_description": "SpaceX is a private aerospace manufacturer and space transportation company founded and led by Elon Musk.",
      "entity_questions": ["What kind of company is SpaceX?", "Who founded SpaceX?"]
    },
    {
      "entity_name": "OpenAI",
      "entity_groups": [2, 11],
      "entity_description": "OpenAI is an artificial intelligence research lab led by Sam Altman that aims to ensure artificial general intelligence benefits humanity.",
      "entity_questions": ["What is OpenAI?", "What does OpenAI aim to ensure?"]
    },
    {
      "entity_name": "Tesla, Inc.",
      "entity_groups": [2, 11],
      "entity_description": "Tesla, Inc. is a company led by Elon Musk and known for electric vehicles and clean energy solutions.",
      "entity_questions": ["What is Tesla, Inc. known for?", "Who is the CEO of Tesla, Inc.?"]
    },
    {
      "entity_name": "Bitcoin",
      "entity_groups": [11, 15],
      "entity_description": "Bitcoin is a decentralized digital currency implemented using blockchain technology.",
      "entity_questions": ["What is Bitcoin?", "What technology is Bitcoin implemented with?"]
    },
    {
      "entity_name": "Project Aurora",
      "entity_groups": [16],
      "entity_description": "Project Aurora is the internal codename for an initiative coordinating the work described in the text.",
      "entity_questions": ["What is Project Aurora?", "What does Project Aurora coordinate?"]
    }
  ]
}
"""
