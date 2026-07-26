from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

class Psuedoquery(BaseModel):
    Questions: List[str]

class Entity(BaseModel):
    entity_name: str
    entity_type: int
    entity_description: str
    entity_questions: Psuedoquery

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
You are a Knowledge Graph Specialist responsible for extracting named entities from the `<Input>` section that satisfies one of the conditions described the <Entity_Type> section.
</Role>

<Instructions>
**Named Entity Extraction:**
   - Carefully read the '<Entity_Type>' section to understand the type of named entities that you are looking for.Do not extract anything that does not meet one of the entity type's condition and do not invent any entity types.
   - Identify all clearly defined and meaningiful entities in the `<Input>` section that match the entity type described in the `<Entity_Type>` section. Extract as many as you can. Each entity can belong to at most one entity type. If an entity can belong to multiple entity types, choose the most specific one.
    - For each entity, extract:
        - `entity_name`: The exact name of the entity. Ensure **consistent naming** across the entire extraction process.
        - 'entity_type': Categorize the entity using the type guidance provided in the `<Entity_Type>` section. Use the most specific type available. Additionally, use the number associated with the type. 
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

DEFAULT_ENTITY_TYPES = {
    "person": "A named fictional or non-fictional individual human",
    "organization": "A group or institution of fictional or non-fictional people.",
    "location": "A fictional or non-fictional named place or geographic area, such as a city, country, or landmark",
    "substance": "A named edible substance, drug, element, or chemical compound",
    "event": "A planned, organized occasion or occurrence, such as a festival, war, or historical event",
    "creative_work": "A named non-academic focused artistic or literary creation, such as a book, movie, album, game, song, article, or painting",
    "scholarly_work": "A named academic or scientific publication, such as a research paper, thesis, or journal article",
    "legal_work": "A named legal document, such as a law, regulation, or court case",
    "time": "A named date or year",
    "medical_condition": "A named disease, disorder, syndrome, injury, or symptom",
    "tech": "A named AI model, non-videogame computer application, device, or operating system such as GPT4o, Photoshop, PS5, and Android",
    "object": "A nontechology named inanimate object or item that can be held or used by an individual to assist in a task, such as a hammer or gun",
    "concept": "A named idea, theory, framework, or principle, such as relativity, democracy, RAG, or the butterfly effect",
    "language": "A named system of communication or formal notation, such as human or programming language",
    "currency": "A named fictional or non-fictional system of money or medium of exchange, such as the US dollar, Bitcoin, or the Galactic Credit Standard",
}

INPUT_EXAMPLE = """
Elon Musk and Sam Altman are both influential figures in the AI industry. Elon Musk is the CEO of Tesla, Inc., a company known for its electric vehicles and clean energy solutions. He is also the founder and CEO of SpaceX, a private aerospace manufacturer and space transportation company. Sam Altman is the CEO of OpenAI, an artificial intelligence research lab that aims to ensure that artificial general intelligence benefits all of humanity.
"""

OUTPUT_EXAMPLE = """
[{
entity_name: "Elon Musk"
entity_type: 1
entity_description: "Elon Musk is the CEO of Tesla, Inc. He is also the founder and CEO of SpaceX. "
entity_questions: ["Who is the CEO of Tesla and SpaceX?", "Who founded Space X?"]
},
{
entity_name: "Sam Altman"
entity_type: 1
entity_description: "Sam Altman is the CEO of OpenAI."
entity_questions: ["Who is the CEO of OpenAI?"]
},
{
entity_name: "SpaceX"
entity_type: 2
entity_description: "SpaceX is a private aerospace manufacturer and space transportation company."
entity_questions: ["What industry does SpaceX belong to?", "Is SpaceX a private or public company?"]
},
{
entity_name: "OpenAI"
entity_type: 2
entity_description: "OpenAI is an artificial intelligence research lab that aims to ensure that artificial general intelligence benefits all of humanity."
entity_questions: ["What is the goal of OpenAI?", "What type of research does OpenAI conduct?"]
},
{
entity_name: "Tesla, Inc."
entity_type: 2
entity_description: "Tesla, Inc. is a company known for its electric vehicles and clean energy solutions."
entity_questions: ["What products is Tesla, Inc. known for?", "What industry does Tesla, Inc. operate in?"]
}
"""