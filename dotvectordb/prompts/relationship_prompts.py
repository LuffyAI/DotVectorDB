from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

class Psuedoquery(BaseModel):
    Questions: List[str]

class Relationship(BaseModel):
    src_entity_id: int
    dst_entity_id: int
    description: str
    relation_questions: Psuedoquery

class Relationship(BaseModel):
    relations: List[Relationship]


RELATIONSHIP_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for reading a list of entities from the '<Entities> section and describing the relationships observed between each entity based on the `<Input>` section.
</Role>

<Instructions>
**Relationship Extraction:**
   - Carefully read the '<Entities>' section to understand entities to focus on. Do not extract any additional entities from '<Input>'.
   - Identify all relationships clearly defined between the identified entities in the `<Input>` section. Extract as many as you can. Do not invent any relationships between the entities that is not supported by the information in '<Input>.'
    - For each relationship, extract:
        - `src_entity_id`: The ID of the source entity as seen in the '<Entities>' section.
        - `dst_entity_id`: The ID of the destination entity as seen in the '<Entities>' section.
        - 'description': A concise yet comprehensive description of the relationship, based *solely* on the information present in the input text.
        - 'relation_questions': Generate a list of at least one question that could be asked and answered about the relationship based on the information in the input text. The question should be relevant to the relationship and its context within the input text. For example, if the relationship is "Alice is Bob's sister", a relevant question could be "Who is Bob's sister?"

**Language & Style:**
   - Ensure all descriptions are written in the **third person**. 
</Instructions>

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
</Examples>
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