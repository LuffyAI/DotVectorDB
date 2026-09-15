from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field


class Relationships(BaseModel):
    src_entity: str
    src_entity_types: List[str]
    dst_entity: str
    dst_entity_types: List[str]
    relationship_description: str
    follows_named_entity_rules: bool
    named_entity_rule_explanation: str

class TypeProposals(BaseModel):
    Proposals: List[Relationships]


TYPE_DISCOVERY_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for reading the chunk from the '<Input>' section and extracting relationships between the named entities in the text. You will also invent appropriate entity types for each entity based on the context of the input text. Try to stay consistent if entity types are reusable. 
</Role>

<NAMED_ENTITY_DEFINITIONS>
A named entity is a specific, identifiable subject that the text refers to by a proper name, official designation, distinctive title, model number, or other stable identifier. Extract it only when the same subject could reasonably be recognized or referenced again elsewhere.

Generally exclude:
- Pronouns and conversational references
- Unnamed people, roles, and groups
- Non-specific category mentions that do not identify a particular entity
- Actions, feelings, qualities, and ordinary phrases
- Possessive descriptions that merely refer to something contextually
- Categories mistaken for individual entities
- Concepts or events fabricated by summarizing a sentence
- Relationship phrases presented as entities

Objects and substances are valid named entities when the text identifies a specific object, item, material, food, drug, element, or chemical substance that can function as a stable graph node. Do not reject an entity merely because its routing type is object or substance. Reject only a generic, incidental mention that does not identify a specific entity in context.

The domain-independent test is:
    Could this candidate function as a stable node across documents, or does it make sense only inside this particular sentence?
</NAMED_ENTITY_DEFINITIONS>

<Steps>
1. Read the '<Input>' section carefully 
2. Read the '<DEFAULT_ENTITY_TYPES>' section to understand the default entity types that you can use as a reference for inventing new entity types. **Only invent a new entity type if it is not covered by the default entity types.
3. Then, identify all unique relationships that seem important to understanding the input text. For each relationship, extract the following information:
- src_entity: The name of the source entity in the relationship. Ensure **consistent naming** across the entire extraction process.
- src_entity_types: Invent appropriate entity types for the source entity based on the context of the input text. If the entity can belong to multiple types, choose all that applies. Ensure **consistent naming** across the entire extraction process.
- dst_entity: The name of the destination entity in the relationship. Ensure **consistent naming** across the entire extraction process.
- dst_entity_types: Invent appropriate entity types for the destination entity based on the context of the input text. If the entity can belong to multiple types, choose all that applies. Ensure **consistent naming** across the entire extraction process.
- relationship_description: A concise yet comprehensive description of the relationship, based *solely* on the information present in the input text.
- follows_named_entity_rules: Return true only if both the source and destination are named entities according to the '<NAMED_ENTITY_DEFINITIONS>' section. Otherwise, return false.
- named_entity_rule_explanation: Briefly explain why the relationship passes or fails the named-entity rules. Identify the invalid endpoint when returning false.

Evaluate every proposed relationship independently. Always return both validation fields. Do not change a false result to true merely because the relationship itself is meaningful.

<DEFAULT_ENTITY_TYPES>
["person", "organization", "location", "event", "work_of_art", "law", "human_language", "time"]
</DEFAULT_ENTITY_TYPES>

<Input>
{chunk}
</Input>

<Output_Example>
{output_example}
</Output_Example>
</Examples>
"""


OUTPUT_EXAMPLE = ["""
Example 1:
Example Entity Types: [person, technology, mission, organization, location]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

Output:
{"relationships":[{"src_entity":"Alex","src_entity_types":["person"],"dst_entity":"Taylor","dst_entity_types":["person"],"relationship_description":"Alex feels frustrated by Taylor's authoritarian certainty and narrowing vision of control and order."},{"src_entity":"Alex","src_entity_types":["person"],"dst_entity":"Jordan","dst_entity_types":["person"],"relationship_description":"Alex and Jordan share a commitment to discovery that Alex regards as an unspoken rebellion."},{"src_entity":"Taylor","src_entity_types":["person"],"dst_entity":"Jordan","dst_entity_types":["person"],"relationship_description":"Taylor pauses beside Jordan, observes the device with reverence, and shares a moment of reluctant respect and uneasy truce with Jordan."},{"src_entity":"Jordan","src_entity_types":["person"],"dst_entity":"Cruz","dst_entity_types":["person"],"relationship_description":"Jordan's commitment to discovery stands in opposition to Cruz's narrowing vision of control and order."},{"src_entity":"Taylor","src_entity_types":["person"],"dst_entity":"The Device","dst_entity_types":["technology"],"relationship_description":"Taylor regards the device with reverence and believes understanding its technology could change the game for everyone."}]}
#############################""",
    """Example 2:

Entity_types: [company, index, commodity, market_trend, economic_policy, biological]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

Output:
{"relationships":[{"src_entity":"Global Tech Index","src_entity_types":["index"],"dst_entity":"Market Selloff","dst_entity_types":["market_trend"],"relationship_description":"The Global Tech Index fell 3.4% during the broader market selloff."},{"src_entity":"Nexon Technologies","src_entity_types":["company"],"dst_entity":"Global Tech Index","dst_entity_types":["index"],"relationship_description":"Nexon Technologies was among the technology companies hit by the downturn, with its stock falling 7.8% after weak quarterly earnings."},{"src_entity":"Omega Energy","src_entity_types":["company"],"dst_entity":"Crude Oil","dst_entity_types":["commodity"],"relationship_description":"Omega Energy gained 2.1%, driven by rising crude oil prices."},{"src_entity":"Gold Futures","src_entity_types":["commodity","financial_instrument"],"dst_entity":"Market Selloff","dst_entity_types":["market_trend"],"relationship_description":"Gold futures rose 1.5% as investors sought safe-haven assets during the market downturn."},{"src_entity":"Federal Reserve","src_entity_types":["central_bank"],"dst_entity":"Federal Reserve Policy Announcement","dst_entity_types":["economic_policy","event"],"relationship_description":"The Federal Reserve is expected to make an upcoming policy announcement amid speculation about potential rate hikes."},{"src_entity":"Federal Reserve Policy Announcement","src_entity_types":["economic_policy","event"],"dst_entity":"Market Selloff","dst_entity_types":["market_trend"],"relationship_description":"The upcoming Federal Reserve policy announcement is expected to influence investor confidence and overall market stability."}]}

""","""
Example 3:
Entity_types: [economic_policy, athlete, event, location, record, organization, equipment]
Text:
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

Output:
{"relationships":[{"src_entity":"World Athletics Championship","src_entity_types":["event"],"dst_entity":"Tokyo","dst_entity_types":["location"],"relationship_description":"The World Athletics Championship took place in Tokyo."},{"src_entity":"Noah Carter","src_entity_types":["athlete"],"dst_entity":"100m Sprint Record","dst_entity_types":["record"],"relationship_description":"Noah Carter broke the 100m sprint record."},{"src_entity":"Noah Carter","src_entity_types":["athlete"],"dst_entity":"Carbon-Fiber Spikes","dst_entity_types":["equipment"],"relationship_description":"Noah Carter used cutting-edge carbon-fiber spikes when breaking the record."},{"src_entity":"Noah Carter","src_entity_types":["athlete"],"dst_entity":"World Athletics Championship","dst_entity_types":["event"],"relationship_description":"Noah Carter competed at the World Athletics Championship."}]}
"""


]
