GLOBAL_TYPE_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for reading a list of randomly sampled entities of a specific type with the objective
of trying to figure out a global description of what the full corpus could be about.
</Role>

<Instructions>
   - Carefully read the '<Type>' section to understand the type of entities that were sampled from the graph database.
   - Read the '<Sample>' section to understand the sampled entities of the specific type.
   - Based on the information in the '<Type>' and '<Sample>' sections, observe the commonalities and patterns among the sampled entities to infer a global description of what the full corpus could be about.
</Instructions>

<Type>
{type_name}
</Type>

<Sample>
{sample}
</Sample>
"""

GLOBAL_DESCRIPTION_PROMPT = """
<Role>
You are a Knowledge Graph Specialist responsible for reading a list of global descriptions of a specific entity type with the objective of trying to figure out a global description of what the full corpus could be about.
</Role>

<Instructions>
    - Carefully read the global description of the specific entity type in the '<Global_Descriptions>' section.
    - Based on the information in the '<Global_Descriptions>' section, observe the commonalities and patterns among the global descriptions to infer a global description of what the full corpus could be about.
    - Generate a global summary of no more than 200 words that captures the essence of the full corpus based on the information provided in the '<Global_Descriptions>' section.
</Instructions>

<Global_Descriptions>
{global_descriptions}
</Global_Descriptions>
"""