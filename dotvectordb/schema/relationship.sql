/*********************************
---->  Relationship <----
Purpose: This file contains the relationship schema for the database.
- relation_id: This is the primary key for the relationship table. To get this value programmatically, the relation_id is a hash between the entity name & type.
- src_entity_id: This is the source entity ID for the relationship. It is a foreign key that references the entity_id in the entity table. (e.g, for A->B, the src_entity_id is A)
- dst_entity_id: This is the destination entity ID for the relationship. It is a foreign key that references the entity_id in the entity table. (e.g, for A->B, the dst_entity_id is B)
- relation_description: This is a text description of the relationship between the source and destination entities.
- questions: This is a JSON array of questions we suspect a user would ask about the relationship . (e.g, for A->B, a question could "What is the next letter after A?", or "What is the letter before B?") In these questions, the answer to the question is either the source or destination entity.
- tags: This is a deprecrated field.
- embedding_model_id: This is the ID of the embedding model used to generate embeddings for the relationship. It is a foreign key that references the id in the embedding_model table.
- is_embedded: This is a boolean flag that indicates whether the relationship has been embedded or not.
- is_deduplicated: This is a boolean flag that indicates whether the relationship has been deduplicated or not. THis is deprecated and will be removed in the future.
- tokens: This is the number of tokens used to generate the relationship.
*********************************/
CREATE TABLE IF NOT EXISTS relationship (
    relation_id INTEGER NOT NULL,
    src_entity_id TEXT NOT NULL,
    dst_entity_id TEXT NOT NULL,
    relation_description TEXT NOT NULL,
    questions JSON NOT NULL,
    tags JSON,
    embedding_model_id TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    is_deduplicated BOOLEAN DEFAULT FALSE,
    tokens INT NOT NULL,
    PRIMARY KEY (relation_id),
    FOREIGN KEY (src_entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (dst_entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE CASCADE
);


