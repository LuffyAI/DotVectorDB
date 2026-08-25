/*********************************
Design note: Relationships should ideally store their associated source_id directly rather than infer it through an entity. An entity may appear in multiple sources, so resolving a relationship’s source through chunk_to_entity_mapping can be ambiguous and LIMIT 1 may select an arbitrary source.
*/

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

/*********************************
----> Increment/Decrement triggers <----
Purpose: This these triggers are used to automatically update the num_relations field in the source table when relationships are added or removed from the relationship table:
In each of these operations, we must increment global_config's current_version_count by 1 to ensure that the database version is updated after each write operation.
- increment_source_num_relations: This trigger is fired after a relationship is inserted into the relationship table. It increments the num_relations field in the source table for the corresponding source_id.
- decrement_source_num_relations: This trigger is fired after a relationship is deleted from the relationship table. It decrements the num_relations field in the source table for the corresponding source_id.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_source_num_relations
AFTER INSERT ON relationship
BEGIN
    UPDATE source
    SET num_relations = num_relations + 1
    WHERE id = (
        SELECT chunk.source_id
        FROM chunk_to_entity_mapping
        JOIN chunk ON chunk.chunk_id = chunk_to_entity_mapping.chunk_id
        WHERE chunk_to_entity_mapping.entity_id = NEW.src_entity_id
        LIMIT 1
    );
    UPDATE global_config
    SET current_version_count = current_version_count + 1;
END;


CREATE TRIGGER IF NOT EXISTS decrement_source_num_relations
AFTER DELETE ON relationship
BEGIN
    UPDATE source
    SET num_relations = MAX(num_relations - 1, 0)
    WHERE id = (
        SELECT chunk.source_id
        FROM chunk_to_entity_mapping
        JOIN chunk ON chunk.chunk_id = chunk_to_entity_mapping.chunk_id
        WHERE chunk_to_entity_mapping.entity_id = OLD.src_entity_id
        LIMIT 1
    );
    UPDATE global_config
    SET current_version_count = current_version_count + 1;
END;

