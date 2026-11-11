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
- tags: This is a deprecrated field.
- embedding_model_id: This is the ID of the embedding model used to generate embeddings for the relationship. It is a foreign key that references the id in the embedding_model table.
- is_embedded: This is a boolean flag that indicates whether the relationship has been embedded or not.
- is_deduplicated: This is a boolean flag that indicates whether the relationship has been deduplicated or not. THis is deprecated and will be removed in the future.
- tokens: This is the number of tokens used to generate the relationship.
*********************************/
CREATE TABLE IF NOT EXISTS relationship (
    relation_id INTEGER NOT NULL,
    src_entity_id INTEGER NOT NULL,
    dst_entity_id INTEGER NOT NULL,
    relation_description TEXT NOT NULL,
    keywords JSON NOT NULL,
    supporting_lines JSON NOT NULL,
    last_version_update INTEGER NOT NULL DEFAULT 0,
    embedding_model_id TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    is_summarized BOOLEAN DEFAULT TRUE,
    tokens INT NOT NULL,
    PRIMARY KEY (relation_id),
    FOREIGN KEY (src_entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE,
    FOREIGN KEY (dst_entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE,
    FOREIGN KEY (embedding_model_id)
        REFERENCES embedding_model(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS new_and_updated_relationship (
    relation_id INTEGER NOT NULL PRIMARY KEY,
    is_new BOOLEAN NOT NULL CHECK (is_new IN (0, 1)),
    FOREIGN KEY (relation_id)
        REFERENCES relationship(relation_id)
        ON DELETE CASCADE
);

-- Deleted IDs must survive removal of their original rows.
CREATE TABLE IF NOT EXISTS deleted_relationship (
    relation_id INTEGER NOT NULL PRIMARY KEY
);

CREATE TRIGGER IF NOT EXISTS track_deleted_relationship
AFTER DELETE ON relationship
BEGIN
    INSERT OR IGNORE INTO deleted_relationship (relation_id)
    VALUES (OLD.relation_id);
END;

CREATE TRIGGER IF NOT EXISTS track_new_relationship
AFTER INSERT ON relationship
BEGIN
    INSERT INTO new_and_updated_relationship (relation_id, is_new)
    VALUES (NEW.relation_id, TRUE)
    ON CONFLICT(relation_id) DO UPDATE SET is_new = TRUE;
END;

CREATE TRIGGER IF NOT EXISTS track_updated_relationship
AFTER UPDATE OF src_entity_id, dst_entity_id, relation_description, keywords, supporting_lines, embedding_model_id, tokens ON relationship
WHEN NEW.src_entity_id IS NOT OLD.src_entity_id
  OR NEW.dst_entity_id IS NOT OLD.dst_entity_id
  OR NEW.relation_description IS NOT OLD.relation_description
  OR NEW.keywords IS NOT OLD.keywords
  OR NEW.supporting_lines IS NOT OLD.supporting_lines
  OR NEW.embedding_model_id IS NOT OLD.embedding_model_id
  OR NEW.tokens IS NOT OLD.tokens
BEGIN
    INSERT INTO new_and_updated_relationship (relation_id, is_new)
    VALUES (NEW.relation_id, FALSE)
    ON CONFLICT(relation_id) DO NOTHING;
END;

/*********************************
----> Chunk to relationship mapping <----
Purpose: Associates each relationship with the chunks where it was found.
*********************************/
CREATE TABLE IF NOT EXISTS chunk_to_relationship_mapping (
    chunk_id INTEGER NOT NULL,
    relation_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, relation_id),
    FOREIGN KEY (chunk_id)
        REFERENCES chunk(chunk_id)
        ON DELETE CASCADE,
    FOREIGN KEY (relation_id)
        REFERENCES relationship(relation_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sentence_to_relationship_mapping (
    chunk_id INTEGER NOT NULL,
    sentence_num INTEGER NOT NULL,
    relation_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, sentence_num, relation_id),
    FOREIGN KEY (chunk_id, sentence_num)
        REFERENCES chunk_to_sentence_mapping(chunk_id, sentence_number)
        ON DELETE CASCADE,
    FOREIGN KEY (relation_id)
        REFERENCES relationship(relation_id)
        ON DELETE CASCADE
);

/*********************************
----> Entity degree triggers <----
Purpose: Keep entity degree counts synchronized with temporary and finalized
relationships. Each relationship contributes one degree to each endpoint.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_entity_degrees
AFTER INSERT ON relationship
BEGIN
    UPDATE entity
    SET degrees = degrees + 1
    WHERE entity_id = NEW.src_entity_id;

    UPDATE entity
    SET degrees = degrees + 1
    WHERE entity_id = NEW.dst_entity_id;
END;

CREATE TRIGGER IF NOT EXISTS decrement_entity_degrees
AFTER DELETE ON relationship
BEGIN
    UPDATE entity
    SET degrees = MAX(degrees - 1, 0)
    WHERE entity_id = OLD.src_entity_id;

    UPDATE entity
    SET degrees = MAX(degrees - 1, 0)
    WHERE entity_id = OLD.dst_entity_id;
END;

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
        JOIN chunk
            ON chunk.chunk_id =
               chunk_to_entity_mapping.chunk_id
        WHERE chunk_to_entity_mapping.entity_id =
              NEW.src_entity_id
        LIMIT 1
    );

    UPDATE global_config
    SET avg_relationship_tokens =
            CAST(
                COALESCE(total_relationship_tokens, 0) + NEW.tokens
                AS REAL
            ) / (COALESCE(num_relations, 0) + 1),
        total_relationship_tokens =
            COALESCE(total_relationship_tokens, 0) + NEW.tokens,
        avg_degree =
            CASE
                WHEN COALESCE(num_entities, 0) = 0 THEN 0.0
                ELSE (2.0 * (COALESCE(num_relations, 0) + 1))
                    / num_entities
            END,
        num_relations = COALESCE(num_relations, 0) + 1
    WHERE id = 1;

    UPDATE global_config
    SET current_version_count =
        current_version_count + 1;

    UPDATE relationship
    SET last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE relation_id = NEW.relation_id;
END;

/*********************************
Update the global database version whenever an existing relationship row is
modified, then stamp that relationship with the new version. The WHEN
condition prevents the internal version-stamping update from recursively
incrementing the global version.
*********************************/
CREATE TRIGGER IF NOT EXISTS update_relationship_version
AFTER UPDATE ON relationship
WHEN NEW.last_version_update =
     OLD.last_version_update
BEGIN
    UPDATE global_config
    SET current_version_count =
        current_version_count + 1
    WHERE id = 1;

    UPDATE relationship
    SET last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE relation_id = NEW.relation_id;
END;

CREATE TRIGGER IF NOT EXISTS update_global_relationship_token_stats
AFTER UPDATE OF tokens ON relationship
WHEN NEW.tokens <> OLD.tokens
BEGIN
    UPDATE global_config
    SET avg_relationship_tokens =
            CASE
                WHEN COALESCE(num_relations, 0) = 0 THEN 0.0
                ELSE CAST(
                    COALESCE(total_relationship_tokens, 0)
                    + NEW.tokens
                    - OLD.tokens
                    AS REAL
                ) / num_relations
            END,
        total_relationship_tokens = MAX(
            COALESCE(total_relationship_tokens, 0)
            + NEW.tokens
            - OLD.tokens,
            0
        )
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS decrement_source_num_relations
AFTER DELETE ON relationship
BEGIN
    UPDATE source
    SET num_relations = MAX(
        num_relations - 1,
        0
    )
    WHERE id = (
        SELECT chunk.source_id
        FROM chunk_to_entity_mapping
        JOIN chunk
            ON chunk.chunk_id =
               chunk_to_entity_mapping.chunk_id
        WHERE chunk_to_entity_mapping.entity_id =
              OLD.src_entity_id
        LIMIT 1
    );

    UPDATE global_config
    SET avg_relationship_tokens =
            CASE
                WHEN COALESCE(num_relations, 0) <= 1 THEN 0.0
                ELSE CAST(
                    MAX(
                        COALESCE(total_relationship_tokens, 0) - OLD.tokens,
                        0
                    ) AS REAL
                ) / (num_relations - 1)
            END,
        total_relationship_tokens = MAX(
            COALESCE(total_relationship_tokens, 0) - OLD.tokens,
            0
        ),
        avg_degree =
            CASE
                WHEN COALESCE(num_entities, 0) = 0 THEN 0.0
                ELSE (
                    2.0 * MAX(COALESCE(num_relations, 0) - 1, 0)
                ) / num_entities
            END,
        num_relations = MAX(COALESCE(num_relations, 0) - 1, 0)
    WHERE id = 1;

    UPDATE global_config
    SET current_version_count =
        current_version_count + 1;
END;
