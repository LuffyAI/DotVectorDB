/*********************************
----> Entity <----
Purpose: This file contains the entity schema for the database.
- entity_id: This is the primary key for the entity table. To get this value programmatically, the entity_id is a hash between the entity name & type.
- entity_name: This is the name of the entity.
- entity_description: This is a text description of the entity.
- degrees: This is the number of connections an entity has to other entities. Its 1-hop neighbors in the graph.
- embedding_model_id: This is the ID of the embedding model used to generate embeddings for the entity. It is a foreign key that references the id in the embedding_model table.
- is_embedded: This is a boolean flag that indicates whether the entity has been embedded or not.
- is_deduplicated: This is deprecated and will be removed in the future.
- tokens: This is the number of tokens used to generate the entity.
*********************************/
CREATE TABLE IF NOT EXISTS entity (
    entity_id INTEGER NOT NULL,
    entity_name TEXT NOT NULL,
    entity_description TEXT NOT NULL,
    degrees INTEGER DEFAULT 0,
    embedding_model_id TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    is_summarized BOOLEAN DEFAULT TRUE,
    last_summarized_update INTEGER NOT NULL DEFAULT 0,
    created_version INTEGER NOT NULL DEFAULT 0,
    last_dedupe_update INTEGER NOT NULL DEFAULT 0,
    last_embedding_update INTEGER NOT NULL DEFAULT 0,
    last_version_update INTEGER NOT NULL DEFAULT 0,
    tokens INT NOT NULL,
    PRIMARY KEY (entity_id),
    FOREIGN KEY (embedding_model_id)
        REFERENCES embedding_model(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS new_and_updated_entity (
    entity_id INTEGER NOT NULL PRIMARY KEY,
    is_new BOOLEAN NOT NULL CHECK (is_new IN (0, 1)),
    FOREIGN KEY (entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE
);

-- Deleted IDs must survive removal of their original rows.
CREATE TABLE IF NOT EXISTS deleted_entity (
    entity_id INTEGER NOT NULL PRIMARY KEY
);

CREATE TRIGGER IF NOT EXISTS track_deleted_entity
AFTER DELETE ON entity
BEGIN
    INSERT OR IGNORE INTO deleted_entity (entity_id)
    VALUES (OLD.entity_id);
END;

CREATE TRIGGER IF NOT EXISTS track_new_entity
AFTER INSERT ON entity
BEGIN
    INSERT INTO new_and_updated_entity (entity_id, is_new)
    VALUES (NEW.entity_id, TRUE)
    ON CONFLICT(entity_id) DO UPDATE SET is_new = TRUE;
END;

CREATE TRIGGER IF NOT EXISTS track_updated_entity
AFTER UPDATE OF entity_name, entity_description, degrees, embedding_model_id, tokens ON entity
WHEN NEW.entity_name IS NOT OLD.entity_name
  OR NEW.entity_description IS NOT OLD.entity_description
  OR NEW.degrees IS NOT OLD.degrees
  OR NEW.embedding_model_id IS NOT OLD.embedding_model_id
  OR NEW.tokens IS NOT OLD.tokens
BEGIN
    INSERT INTO new_and_updated_entity (entity_id, is_new)
    VALUES (NEW.entity_id, FALSE)
    ON CONFLICT(entity_id) DO NOTHING;
END;

-- Ordered degree index lets extrema use endpoint lookups, without scanning entities.
CREATE INDEX IF NOT EXISTS idx_entity_degrees ON entity(degrees);

CREATE TRIGGER IF NOT EXISTS insert_global_degree_extrema
AFTER INSERT ON entity
BEGIN
    UPDATE global_config
    SET min_degree = COALESCE((SELECT degrees FROM entity
                              WHERE degrees IS NOT NULL ORDER BY degrees LIMIT 1), 0),
        max_degree = MAX(max_degree, COALESCE(NEW.degrees, 0))
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS update_global_degree_extrema
AFTER UPDATE OF degrees ON entity
WHEN NEW.degrees IS NOT OLD.degrees
BEGIN
    UPDATE global_config
    SET min_degree = CASE
            WHEN OLD.degrees = min_degree THEN COALESCE(
                (SELECT degrees FROM entity WHERE degrees IS NOT NULL
                 ORDER BY degrees LIMIT 1), 0)
            ELSE MIN(min_degree, COALESCE(NEW.degrees, min_degree)) END,
        max_degree = CASE
            WHEN OLD.degrees = max_degree THEN COALESCE(
                (SELECT degrees FROM entity WHERE degrees IS NOT NULL
                 ORDER BY degrees DESC LIMIT 1), 0)
            ELSE MAX(max_degree, COALESCE(NEW.degrees, max_degree)) END
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS delete_global_degree_extrema
AFTER DELETE ON entity
BEGIN
    UPDATE global_config
    SET min_degree = CASE WHEN OLD.degrees = min_degree THEN COALESCE(
            (SELECT degrees FROM entity WHERE degrees IS NOT NULL
             ORDER BY degrees LIMIT 1), 0) ELSE min_degree END,
        max_degree = CASE WHEN OLD.degrees = max_degree THEN COALESCE(
            (SELECT degrees FROM entity WHERE degrees IS NOT NULL
             ORDER BY degrees DESC LIMIT 1), 0) ELSE max_degree END
    WHERE id = 1;
END;

CREATE TABLE IF NOT EXISTS entity_to_entity_weights (
    entity_id INTEGER NOT NULL,
    similar_entity_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    PRIMARY KEY (entity_id, similar_entity_id),
    FOREIGN KEY (entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE,
    FOREIGN KEY (similar_entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE
);


CREATE INDEX IF NOT EXISTS idx_entity_last_version_update
ON entity(last_version_update);

CREATE INDEX IF NOT EXISTS idx_entity_created_version
ON entity(created_version);

CREATE INDEX IF NOT EXISTS idx_entity_last_dedupe_check
ON entity(last_dedupe_update);

CREATE TABLE IF NOT EXISTS entity_types (
    entity_type TEXT NOT NULL,
    type_description TEXT NOT NULL,
    num_entities INTEGER DEFAULT 0,
    PRIMARY KEY (entity_type)
);

/*********************************
----> Entity to entity type mapping <----
Purpose: This file contains the entity schema for mapping entities to their associated entity types in the database. In DotRAG, an entity can have more than one type:
0 entity_id: This is the primary key for the entity_to_entity_type_mapping table. It is a foreign key that references the entity_id in the entity table.
1 entity_type: This is a foreign key that references the entity_type in the entity_types table
*********************************/
CREATE TABLE IF NOT EXISTS entity_to_entity_type_mapping (
    entity_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    PRIMARY KEY (entity_id, entity_type),
    FOREIGN KEY (entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE,
    FOREIGN KEY (entity_type)
        REFERENCES entity_types(entity_type)
        ON DELETE CASCADE
);

/*********************************
----> Chunk to entity mapping <----
Purpose: This file contains the entity schema for mapping chunks to its associated entities in the database.
- chunk_id: This is the primary key for the chunk_to_entity_mapping table. It is a foreign key that references the chunk_id in the chunk table.
- entity_id: This is a foreign key that references the entity_id in the entity table.
*********************************/
CREATE TABLE IF NOT EXISTS chunk_to_entity_mapping (
    chunk_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    is_deduplicated BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id)
        REFERENCES chunk(chunk_id)
        ON DELETE CASCADE,
    FOREIGN KEY (entity_id)
        REFERENCES entity(entity_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunk_entity_by_entity
ON chunk_to_entity_mapping(entity_id, chunk_id);

-- Count an entity once per source, even when several chunks mention it.
CREATE TRIGGER IF NOT EXISTS increment_source_num_entities
AFTER INSERT ON chunk_to_entity_mapping
BEGIN
    UPDATE source
    SET num_entities = num_entities + 1
    WHERE id = (SELECT source_id FROM chunk WHERE chunk_id = NEW.chunk_id)
      AND NOT EXISTS (
          SELECT 1
          FROM chunk_to_entity_mapping AS mapping
          JOIN chunk ON chunk.chunk_id = mapping.chunk_id
          WHERE mapping.entity_id = NEW.entity_id
            AND mapping.chunk_id <> NEW.chunk_id
            AND chunk.source_id = source.id
      );
END;

CREATE TRIGGER IF NOT EXISTS decrement_source_num_entities
BEFORE DELETE ON chunk_to_entity_mapping
BEGIN
    UPDATE source
    SET num_entities = MAX(num_entities - 1, 0)
    WHERE id = (SELECT source_id FROM chunk WHERE chunk_id = OLD.chunk_id)
      AND NOT EXISTS (
          SELECT 1
          FROM chunk_to_entity_mapping AS mapping
          JOIN chunk ON chunk.chunk_id = mapping.chunk_id
          WHERE mapping.entity_id = OLD.entity_id
            AND mapping.chunk_id <> OLD.chunk_id
            AND chunk.source_id = source.id
      );
END;

/*********************************
----> Increment/Decrement triggers <----
Purpose: Keep global entity counts and per-type entity counts synchronized with
the entity table and its many-to-many type mapping table.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_global_entity_count
AFTER INSERT ON entity
BEGIN
    UPDATE global_config
    SET avg_entity_tokens =
            CAST(
                COALESCE(total_entity_tokens, 0) + NEW.tokens
                AS REAL
            ) / (COALESCE(num_entities, 0) + 1),
        total_entity_tokens =
            COALESCE(total_entity_tokens, 0) + NEW.tokens,
        avg_degree =
            CASE
                WHEN COALESCE(num_entities, 0) + 1 = 0 THEN 0.0
                ELSE (2.0 * COALESCE(num_relations, 0))
                    / (COALESCE(num_entities, 0) + 1)
            END,
        num_entities = COALESCE(num_entities, 0) + 1,
        current_version_count =
            current_version_count + 1
    WHERE id = 1;

    UPDATE entity
    SET created_version = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    ),
    last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE entity_id = NEW.entity_id;
END;

/*********************************
Update the global database version when entity data is modified, then
stamp that entity with the new version. Processing flags and checkpoints
do not invalidate completed work. The WHEN condition
prevents the internal version-stamping update from recursively incrementing
the global version.
*********************************/
CREATE TRIGGER IF NOT EXISTS update_entity_version
AFTER UPDATE OF entity_id, entity_name, entity_description, degrees,
    embedding_model_id, tokens ON entity
WHEN NEW.last_version_update =
     OLD.last_version_update
BEGIN
    UPDATE global_config
    SET current_version_count =
        current_version_count + 1
    WHERE id = 1;

    UPDATE entity
    SET last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE entity_id = NEW.entity_id;
END;

CREATE TRIGGER IF NOT EXISTS update_global_entity_token_stats
AFTER UPDATE OF tokens ON entity
WHEN NEW.tokens <> OLD.tokens
BEGIN
    UPDATE global_config
    SET avg_entity_tokens =
            CASE
                WHEN COALESCE(num_entities, 0) = 0 THEN 0.0
                ELSE CAST(
                    COALESCE(total_entity_tokens, 0)
                    + NEW.tokens
                    - OLD.tokens
                    AS REAL
                ) / num_entities
            END,
        total_entity_tokens = MAX(
            COALESCE(total_entity_tokens, 0)
            + NEW.tokens
            - OLD.tokens,
            0
        )
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS decrement_global_entity_count
AFTER DELETE ON entity
BEGIN
    UPDATE global_config
    SET avg_entity_tokens =
            CASE
                WHEN COALESCE(num_entities, 0) <= 1 THEN 0.0
                ELSE CAST(
                    MAX(
                        COALESCE(total_entity_tokens, 0) - OLD.tokens,
                        0
                    ) AS REAL
                ) / (num_entities - 1)
            END,
        total_entity_tokens = MAX(
            COALESCE(total_entity_tokens, 0) - OLD.tokens,
            0
        ),
        avg_degree =
            CASE
                WHEN COALESCE(num_entities, 0) <= 1 THEN 0.0
                ELSE (2.0 * COALESCE(num_relations, 0))
                    / (num_entities - 1)
            END,
        num_entities = MAX(COALESCE(num_entities, 0) - 1, 0),
        current_version_count =
            current_version_count + 1
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS increment_entity_type_count
AFTER INSERT ON entity_to_entity_type_mapping
BEGIN
    UPDATE entity_types
    SET num_entities = num_entities + 1
    WHERE entity_type = NEW.entity_type;

    UPDATE global_config
    SET current_version_count =
        current_version_count + 1
    WHERE id = 1;

    UPDATE entity
    SET last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE entity_id = NEW.entity_id;
END;

CREATE TRIGGER IF NOT EXISTS decrement_entity_type_count
AFTER DELETE ON entity_to_entity_type_mapping
BEGIN
    UPDATE entity_types
    SET num_entities = MAX(
        num_entities - 1,
        0
    )
    WHERE entity_type = OLD.entity_type;

    UPDATE global_config
    SET current_version_count =
        current_version_count + 1
    WHERE id = 1;

    UPDATE entity
    SET last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE entity_id = OLD.entity_id;
END;

CREATE TRIGGER IF NOT EXISTS update_entity_type_count
AFTER UPDATE OF entity_type
ON entity_to_entity_type_mapping
WHEN OLD.entity_type <> NEW.entity_type
BEGIN
    UPDATE entity_types
    SET num_entities = MAX(
        num_entities - 1,
        0
    )
    WHERE entity_type = OLD.entity_type;

    UPDATE entity_types
    SET num_entities = num_entities + 1
    WHERE entity_type = NEW.entity_type;

    UPDATE global_config
    SET current_version_count =
        current_version_count + 1
    WHERE id = 1;

    UPDATE entity
    SET last_version_update = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE entity_id = NEW.entity_id;
END;
