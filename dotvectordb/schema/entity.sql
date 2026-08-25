/*********************************
----> Entity <----
Purpose: This file contains the entity schema for the database.
- entity_id: This is the primary key for the entity table. To get this value programmatically, the entity_id is a hash between the entity name & type.
- entity_name: This is the name of the entity.
- entity_type: This is the type of the entity. It is a foreign key that references the entity_type in the entity_types table.
- entity_description: This is a text description of the entity.
- questions: This is a JSON array of questions we suspect a user would ask about the entity. (e.g, for about Shrek, a question could be "Who is the green ogre in the story?" The answer is the entity.
- alias: This is depreciated. 
- embedding_model_id: This is the ID of the embedding model used to generate embeddings for the entity. It is a foreign key that references the id in the embedding_model table.
- is_embedded: This is a boolean flag that indicates whether the entity has been embedded or not.
- is_deduplicated: This is deprecated and will be removed in the future.
- tokens: This is the number of tokens used to generate the entity.
*********************************/
CREATE TABLE IF NOT EXISTS entity (
    entity_id INTEGER NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_description TEXT NOT NULL,
    questions JSON NOT NULL,
    alias JSON,
    embedding_model_id TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    is_deduplicated BOOLEAN DEFAULT FALSE,
    tokens INT NOT NULL,
    PRIMARY KEY (entity_id),
    FOREIGN KEY (entity_type) REFERENCES entity_types(entity_type) ON DELETE CASCADE,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE CASCADE
);
/*********************************
----> Entity Types <----
Purpose: This file contains the entity schema for entity types in the database.
- entity_type: This is the primary key for the entity_types table. It is a text field that represents the type of the entity.
- type_description: This is a text description of the entity type.
- num_entities: This is an integer that represents the number of entities of this type in the database. It is updated automatically by triggers when entities are added or removed.
*********************************/
CREATE TABLE IF NOT EXISTS entity_types (
    entity_type TEXT NOT NULL,
    type_description TEXT NOT NULL,
    num_entities INT DEFAULT 0,
    PRIMARY KEY (entity_type)
);
/*********************************
----> Increment/Decrement triggers <----
Purpose: This these triggers are used to automatically update the num_entities field in the entity_types table when entities are added or removed from the entity table.
- increment_entity_type_count: This trigger is fired after an entity is inserted into the entity table. It increments the num_entities field in the entity_types table for the corresponding entity_type.
- decrement_entity_type_count: This trigger is fired after an entity is deleted from the entity table. It decrements the num_entities field in the entity_types table for the corresponding entity_type.
- update_entity_type_count: This trigger is fired after an entity's entity_type is updated in the entity table. It decrements the num_entities field in the entity_types table for the old entity_type and increments the num_entities field in the entity_types table for the new entity_type.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_entity_type_count
AFTER INSERT ON entity
BEGIN
    UPDATE entity_types
    SET num_entities = num_entities + 1
    WHERE entity_type = NEW.entity_type;
END;

CREATE TRIGGER IF NOT EXISTS decrement_entity_type_count
AFTER DELETE ON entity
BEGIN
    UPDATE entity_types
    SET num_entities = MAX(num_entities - 1, 0)
    WHERE entity_type = OLD.entity_type;
END;

CREATE TRIGGER IF NOT EXISTS update_entity_type_count
AFTER UPDATE OF entity_type ON entity
WHEN OLD.entity_type <> NEW.entity_type
BEGIN
    UPDATE entity_types
    SET num_entities = MAX(num_entities - 1, 0)
    WHERE entity_type = OLD.entity_type;

    UPDATE entity_types
    SET num_entities = num_entities + 1
    WHERE entity_type = NEW.entity_type;
END;

/*********************************
----> Chunk to entity mapping <----
Purpose: This file contains the entity schema for mapping chunks to its associated entities in the database.
- chunk_id: This is the primary key for the chunk_to_entity_mapping table. It is a foreign key that references the chunk_id in the chunk table.
- entity_id: This is a foreign key that references the entity_id in the entity table.
*********************************/
CREATE TABLE IF NOT EXISTS chunk_to_entity_mapping (
    chunk_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE
);

/*********************************
----> DEPRECATED TABLES <----
Purpose: This file contains the schema for deprecated tables in the database. These are no longer used in the current version of the database, but are kept for backward compatibility.
- entity_extraction_jobs: This table is used to track the progress of entity extraction jobs.
- entity_deduplication_jobs: This table is used to track the progress of entity deduplication jobs.
*********************************/
CREATE TABLE IF NOT EXISTS entity_extraction_jobs (
    job_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    is_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tokens INT NOT NULL,
    completed_at TIMESTAMP,
    PRIMARY KEY (job_id),
    FOREIGN KEY (source_id) REFERENCES source(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity_deduplication_jobs (
    job_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    is_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tokens INT NOT NULL,
    completed_at TIMESTAMP,
    PRIMARY KEY (job_id),
    FOREIGN KEY (source_id) REFERENCES source(id) ON DELETE CASCADE
);


