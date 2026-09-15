
/*********************************
----> Entity <----
Purpose: This file contains the entity schema for the database.
- entity_id: This is the primary key for the entity table. To get this value programmatically, the entity_id is a hash between the entity name & type.
- entity_name: This is the name of the entity.
- entity_description: This is a text description of the entity.
- questions: This is a JSON array of questions we suspect a user would ask about the entity. (e.g, for about Shrek, a question could be "Who is the green ogre in the story?" The answer is the entity.
- degrees: This is the number of connections an entity has to other entities. Its 1-hop neighbors in the graph.
- embedding_model_id: This is the ID of the embedding model used to generate embeddings for the entity. It is a foreign key that references the id in the embedding_model table.
- is_embedded: This is a boolean flag that indicates whether the entity has been embedded or not.
- is_deduplicated: This is deprecated and will be removed in the future.
- tokens: This is the number of tokens used to generate the entity.
*********************************/
CREATE TABLE IF NOT EXISTS temp_entity (
    temp_entity_id INTEGER NOT NULL,
    entity_name TEXT NOT NULL,
    entity_tags TEXT NOT NULL,
    degrees INTEGER DEFAULT 0,
    PRIMARY KEY (temp_entity_id)
);

CREATE TABLE IF NOT EXISTS chunk_to_temp_entity_mapping (
    chunk_id INTEGER NOT NULL,
    temp_entity_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, temp_entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (temp_entity_id) REFERENCES temp_entity(temp_entity_id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS entity (
    entity_id INTEGER NOT NULL,
    entity_name TEXT NOT NULL,
    entity_description TEXT NOT NULL,
    questions JSON NOT NULL,
    alias JSON,
    degrees INTEGER DEFAULT 0,
    embedding_model_id TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    is_deduplicated BOOLEAN DEFAULT FALSE,
    tokens INT NOT NULL,
    PRIMARY KEY (entity_id),
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity_types (
    entity_type TEXT NOT NULL,
    type_description TEXT NOT NULL,
    num_entities INTEGER DEFAULT 0,
    PRIMARY KEY (entity_type)
);
/*********************************
----> Entity Tags <----
Purpose: This file contains the entity schema for entity tags in the database. Tags allow for the LLM to discover important entities during construction.
- entity_type: This is the primary key for the entity_tags table. It is a text field that represents the type of the entity.
- num_entities: This is an integer that represents the number of entities of this type in the database. It is updated automatically by triggers when entities are added or removed.
*********************************/
CREATE TABLE IF NOT EXISTS entity_tags (
    entity_tag TEXT NOT NULL,
    num_of_instances INT DEFAULT 0,
    PRIMARY KEY (entity_tag)
);
/*********************************
----> Entity to entity tag mapping <----
Purpose: When DotRAG is discovering entity types, first proposals are called tags.
0 entity_id: This is the primary key for the entity_to_entity_type_mapping table. It is a foreign key that references the entity_id in the entity table.
1 entity_tag: This is a foreign key that references the entity_type in the entity_types table
*********************************/
CREATE TABLE IF NOT EXISTS entity_to_entity_tag_mapping (
    entity_id INTEGER NOT NULL,
    entity_tag TEXT NOT NULL,
    PRIMARY KEY (entity_id, entity_tag),
    FOREIGN KEY (entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_tag) REFERENCES entity_tags(entity_tag) ON DELETE CASCADE
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
    FOREIGN KEY (entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_type) REFERENCES entity_types(entity_type) ON DELETE CASCADE
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
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE
);
/*********************************
----> Increment/Decrement triggers <----
Purpose: Keep global entity counts and per-type entity counts synchronized with
the entity table and its many-to-many type mapping table.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_global_entity_count
AFTER INSERT ON entity
BEGIN
    UPDATE global_config
    SET num_entities = MAX(COALESCE(num_entities, 0) + 1, 0),
        current_version_count = current_version_count + 1
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS decrement_global_entity_count
AFTER DELETE ON entity
BEGIN
    UPDATE global_config
    SET num_entities = MAX(COALESCE(num_entities, 0) - 1, 0),
        current_version_count = current_version_count + 1
    WHERE id = 1;
END;

CREATE TRIGGER IF NOT EXISTS increment_entity_type_count
AFTER INSERT ON entity_to_entity_type_mapping
BEGIN
    UPDATE entity_types
    SET num_entities = num_entities + 1
    WHERE entity_type = NEW.entity_type;

    UPDATE global_config
    SET current_version_count = current_version_count + 1;
END;

CREATE TRIGGER IF NOT EXISTS decrement_entity_type_count
AFTER DELETE ON entity_to_entity_type_mapping
BEGIN
    UPDATE entity_types
    SET num_entities = MAX(num_entities - 1, 0)
    WHERE entity_type = OLD.entity_type;

    UPDATE global_config
    SET current_version_count = current_version_count + 1;
END;

CREATE TRIGGER IF NOT EXISTS update_entity_type_count
AFTER UPDATE OF entity_type ON entity_to_entity_type_mapping
WHEN OLD.entity_type <> NEW.entity_type
BEGIN
    UPDATE entity_types
    SET num_entities = MAX(num_entities - 1, 0)
    WHERE entity_type = OLD.entity_type;

    UPDATE entity_types
    SET num_entities = num_entities + 1
    WHERE entity_type = NEW.entity_type;

    UPDATE global_config
    SET current_version_count = current_version_count + 1;
END;
