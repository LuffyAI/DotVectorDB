CREATE TABLE IF NOT EXISTS entity_types (
    entity_type TEXT NOT NULL,
    type_description TEXT NOT NULL,
    num_entities INT DEFAULT 0,
    PRIMARY KEY (entity_type)
);

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

CREATE TABLE IF NOT EXISTS chunk_to_entity_mapping (
    chunk_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entity(entity_id) ON DELETE CASCADE
);

