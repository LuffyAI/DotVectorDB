
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


