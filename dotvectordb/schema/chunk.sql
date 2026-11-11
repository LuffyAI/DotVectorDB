/*********************************
----> Embedding Model <----
Purpose: This file contains the embedding_model schema for the database.
- id: This is the unique id for the embedding model
- name: This is the name of the embedding model
- description: This is a JSON object that contains the description of the embedding model. It can contain information such as the model's architecture, training data, and any other relevant details.
*********************************/
CREATE TABLE IF NOT EXISTS embedding_model (
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    description JSON,
    PRIMARY KEY (id)
);

INSERT OR IGNORE INTO global_config (
    id,
    current_version_count,
    description_version,
    description,
    DIM,
    embedding_model_id,
    num_chunks,
    num_entities,
    num_relations,
    total_chunk_tokens,
    total_entity_tokens,
    total_relationship_tokens,
    average_chunk_tokens,
    avg_entity_tokens,
    avg_relationship_tokens,
    avg_degree
)
VALUES (
    1, 0, 0, NULL, NULL, NULL, 0, 0, 0,
    0, 0, 0, 0.0, 0.0, 0.0, 0.0
);
/*********************************
----> Source <----
Purpose: This file contains the source schema for the database. A source is a document that is ingested into the system and can be chunked into smaller pieces for processing. Each source has a unique id, name, and description. The description is a JSON object that can contain information such as the source's origin, content, and any other relevant details.
- id: This is the unique id for the source
- name: This is the name of the source
- description: This is a JSON object that contains the description of the source. It can contain information such as the source's origin, content, and any other relevant details.
- created_at: This is a timestamp that records when the source was created. It is automatically set to the current timestamp when the source is inserted into the database.
- num_chunks: This is an integer that represents the number of chunks associated with the source. It is updated automatically by triggers when chunks are added or removed.
- num_entities: This is an integer that represents the number of entities associated with the source. It is updated automatically by triggers when entities are added or removed.
- num_relations: This is an integer that represents the number of relationships associated with the source. It is updated automatically by triggers when relationships are added or removed.
- raw_corpus_tokens: This is an integer that represents the number of tokens in the raw corpus before we extract entities and relationships from it.
- entity_construct_tokens: This is an integer that represents the number of tokens used when we use the LLM to extract entities from the raw corpus.
- entity_deduplication_tokens: This is deprecated and will be removed in the future. It was an integer that represented the number of tokens used when we use the LLM to deduplicate entities.
- relation_construct_tokens: This is an integer that represents the number of tokens used when we use the LLM to extract relationships from the raw corpus.
*********************************/
CREATE TABLE IF NOT EXISTS source (
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    description JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_chunked BOOLEAN DEFAULT FALSE,
    is_chunk_number_known BOOLEAN DEFAULT FALSE,
    num_of_chunks_found INTEGER DEFAULT 0,
    num_of_chunks_added INTEGER DEFAULT 0,
    num_chunks INT DEFAULT 0,
    num_entities INT DEFAULT 0,
    num_relations INT DEFAULT 0,
    raw_corpus_tokens INT DEFAULT 0,
    entity_construct_tokens INT DEFAULT 0,
    entity_deduplication_tokens INT DEFAULT 0,
    relation_construct_tokens INT DEFAULT 0,
    PRIMARY KEY (id)
);
/*********************************
----> Chunk <----
Purpose: This file contains the chunk schema for the database. A chunk represents a small slice of a source document. Each chunk has a unique id, a reference to the source it belongs to, the content of the chunk, and a reference to the embedding model used to generate its embedding. Additionally, there are boolean flags to indicate whether the chunk has been embedded, whether entities have been extracted from it, and whether relationships have been extracted from it.
- chunk_id: This is the unique id for the chunk.
- source_id: This is a foreign key that references the id of the source to which the chunk belongs.
- content: This is the actual content of the chunk, stored as text.
- embedding_model_id: This is a foreign key that references the id of the embedding model used to generate the embedding for the chunk.
- is_embedded: This is a boolean flag that indicates whether the chunk has been embedded or not.
- are_entities_extracted: This is a boolean flag that indicates whether entities have been extracted from the chunk or not.
- are_relations_extracted: This is a boolean flag that indicates whether relationships have been extracted from the chunk or not.
*********************************/
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    is_embedded BOOLEAN DEFAULT FALSE,
    are_entities_extracted BOOLEAN DEFAULT FALSE,
    are_relations_extracted BOOLEAN DEFAULT FALSE,
    tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (chunk_id),
    FOREIGN KEY (source_id) REFERENCES source(id) ON DELETE CASCADE,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk_to_sentence_mapping (
    chunk_id INTEGER NOT NULL,
    sentence_number INTEGER NOT NULL,
    sentence_text TEXT NOT NULL,
    PRIMARY KEY (chunk_id, sentence_number),
    FOREIGN KEY (chunk_id)
        REFERENCES chunk(chunk_id)
        ON DELETE CASCADE
);

/*********************************
----> Increment/Decrement triggers <----
Purpose: This these triggers are used to automatically update the num_chunks field in the source table when chunks are added or removed from the chunk table:
Chunk inserts and deletes update counts and token statistics without changing the graph version.
- increment_source_num_chunks: This trigger is fired after a chunk is inserted into the chunk table. It increments the num_chunks field in the source table for the corresponding source_id.
- decrement_source_num_chunks: This trigger is fired after a chunk is deleted from the chunk table. It decrements the num_chunks field in the source table for the corresponding source_id.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_source_num_chunks
AFTER INSERT ON chunk
BEGIN
    UPDATE source
    SET num_chunks = num_chunks + 1,
        num_of_chunks_added = num_of_chunks_added + 1
    WHERE id = NEW.source_id;

    UPDATE global_config
    SET average_chunk_tokens =
            CAST(
                COALESCE(total_chunk_tokens, 0) + NEW.tokens
                AS REAL
            ) / (COALESCE(num_chunks, 0) + 1),
        total_chunk_tokens =
            COALESCE(total_chunk_tokens, 0) + NEW.tokens,
        num_chunks = COALESCE(num_chunks, 0) + 1
    WHERE id = 1;

END;

CREATE TRIGGER IF NOT EXISTS decrement_source_num_chunks
AFTER DELETE ON chunk
BEGIN
    UPDATE source
    SET num_chunks = MAX(num_chunks - 1, 0)
    WHERE id = OLD.source_id;

    UPDATE global_config
    SET average_chunk_tokens =
            CASE
                WHEN COALESCE(num_chunks, 0) <= 1 THEN 0.0
                ELSE CAST(
                    MAX(
                        COALESCE(total_chunk_tokens, 0) - OLD.tokens,
                        0
                    ) AS REAL
                ) / (num_chunks - 1)
            END,
        total_chunk_tokens = MAX(
            COALESCE(total_chunk_tokens, 0) - OLD.tokens,
            0
        ),
        num_chunks = MAX(COALESCE(num_chunks, 0) - 1, 0)
    WHERE id = 1;

END;

CREATE TRIGGER IF NOT EXISTS update_global_chunk_token_stats
AFTER UPDATE OF tokens ON chunk
WHEN NEW.tokens <> OLD.tokens
BEGIN
    UPDATE global_config
    SET average_chunk_tokens =
            CASE
                WHEN COALESCE(num_chunks, 0) = 0 THEN 0.0
                ELSE CAST(
                    COALESCE(total_chunk_tokens, 0)
                    + NEW.tokens
                    - OLD.tokens
                    AS REAL
                ) / num_chunks
            END,
        total_chunk_tokens = MAX(
            COALESCE(total_chunk_tokens, 0)
            + NEW.tokens
            - OLD.tokens,
            0
        )
    WHERE id = 1;
END;

-- Inspect this chunk's mappings before foreign-key cascades remove them.
CREATE TRIGGER IF NOT EXISTS delete_unsupported_chunk_graph
BEFORE DELETE ON chunk
BEGIN
    DELETE FROM relationship
    WHERE relation_id IN (
        SELECT relation_id FROM chunk_to_relationship_mapping
        WHERE chunk_id = OLD.chunk_id
    )
    AND NOT EXISTS (
        SELECT 1 FROM chunk_to_relationship_mapping AS support
        WHERE support.relation_id = relationship.relation_id
          AND support.chunk_id <> OLD.chunk_id
    );

    UPDATE relationship
    SET supporting_lines = json_remove(
        supporting_lines, '$."' || OLD.chunk_id || '"'
    )
    WHERE relation_id IN (
        SELECT relation_id FROM chunk_to_relationship_mapping
        WHERE chunk_id = OLD.chunk_id
    )
    AND json_type(supporting_lines, '$."' || OLD.chunk_id || '"') IS NOT NULL;

    DELETE FROM entity
    WHERE entity_id IN (
        SELECT entity_id FROM chunk_to_entity_mapping
        WHERE chunk_id = OLD.chunk_id
    )
    AND NOT EXISTS (
        SELECT 1 FROM chunk_to_entity_mapping AS support
        WHERE support.entity_id = entity.entity_id
          AND support.chunk_id <> OLD.chunk_id
    )
    AND NOT EXISTS (
        SELECT 1 FROM relationship
        WHERE src_entity_id = entity.entity_id
           OR dst_entity_id = entity.entity_id
    );

    -- Entity deletions above already adjusted their source counts. Account
    -- for remaining mappings now, before this chunk disappears in the cascade.
    UPDATE source
    SET num_entities = MAX(num_entities - (
        SELECT COUNT(*)
        FROM chunk_to_entity_mapping AS mapping
        WHERE mapping.chunk_id = OLD.chunk_id
          AND NOT EXISTS (
              SELECT 1
              FROM chunk_to_entity_mapping AS other
              JOIN chunk ON chunk.chunk_id = other.chunk_id
              WHERE other.entity_id = mapping.entity_id
                AND other.chunk_id <> OLD.chunk_id
                AND chunk.source_id = OLD.source_id
          )
    ), 0)
    WHERE id = OLD.source_id;
END;
