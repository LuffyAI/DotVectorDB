
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
    PRIMARY KEY (chunk_id),
    FOREIGN KEY (source_id) REFERENCES source(id) ON DELETE CASCADE,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE CASCADE
);

/*********************************
----> Increment/Decrement triggers <----
Purpose: This these triggers are used to automatically update the num_chunks field in the source table when chunks are added or removed from the chunk table:
- increment_source_num_chunks: This trigger is fired after a chunk is inserted into the chunk table. It increments the num_chunks field in the source table for the corresponding source_id.
- decrement_source_num_chunks: This trigger is fired after a chunk is deleted from the chunk table. It decrements the num_chunks field in the source table for the corresponding source_id.
*********************************/
CREATE TRIGGER IF NOT EXISTS increment_source_num_chunks
AFTER INSERT ON chunk
BEGIN
    UPDATE source
    SET num_chunks = num_chunks + 1 
    WHERE id = NEW.source_id;
END;

CREATE TRIGGER IF NOT EXISTS decrement_source_num_chunks
AFTER DELETE ON chunk
BEGIN
    UPDATE source
    SET num_chunks = MAX(num_chunks - 1, 0)
    WHERE id = OLD.source_id;
END;
