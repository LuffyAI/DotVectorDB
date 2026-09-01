/*********************************
---->  SETTINGS <----
Purpose: This file contains the settings for the database.
- journal_mode: WAL (Write-Ahead Logging) is enabled for better concurrency and performance.
- busy_timeout: Set to 10000 milliseconds (10 seconds) to handle database locks gracefully.
- temp_store: Set to MEMORY to store temporary tables in memory for faster access.
- mmap_size: Set to 30GB to allow memory-mapped I/O for large databases
- cache_size: Set to -200000 pages (negative value indicates size in KB) to control the number of pages in the cache.
- foreign_keys: Enabled to enforce foreign key constraints for data integrity.
*********************************/
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 10000;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 30000000000;
PRAGMA cache_size = -200000;
PRAGMA foreign_keys = ON;
/*********************************
----> global_config <----
Purpose: This file contains the schema for the global_config metadata in the database.
- id: This ensures the SQL table strictly holds a single row and column.
- current_version_count: After every write operation to the database, this integer is incremented by 1.
- description_version: This is the version of the database when the global description was last updated. It is used to determine if the global description needs to be updated.
- description: The global description serves to provide a global paragraph description of the contents the database holds.
- DIM: This is the dimension of the embedding model used in the database.
- embedding_model_id: This is the ID of the embedding model used to generate embeddings for the database. It is a foreign key that references the id in the embedding_model table.
*********************************/
CREATE TABLE IF NOT EXISTS global_config (
    id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    current_version_count INTEGER NOT NULL,
    description_version INTEGER NOT NULL,
    description JSON,
    DIM INTEGER,
    embedding_model_id INTEGER,
    num_chunks INTEGER,
    num_entities INTEGER,
    num_relations INTEGER,
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE SET NULL
);
/*********************************
----> llm_token_usage <----
Purpose: This file contains the schema for the llm_token_usage metadata in the database.
- id: This is the primary key for the table and is auto-incremented.
- database_version: This is the version of the database when the LLM token usage was recorded. It is used to track the token usage over time and is set to the current_version_count from the global_config table.
- stage: This is a text field that indicates the stage of the construction pipeline, such as "entity", "relation", "global_desc" or "retrieval".
- total_tokens: This is an integer that records the total number of tokens used by the LLM during the specified stage. It is used to monitor and manage the token usage for cost and performance optimization.
*********************************/
CREATE TABLE IF NOT EXISTS llm_token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    database_version INTEGER,
    stage TEXT NOT NULL,
    total_tokens INTEGER NOT NULL
);

CREATE TRIGGER set_llm_token_usage_version
AFTER INSERT ON llm_token_usage
FOR EACH ROW
BEGIN
    UPDATE llm_token_usage
    SET database_version = (
        SELECT current_version_count
        FROM global_config
        WHERE id = 1
    )
    WHERE id = NEW.id;
END;