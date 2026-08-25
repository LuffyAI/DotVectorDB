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
    FOREIGN KEY (embedding_model_id) REFERENCES embedding_model(id) ON DELETE SET NULL
);

