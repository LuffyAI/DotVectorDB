/*********************************
----> Query Embedding Cache <----
Purpose: Persist embeddings for exact retrieval queries so repeated searches
do not call the embedding provider again. The model ID and original query are
part of the key so model changes and the unlikely event of a hash collision do
not return an incompatible embedding.
*********************************/
CREATE TABLE IF NOT EXISTS query_embeddings (
    query_hash INTEGER NOT NULL,
    embedding_model_id TEXT NOT NULL,
    query TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (query_hash, embedding_model_id, query)
);
