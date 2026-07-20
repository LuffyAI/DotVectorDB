# DotVectorDB

DotVectorDB is an asynchronous vector and graph database built for research paper **DotRAG: Retrieval-Time Reasoning Along Paths**. It combines document ingestion, text chunking, embeddings, SQLite vector search, entity extraction, relationship extraction, and graph-based retrieval.

## Features

- Asynchronous SQLite database access
- Vector storage and similarity search
- GraphRAG construction from document collections
- Entity and relationship extraction
- Temporary subgraph-constrained vector spaces
- Custom embedding, language-model, chunking, and hashing providers
- Resumable document embedding
- K-nearest-neighbor and approximate vector search
- Document extraction from TXT, PDF, DOCX, XLSX, XLSM, and PPTX files

## Requirements

- Python 3.9 or newer
- Linux, including Ubuntu
- API credentials for the embedding and language-model providers being used

## Installation

After DotVectorDB is published to PyPI:

```bash
pip install dotvectordb
```

To install directly from the project directory:

```bash
pip install .
```

For development, install it in editable mode:

```bash
pip install -e .
```

## Environment Variables

Create a `.env` file in your application when using API-based providers.

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key

OPENAI_TOKENIZER=gpt-4o
TOKENIZER=gpt-4o
```

Only configure the keys and provider settings used by your application.

## Basic Usage

DotVectorDB uses an asynchronous API.

```python
import asyncio

from dot_vectordb.DotVDB import DotVectorDB


async def main():
    database = DotVectorDB(
        path="dotrag.db",
        DIM=1536,
    )

    await database.open()

    try:
        await database.construct_GraphRAG(
            input_document_list=["./documents"],
            verbose=True,
        )

        results, token_count = await database.vector_search(
            query="What are the main ideas in these documents?",
            table="chunk",
            top_k=5,
        )

        for result in results:
            print(result)

        print("Query tokens:", token_count)
    finally:
        await database.close()


asyncio.run(main())
```

The `DIM` value must match the output dimension of the selected embedding model.

The paths inside `input_document_list` should point to directories containing supported documents.

## Opening and Closing the Database

```python
database = DotVectorDB(
    path="dotrag.db",
    DIM=1536,
)

await database.open()

# Use the database here.

await database.close()
```

Always close the database after use.

## Ingesting Documents

```python
await database.ingest_documents(
    input_document_list=[
        "./documents",
        "./more_documents",
    ],
    requests_per_minute=1000,
    async_cooldown=60,
    verbose=True,
)
```

Document ingestion performs the following steps:

1. Extracts text from supported documents.
2. Splits extracted text into chunks.
3. Stores chunk metadata.
4. Generates embeddings.
5. Stores embeddings in SQLite.
6. Retries unfinished embeddings in later batches.

## Constructing a GraphRAG Database

```python
await database.construct_GraphRAG(
    input_document_list=["./documents"],
    entity_types=None,
    requests_per_minute=1000,
    async_cooldown=60,
    verbose=True,
)
```

GraphRAG construction:

1. Ingests and embeds document chunks.
2. Adds entity types.
3. Extracts and embeds entities.
4. Extracts and embeds relationships.
5. Prepares the database for vector and graph retrieval.

Custom entity types can be supplied as a dictionary:

```python
entity_types = {
    "PERSON": "A named individual.",
    "ORGANIZATION": "A company, institution, or group.",
    "TECHNOLOGY": "A software tool, model, framework, or system.",
}

await database.construct_GraphRAG(
    input_document_list=["./documents"],
    entity_types=entity_types,
    verbose=True,
)
```

## Vector Search

```python
results, token_count = await database.vector_search(
    query="Explain the relationship between the main entities.",
    table="chunk",
    top_k=10,
    KNN=True,
    distance_metric="cos",
)
```

Supported distance metrics:

- `"cos"` for cosine distance
- `"l2"` for Euclidean distance

Set `KNN=False` to use the approximate search path with reranking.

## Custom Providers

DotVectorDB defines interfaces for its primary components:

- `EmbeddingModel`
- `LanguageModel`
- `ChunkingFunc`
- `HashingFunc`

Custom implementations can be passed into `DotVectorDB`:

```python
database = DotVectorDB(
    path="dotrag.db",
    DIM=my_embedding_model.embedding_dim,
    embedding_model=my_embedding_model,
    language_model=my_language_model,
    chunking_func=my_chunker,
    hashing_func=my_hashing_function,
)
```

Each custom provider must implement the methods required by its corresponding interface.

## Fixed-Size Chunking

DotVectorDB includes a fixed-size token chunker with overlapping chunks.

```python
from dot_vectordb.plugins.chunk_functions.FixedSizeChunker import FixedSizeChunker
from dot_vectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc


chunker = FixedSizeChunker(
    hash_func=SHA256HashingFunc(),
    max_token_size=1024,
    overlap_token_size=128,
)
```

Each generated chunk includes:

- A chunk index
- A token count
- The chunk content
- A content hash

## Document Support

| Document type | Extension |
|---|---|
| Plain text | `.txt` |
| PDF | `.pdf` |
| Microsoft Word | `.docx` |
| Microsoft Excel | `.xlsx`, `.xlsm` |
| Microsoft PowerPoint | `.pptx` |

## Linux Support

DotVectorDB currently loads the compiled SQLite vector extension from:

```text
dot_vectordb/compiled_vec1/vec1.so
```

Because `.so` is a Linux shared-library format, the current release is intended for Linux systems such as Ubuntu.

## Running Tests

```bash
pytest
```

## Project Status

DotVectorDB is under active development as the vector and graph storage layer for DotRAG. APIs, schemas, and provider implementations may change before a stable release.

## Author

Larnell Moore

## License

Add the project's license before publishing the package.