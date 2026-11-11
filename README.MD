# DotVectorDB

**A SQLite-based vector and graph database for building GraphRAG pipelines.**

DotVectorDB turns documents into a graph of entities and relationships, keeps relationships linked to their supporting text, and provides vector, BM25, and hybrid retrieval. Compressed Roaring bitmap indexes narrow searches to an anchor's neighborhood and entity types before ranking the candidates.

The database, graph construction, and retrieval components are separate Python classes. You can use the search primitives directly or build a retrieval algorithm on top of them.

## What it does

- **Document ingestion:** extract text from TXT, PDF, DOCX, Excel, and PowerPoint files, then chunk it with a configurable chunker.
- **Graph construction:** extract entities and relationships, assign entity types, and embed graph content. Extraction prompts and examples can be customized.
- **Evidence tracking:** relationships retain supporting line references for each source chunk, so retrieved facts can be traced back to stored text.
- **Search:** vector search through SQLite Vec1, BM25 through FTS5, and hybrid search using reciprocal rank fusion.
- **Constrained retrieval:** retrieve entity or relationship IDs by anchor, hop, and entity type, then rank only those IDs.
- **Incremental construction:** track new and updated graph records, extend bitmap indexes, and resume completed insertion and bitmap batches.
- **Reusable embeddings:** supply an existing embedding directly to vector or hybrid search, or retrieve stored embeddings as NumPy arrays.

SQLite stores documents, graph records, metadata, and embeddings. Roaring bitmap indexes are stored separately in LMDB.

## Install from source

The native build currently targets **Linux**. Use a C/C++ compiler with OpenMP support and `make`. Python 3.10 and 3.11 have been used during development.

On Debian or Ubuntu, install the build tools:

```bash
sudo apt install build-essential
```

Then install the library:

```bash
git clone https://github.com/LuffyAI/DotVectorDB.git
cd DotVectorDB
python -m pip install .
```

The package build compiles Vec1 from the bundled source. LMDB and PyRoaring are installed as Python dependencies. See [BUILD_LOCAL.md](BUILD_LOCAL.md) for local wheel-build commands.

## Build a graph

The default providers use OpenAI. Create a `.env` file with your credentials and model settings, and load it **before importing the library**:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o-mini
TOKENIZER=gpt-4o
```

```python
import asyncio
from dotenv import load_dotenv

load_dotenv()

from dotvectordb.UnstructuredGraphConstructor import UnstructuredGraphConstructor


async def main():
    async with UnstructuredGraphConstructor(
        path="./article_graph",
        tokenizer_name="gpt-4o",
    ) as graph:
        await graph.construct_GraphRAG(
            input_document_list=["./article.txt"],
            global_desc=False,
        )


asyncio.run(main())
```

This ingests and embeds chunks, builds the entity and relationship graph, creates reachability indexes, and builds graph BM25 indexes. `global_desc=False` skips LLM-generated global descriptions. Graph construction still calls the configured language and embedding models.

The path above creates a database directory containing `article_graph.db` and its graph index artifacts. Reuse the same path when opening a retriever.

## Search the graph

```python
from dotvectordb.GraphRetriever import GraphRetriever


async def search():
    async with GraphRetriever(
        path="./article_graph",
        tokenizer_name="gpt-4o",
        bitmap_mode="exact_hop",
    ) as retriever:
        results, query_tokens = await retriever.hybrid_vector_BM25_search(
            query="What makes aviation difficult to decarbonize?",
            table="relationship",
            top_k=5,
        )
        return results
```

Search tables are `chunk`, `entity`, and `relationship`. The return value is `(results, query_tokens)`.

To reuse embeddings rather than generate a new query embedding:

```python
embeddings = await retriever.batch_get_embeddings(
    ids={entity_id},
    table="entity",
)

results, query_tokens = await retriever.vector_search(
    query=None,
    table="entity",
    embedding=embeddings[entity_id],
    top_k=5,
)
```

Supplied embeddings must use the same model and dimensions as the target index. Hybrid search also accepts `embedding=...`; its BM25 component still requires query text.

## Search a neighborhood

Precomputed exact-hop buckets provide direct lookups for the indexed hop and entity type. They return database IDs that can be passed to `search_id_batch`:

```python
relationship_ids = await retriever.bitmap_retriever.constrained_relationship_search(
    anchor_entity_id=anchor_id,
    exact_hop=2,
    entity_types=["person"],
)

results, query_tokens = await retriever.search_id_batch(
    query="Who worked on these projects?",
    table="relationship",
    rowids=relationship_ids,
    top_k=10,
    search_type="hybrid",
)
```

Entities can belong to multiple types. Type filters match any requested type by default; `match_all_types=True` requires all requested types.

Precomputed hops do not expand the graph during lookup. Requests beyond the precomputed range use one-hop traversal for the remaining distance. Choose `bitmap_mode="one_hop"` to use traversal directly, or `load_all_into_memory=True` to cache the bitmap stores in memory. A memory-loaded bitmap retriever can pick up index changes with `await retriever.bitmap_retriever.refresh()`.

## Document-only retrieval

Use `NaiveVDBConstructor` when you want document retrieval without entity or relationship extraction:

```python
from dotvectordb.NaiveVDBConstructor import NaiveVDBConstructor


async def build_document_index():
    async with NaiveVDBConstructor(
        path="./document_index",
        tokenizer_name="gpt-4o",
    ) as database:
        await database.construct_document_bm25_index(
            input_document_list=["./article.txt"],
        )
```

`construct_document_bm25_index` skips chunk embeddings. Database initialization currently still configures the default models and may generate an embedding to establish dimensions for a fresh database.

The same constructor also provides `construct_document_vectordatabase` and `construct_bm25_and_vdb`.

## Code layout

| Component | Responsibility |
|---|---|
| `DotVectorDB` | SQLite connection, schema, metadata, and embedding storage |
| `UnstructuredGraphConstructor` | Coordinate document ingestion and graph construction |
| `NaiveVDBConstructor` | Document-only BM25 and vector index construction |
| `GraphRetriever` | BM25, vector, hybrid, and ID-constrained search |
| `RetrievalAlgos/` | Retrieval algorithms built on the search primitives; several are still in development |
| `plugins/` | Document extraction, chunking, hashing, and model providers |
| `data_structures/bitmap/` | Roaring bitmap construction and retrieval backed by LMDB |
| `schema/` | SQL tables, indexes, and triggers |

Custom chunkers, hashing functions, embedding models, and language models can be supplied through the corresponding constructor arguments.

## Native extension details

The bundled source is pinned to **Vec1 0.8**, independently of the SQLite engine version:

- Source: `dotvectordb/compiled_vec1/original_files/ext/vec1/vec1.c`
- SQLite source revision: `213d76c009bcd860b5a624be6ffe3b89aab063bc6f2c30bdb40384c5d7f8b35f`
- `vec1.c` SHA-256: `b341c15fe10e41a9b96981ff3a9c217ea0774b4e04932b5acf3dac1320aeceb1`

Matching SQLite headers are bundled. Compilation requires no Vec1 download. After loading the extension, `SELECT vec1_info();` reports its version and enabled features.

The default Vec1 compiler flags include `-march=native`: build on the target machine rather than distributing that wheel across different CPUs. Compiler flags can be overridden through `CC`, `CPPFLAGS`, `CFLAGS`, and `LDFLAGS`.

## Status

DotVectorDB is a research project under active development. APIs and schemas can change between revisions. Performance depends on graph structure, storage, models, and the index configuration; this README does not claim a universal benchmark result.
