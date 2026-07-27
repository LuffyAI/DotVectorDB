from dotvectordb.interfaces.EmbeddingModel import EmbeddingModel
from dotvectordb.interfaces.LanguageModel import LanguageModel
from dotvectordb.interfaces.ChunkingFunc import ChunkingFunc
from dotvectordb.interfaces.HashingFunc import HashingFunc
from dotvectordb.plugins.embedding_providers.openai_embeddings import OpenAIEmbeddingModel
from dotvectordb.plugins.llm_providers.openai_llms import OpenAiAgent
from dotvectordb.plugins.chunk_functions.FixedSizeChunker import FixedSizeChunker
from dotvectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc
from dotvectordb.plugins.loaders.TextConverter import DocumentTextExtractor
from dotvectordb.prompts.entity_prompts import DEFAULT_ENTITY_TYPES, isEntityTypesFormatCorrect
from dotvectordb.graph.entity_indexer import EntityIndexer
from dotvectordb.graph.relatonship_indexer import RelationshipIndexer
from dotvectordb.graph.graph_retriever import GraphRetriever
from dotvectordb.graph.TempVectorSpace import AsyncTempVectorSpace
import os
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import json
import aiosqlite
import asyncio
import traceback
BASE_DIR = Path(__file__).resolve().parent
class DotVectorDB:
    def __init__(self, path: str, DIM: int, 
                 embedding_model: EmbeddingModel = OpenAIEmbeddingModel() , 
                 language_model: LanguageModel = OpenAiAgent(), 
                 chunking_func: ChunkingFunc = FixedSizeChunker(hash_func=SHA256HashingFunc()),
                 hashing_func: HashingFunc = SHA256HashingFunc(),
                 tokenizer_name: str = os.getenv("TOKENIZER"),
                 verbose=False):
        self.path = path
        self.DIM = DIM
        self.tokenizer_name = tokenizer_name
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.chunking_func = chunking_func
        self.hashing_func = hashing_func
        self.conn: aiosqlite.Connection | None = None
        self.lock = asyncio.Lock()
############################################################
#   Functions For Opening and Closing DotVDB               # 
############################################################
    async def open(self):
        self.conn = await aiosqlite.connect(self.path)
        def setup():
            try:
                # Get the raw connection to set up the database schema
                raw = self.conn._conn
                # Set PRAGMA settings for performance
                pragma_sql = (BASE_DIR / "schema" / "settings.sql").read_text(
                encoding="utf-8"
                )
                raw.executescript(pragma_sql)
                # Loads the compiled vec1 extension for vector operations
                raw.enable_load_extension(True)
                raw.load_extension(
                str(BASE_DIR / "compiled_vec1" / "vec1.so")
                )
                raw.enable_load_extension(False)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the setup() function inside DotVectorDB's open function: {e}")
        await self.conn._execute(setup)
        await self.initialize_default_schema()
        replace_period_with_underscore = self.path.replace(".", "_")
        self.graph_path = f"""{replace_period_with_underscore}_graph.pkl"""
        self.graph_retriever = GraphRetriever(graph_path=f"""{self.graph_path}""", conn=self.conn, lock=self.lock)
        return self
    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
############################################################
#   Functions For Manipulating DotVDB's Schema             #
############################################################
    async def initialize_default_schema(self):
        try:
            if not self.conn:
                raise RuntimeError("Database connection is not open.")
            chunks_sql = (BASE_DIR / "schema" / "chunk.sql").read_text(encoding="utf-8")
            entities_sql = (BASE_DIR / "schema" / "entity.sql").read_text(encoding="utf-8")
            relationships_sql = (BASE_DIR / "schema" / "relationship.sql").read_text(encoding="utf-8")
            async with self.lock:
                await self.conn.executescript(chunks_sql)
                await self.conn.executescript(entities_sql)
                await self.conn.executescript(relationships_sql)
                await self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the initialize_default_schema function inside DotVectorDB: {e}")
        
    async def add_entity_types(self, entity_types=None, verbose=False):
        if not self.conn:
            raise RuntimeError("Database not opened")
        try:
            existing_entity_types = await self.get_entity_types()
            if entity_types is None or await isEntityTypesFormatCorrect(entity_types) is False:
                if verbose:
                    print("No valid entity types provided. Using default entity types.")
                entity_types = DEFAULT_ENTITY_TYPES
            if verbose:
                print("Current entity types:")
                print(json.dumps(entity_types, indent=4))
            if existing_entity_types:
                async with self.lock:
                    for entity_type in entity_types:
                        if entity_type not in [et[0] for et in existing_entity_types]:
                            await self.conn.execute(
                                """
                                INSERT OR IGNORE INTO entity_types(entity_type, type_description)
                                VALUES (?, ?)
                                """,
                                (entity_type, entity_types[entity_type])
                            )
            else:
                if verbose:
                    print("No existing entity types found in the database. Adding default entity types.")
                async with self.lock:
                    for entity_type in entity_types:
                        await self.conn.execute(
                            """
                            INSERT OR IGNORE INTO entity_types(entity_type, type_description)
                            VALUES (?, ?)
                            """,
                            (entity_type, entity_types[entity_type])
                        )
            await self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the add_entity_types function inside DotVectorDB: {e}")
        
    async def get_entity_types(self):
        """Gets entity types from the database and returns them as a dictionary"""
        if not self.conn:
            raise RuntimeError("Database not opened")
        try: 
            async with self.lock:
                cursor = await self.conn.execute(
                """
                SELECT entity_type, type_description
                FROM entity_types
                """
            )
            existing_entity_types = await cursor.fetchall()
            await cursor.close()
            if existing_entity_types is None:
                return {}
            entity_types_dict = {et[0]: et[1] for et in existing_entity_types}
            return entity_types_dict
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the get_entity_types function inside DotVectorDB: {e}")

    async def add_tables(self,schema_list):
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        async with self.lock:
            for schema in schema_list:
                await self.conn.executescript(schema)
        await self.conn.commit()

    async def remove_tables(self, table_names, verbose=False):
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        try:
            async with self.lock:
                    visited = set()
                    async def drop_table(table_name):
                        if table_name in visited:
                            return
                        visited.add(table_name)
                        cursor = await self.conn.execute("""
                            SELECT name
                            FROM sqlite_master
                            WHERE type = 'table'
                            AND name NOT LIKE 'sqlite_%'
                        """)
                        for (other_table,) in await cursor.fetchall():
                            fk_cursor = await self.conn.execute(
                                f'PRAGMA foreign_key_list("{other_table}")'
                            )
                            if any(row[2] == table_name for row in await fk_cursor.fetchall()):
                                await drop_table(other_table)

                        await self.conn.execute(
                            f'DROP TABLE IF EXISTS "{table_name}"'
                        )
                        if verbose:
                            print(f"Table {table_name} dropped.")
                    for table_name in table_names:
                        await drop_table(table_name)

                    await self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the remove_tables function inside DotVectorDB: {e}")

    async def display_tables(self):
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        async with self.lock:
            cursor = await self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = await cursor.fetchall()
            for table in tables:
                table_name = table[0]
                print(f"Table: {table_name}")
                cursor = await self.conn.execute(f"PRAGMA table_info({table_name});")
                columns = await cursor.fetchall()
                for column in columns:
                    print(f"  Column: {column[1]} (Type: {column[2]})")
############################################################
#   Functions For Adding Content to the Database           # 
############################################################
    async def add_metadata_content(self, table, name):
        if not self.conn:
            raise RuntimeError("Database not opened")
        try:
            source_hash = f"""{self.hashing_func.content_hash(name)}"""
            async with self.lock:
                await self.conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {table}(id, name)
                    VALUES (?, ?)
                    """, (source_hash, name)
                )
                await self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the add_metadata_content function inside DotVectorDB: {e}")

    async def add_chunk(self, content, source_name, model_name):
        if not self.conn:
            raise RuntimeError("Database not opened")
        content_hash = f"""{self.hashing_func.content_hash(content)}"""
        source_hash = f"""{self.hashing_func.content_hash(source_name)}"""
        model_hash = f"""{self.hashing_func.content_hash(model_name)}"""
        await self.add_metadata_content(table="source", name=source_name)
        await self.add_metadata_content(table="embedding_model", name=model_name)
        try: 
            async with self.lock:
                await self.conn.execute(
                    """
                    INSERT OR IGNORE INTO chunk (chunk_id, source_id, content, embedding_model_id)
                    VALUES (?,?,?,?)
                    """,
                    (content_hash, source_hash, content, model_hash)
                )
                await self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error while adding a chunk")
    ############################################################
    #   Functions For DotVDB's Vector Ingestion + Search       # 
    ############################################################
    async def add_embedding(self, table, content, pk_hash):
            # Check if the database connection is open
            if not self.conn:
                raise RuntimeError("Database not opened")
            # Checks whether the base table exists by looking it up in SQLite's internal schema catalog
            # Then, it checks the table has a valid simple primary key by using the pragma_table_info function to inspect the table's schema
            CHECK_BASE_TABLE_AND_PRIMARY_KEY_SQL = f"""
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM sqlite_master
                        WHERE type = 'table'
                        AND name = '{table}'
                    ) AS table_exists,

                    (
                        SELECT COUNT(*) = 1
                        FROM pragma_table_info('{table}')
                        WHERE pk > 0
                        AND UPPER(type) = 'INTEGER'
                    ) AS valid_primary_key,

                    EXISTS (
                        SELECT 1
                        FROM pragma_table_info('{table}')
                        WHERE name = 'is_embedded'
                    ) AS has_is_embedded_column;
                """

            CREATE_EMBEDDING_TABLE_SQL = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table}_embeddings
            USING vec1(embedding);
            """
            INSERT_INTO_EMBEDDING_TABLE_SQL = f"""
            INSERT INTO {table}_embeddings(rowid, embedding)
            VALUES (?, ?)
            """
            UPDATE_TABLE_STATUS = f"""
            UPDATE {table}
            SET is_embedded = 1
            WHERE rowid = ?
            """
            async with self.lock:
                # First, we check if the base table exists and has a valid primary key
                async with self.conn.execute(CHECK_BASE_TABLE_AND_PRIMARY_KEY_SQL) as cursor:
                    row = await cursor.fetchone()
                    table_exists, valid_primary_key, has_is_embedded_column = row
                    if not table_exists:
                        raise RuntimeError(f"Base table '{table}' does not exist.")
                    if not valid_primary_key:
                        raise RuntimeError(f"Base table '{table}' does not have a valid simple primary key.")
                    if not has_is_embedded_column:
                        raise RuntimeError(f"Base table '{table}' does not have an 'is_embedded' column.")
                # Second, we create the embedding table if it doesn't exist
                await self.conn.execute(CREATE_EMBEDDING_TABLE_SQL)
                # Third, we insert the embedding into the embedding table
                try:
                    embedding_blob, tokens = await self.embedding_model.embed(content)
                    await self.conn.execute(INSERT_INTO_EMBEDDING_TABLE_SQL, (pk_hash, embedding_blob))
                    await self.conn.execute(UPDATE_TABLE_STATUS, (pk_hash,))
                    await self.conn.commit()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the add_embedding function inside DotVectorDB: {e}")
    ############################################################
    #   Functions For Ingesting Text Chunks                    # 
    ############################################################
    async def get_unprocessed_metadata(self, table):
        if not self.conn:
            raise RuntimeError("Database not opened")
        try:
            EMBEDDING_PROGRESS_SQL = f"""SELECT * FROM {table} WHERE is_embedded=0"""
            async with self.conn.execute(
                EMBEDDING_PROGRESS_SQL
            ) as cursor:
                rows = await cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
            return [dict(zip(column_names, row)) for row in rows]
        except Exception as e:  
            traceback.print_exc()
            raise RuntimeError(f"Error in the get_unprocessed_metadata function inside DotVectorDB: {e}")
        
    async def ingest_documents(
            self,
            input_document_list=None,
            async_cooldown=60,
            requests_per_minute=1000,
            verbose=False
        ):
            """
            :Docstring for ingest_documents
            :input_document_list is a list of paths to the documents to be ingested into the database.
            :async_cooldown is the cooldown time in seconds between batches of requests to the embedding model to avoid rate limiting.
            :requests_per_minute is the maximum number of requests to the embedding model per minute to avoid rate limiting.
            :verbose is a boolean flag that enables verbose logging for debugging purposes.
            :This function extracts text from the documents, chunks the text, and adds the chunks and their embeddings to the database.

            Ingest a corpus of documents into the database. This function extracts text 
            from the documents, chunks the text, and adds the chunks and their 
            embeddings to the database.
            """ 
            if self.conn is None:
                raise RuntimeError("Database not opened")
            # First, input document can either be None or a list of strings. If neither, we raise a value error.
            # If the user provided valid documents, we extract all text from them.
            try:
                extracted_text = None
                if input_document_list is not None:
                    if not isinstance(input_document_list, list) or not all(isinstance(item, str) for item in input_document_list):
                        raise ValueError("input_document_list must be a list of strings or None.")
                    else:
                        extracted_text = DocumentTextExtractor(directories=input_document_list)
                        results = extracted_text.controller()
                        if results == {}:
                            raise ValueError("No text could be extracted from the provided documents.")
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for extracting text: {e}")       
            # If we have extracted text, we need to chunk them and put them into the database.
            try:
                if extracted_text:
                    total_chunks = []
                    for document, value in results.items():
                        document_chunks = self.chunking_func.chunk(value)
                        if verbose:
                            print("Document:", document)

                        for chunk in document_chunks:
                            content = chunk["content"]

                            await self.add_chunk(
                                content=content,
                                source_name=document,
                                model_name=self.embedding_model.model_name
                            )
                        total_chunks.extend(document_chunks)
                    if verbose:
                        print(f"Total chunks created: {len(total_chunks)}")
                        print(f"Starting to add embeddings for each chunk to the database...")
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for chunking and adding chunks: {e}")
            # By this point, all chunks have been added to the database. Now,  we add the embeddings for each chunk to the database. 
            # We begin by grabbing all chunks where is_embedded flag is 0. This allows us to resume from any chunks that were never embedded yet.
            # We use async to send concurrent requests to the embedding model, but we also implement a cooldown to avoid rate limiting.
            try:
                failed_chunks = await self.get_unprocessed_metadata("chunk")
                while failed_chunks:
                    current_batch = failed_chunks[:requests_per_minute]
                    remaining_chunks = failed_chunks[requests_per_minute:]
                    failed_chunks = []
                    async def embed_chunk(chunk):
                        try:
                            content = chunk["content"]
                            pk_hash = chunk["chunk_id"]
                            await self.add_embedding(
                                content=content,
                                table="chunk",
                                pk_hash=pk_hash
                            )
                            return None
                        except Exception as error:
                            if verbose:
                                print(f"Embedding failed, retrying next batch: {error}")
                            return chunk         
                    if verbose:
                        print(f"Sending batch of {len(current_batch)} chunks to embedding model...")
                    batch_results = await tqdm_asyncio.gather(
                            *(embed_chunk(chunk) for chunk in current_batch)
                        )
                    failed_chunks.extend([
                        chunk for chunk in batch_results
                        if chunk is not None
                    ])
                    failed_chunks.extend(remaining_chunks)
                    if failed_chunks:
                        if verbose:
                            print(
                                f"Cooldown for {async_cooldown} seconds. "
                                f"{len(failed_chunks)} chunks remaining."
                            )
                        await asyncio.sleep(async_cooldown)
                if verbose:
                    print("All embeddings added successfully.")
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for embedding chunks: {e}")
            
    ############################################################
    #   Function For Extracting and Embedding Entities        # 
    ############################################################
    async def extract_and_embed_entities(self, entity_types=None, async_cooldown=60, requests_per_minute=1000, verbose=False):
        if not self.conn:
            raise RuntimeError("Database not opened")
        # First, we add in any passed in entities. If the database is new and no entity types are passed in, we use the default entity types.
        try:
            await self.add_entity_types(entity_types)
            entity_types = await self.get_entity_types()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities function inside DotVectorDB while adding entity types: {e}")
        # Second, we extract entities from the chunks and add them to the database.
        try:
            entity_indexer = EntityIndexer(
                conn=self.conn,
                lock=self.lock,
                language_model=self.language_model.generate,
                embedding_model=self.embedding_model,
                hash_func=self.hashing_func.content_hash,
                entity_types=entity_types,
                add_embedding_func=self.add_embedding
            )
            # Retrieves all unprocessed chunks from the database and sends them to the entity extraction model in batches.
            # The metadata is saved to the database immediately in case of crashes.
            await entity_indexer.extract_entities_from_ingested_chunks(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=verbose
            )
            # Embeds all entities that were extracted by the entity extraction model in batches.
            await entity_indexer.embed_entities(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=verbose
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities function inside DotVectorDB: {e}")
        
    ############################################################
    #   Function For Extracting and Embedding Relations        # 
    ############################################################
    async def extract_and_embed_relationships(self, async_cooldown=60, requests_per_minute=1000, verbose=False):
            if not self.conn:
                raise RuntimeError("Database not opened")
            try:
                relationship_indexer = RelationshipIndexer(
                    conn=self.conn,
                    lock=self.lock,
                    language_model=self.language_model.generate,
                    hash_func=self.hashing_func.content_hash,
                    embedding_model=self.embedding_model,
                    add_embedding_func=self.add_embedding
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the extract_and_embed_relationships function inside DotVectorDB while initializing RelationshipIndexer: {e}")
            try:
                await relationship_indexer.extract_relations_from_chunks(
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                    verbose=verbose
                )
                await relationship_indexer.embed_relationships(
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                    verbose=verbose)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the extract_and_embed_relationships function inside DotVectorDB while extracting and embedding relationships: {e}")

            
    ############################################################
    # Function For a Full Construction (Chunks -> Full Graph)  # 
    ############################################################
    async def construct_GraphRAG(self, input_document_list=None, entity_types=None, async_cooldown=60, requests_per_minute=1000, verbose=False):
        # First, we ingest the documents into the database, which includes extracting text, chunking it, and adding embeddings.
        try:
            await self.ingest_documents(input_document_list=input_document_list, 
                                        async_cooldown=async_cooldown, requests_per_minute=requests_per_minute, 
                                        verbose=verbose)
            # Second, we  add in passed Entity types. If none, we use default types.
            await self.add_entity_types(entity_types)
            entity_types = await self.get_entity_types()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while ingesting text chunks: {e}")
        # Third, we extract entities from the chunks and add them to the database.
        try:
           await self.extract_and_embed_entities(entity_types=entity_types, async_cooldown=async_cooldown, requests_per_minute=requests_per_minute, verbose=verbose)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while adding in the entities: {e}")
        # Fourth, we extract relationships from the entities and add them to the database.
        try:
            await self.extract_and_embed_relationships(async_cooldown=async_cooldown, requests_per_minute=requests_per_minute, verbose=verbose)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while adding in the relationships: {e}")
        if verbose:
            print("Graph construction completed successfully.")
        
    ############################################################
    #   Functions For Graph Retrieval                          # 
    ############################################################
    async def vector_search(self, query, table, top_k=40, KNN=True, distance_metric="l2"):
            if not self.conn:
                    raise RuntimeError("Database not opened") 
            query_embedding, query_tokens = await self.embedding_model.embed(query)
            if distance_metric == "cos":
                distance_func = "vec1_cos_distance"
            else:
                distance_func = "vec1_l2_distance"
            if KNN:
                SQL = f"""
                SELECT
                    v.rowid AS vector_rowid,
                    m.*,
                    1.0 / (1.0 + {distance_func}(:query_embedding, v.embedding)) AS similarity_score
                FROM {table}_embeddings(:query_embedding, '{{k: {top_k}}}') AS v
                JOIN {table} AS m
                    ON m.rowid = v.rowid
                ORDER BY similarity_score DESC
                LIMIT :top_k;
                """
            else:
                rerank_k = top_k * 10

                SQL = f"""
                WITH candidates AS (
                    SELECT
                        v.rowid,
                        v.embedding
                    FROM {table}_embeddings(:query_embedding, '{{k: {rerank_k}, nprobe: 0.08}}') AS v
                )
                SELECT
                    c.rowid AS vector_rowid,
                    m.*,
                    1.0 / (1.0 + {distance_func}(:query_embedding, c.embedding)) AS similarity_score
                FROM candidates AS c
                JOIN {table} AS m
                    ON m.rowid = c.rowid
                ORDER BY similarity_score DESC
                LIMIT :top_k;
                """
            try: 
                async with self.conn.execute(
                        SQL,
                        {
                            "query_embedding": query_embedding,
                            "top_k": top_k,
                        },
                    ) as cursor:
                        rows = await cursor.fetchall()
                        column_names = [description[0] for description in cursor.description]
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the vector_search function inside DotVectorDB: {e}")
            return [dict(zip(column_names, row)) for row in rows], query_tokens
    
    async def construct_temp_graph(self, relationship_ids):
        if not self.conn:
            raise RuntimeError("Database not opened")
        if not self.graph_retriever:
            raise RuntimeError("Graph retriever is not initialized. Please load the graph first.")
        if not self.graph_retriever.graph:
            raise RuntimeError("Graph is not loaded in the graph retriever. Please load the graph first.")
        try:
            temp_graph = await self.graph_retriever.create_temporary_relational_graph(relationship_ids=relationship_ids, verbose=True)
            return temp_graph
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_temp_graph function inside DotVectorDB: {e}")
    
    async def subgraph_constrained_semantic_retrieval(self, custom_func, temp_vector_space_names, memoryHandler=None, **kwargs):
            """
            In this code, you would open TempVectorSpace() and close tempvector space by the end of the function.
            It essentially creates a temporary vector space that is opened annd closed by the end of this function.
            between the two functions will be the custom function you want to run inside the temmp vector space. it should return whatever results you wan
            """
            if not self.conn:
                raise RuntimeError("Database not opened")
            results = None
            async with AsyncTempVectorSpace(conn=self.conn, tids=temp_vector_space_names, lock=self.lock, memory=memoryHandler) as QueryWS:
                results, tokens = await custom_func(QueryWS, **kwargs)
            return results, tokens
        
                



        