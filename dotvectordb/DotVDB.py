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
from dotvectordb.graph.entity_indexer2 import EntityIndexer2
from dotvectordb.graph.entity_indexer3 import EntityIndexer3
from dotvectordb.graph.entity_indexer4 import EntityIndexer4
from dotvectordb.graph.dedupe_indexer import DedupeIndexer
from dotvectordb.graph.relatonship_indexer import RelationshipIndexer
from dotvectordb.graph.relationship_indexer2 import RelationshipIndexer2
from dotvectordb.graph.relationship_indexer3 import (
    RelationshipIndexer as RelationshipIndexer3,
)
from dotvectordb.graph.type_indexer import TypeIndexer
from dotvectordb.graph.graph_retriever import GraphRetriever
from dotvectordb.graph.generate_global_desc import GlobalDescriptionGenerator
from dotvectordb.graph.TempVectorSpace import AsyncTempVectorSpace
from dotvectordb.sql_browser import open_sql_browser
from dotvectordb.prompts.chatbot_prompts import ChatbotRetrievalDecision
from dotvectordb.data_structures.path import PathManager
import os
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import igraph
import json
import aiosqlite
import asyncio
import traceback
import tiktoken
import warnings
import re
BASE_DIR = Path(__file__).resolve().parent
class DotVectorDB:
    def __init__(self, path: str, 
                 embedding_model: EmbeddingModel = None,
                 language_model: LanguageModel = None,
                 chunking_func: ChunkingFunc = FixedSizeChunker(hash_func=SHA256HashingFunc()),
                 hashing_func: HashingFunc = SHA256HashingFunc(),
                 tokenizer_name: str = None,
                 verbose=False):
        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel(api_key=os.getenv("OPENAI_API_KEY"))
        if language_model is None:
            language_model = OpenAiAgent(api_key=os.getenv("OPENAI_API_KEY"))
        if tokenizer_name is None:
            tokenizer_name = os.getenv("TOKENIZER")
        self.path = path
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.chunking_func = chunking_func
        self.hashing_func = hashing_func
        self.verbose = verbose
        self.conn: aiosqlite.Connection | None = None
        self.lock = asyncio.Lock()
        self._sql_browser_server = None
    ############################################################
    #   Helper Functions For DotVDB                            # 
    ############################################################
    def __require_open_connection(self)-> None:
        """Helper function to ensure that the database connection is open before performing any operations."""
        if not self.conn:
            raise RuntimeError("Database connection is not open. Please call the 'open' method first.")
    def __require_graph_retriever(self)-> None:
        """Helper function to ensure that the graph retriever is initialized before performing any operations that require it."""
        if not self.graph_retriever:
            raise RuntimeError("Graph retriever is not initialized. Please load the graph first.")
    def __require_graph(self)-> None:
        """Helper function to ensure that the graph is loaded in the graph retriever before performing any operations that require it."""
        if not self.graph_retriever.graph:
            raise RuntimeError("Graph is not loaded in the graph retriever. Please load the graph first.")   
    def __get_distance_func(self, distance_metric: str) -> str:
        """Helper function to determine the appropriate distance function based on the specified distance metric."""
        if distance_metric == "cos":
            distance_func = "vec1_cos_distance"
        else:
            distance_func = "vec1_l2_distance"
        return distance_func
    def __transform_SQL_row_to_dict(self, row, cursor) -> list[dict]:
        """Helper function to transform SQL query results into a list of dictionaries."""
        if row is None:
            return None
        column_names = [description[0] for description in cursor.description]
        return dict(zip(column_names, row))
    def __read_schema_file(self, filename: str) -> str:
        """Helper function to read a schema file and return its contents as a string."""
        file = (BASE_DIR / "schema" / filename).read_text(encoding="utf-8")
        return file
    def __graph_construction_print(self, message):
        """Helper function to print messages related to graph construction."""
        print("==================================================================")
        print(message)
        print("==================================================================")
    ############################################################
    #   Functions For Opening and Closing DotVDB               # 
    ############################################################
    async def open(self):
        if (
            self._sql_browser_server is not None
            and self._sql_browser_server.write_mode
            and self._sql_browser_server.conn is not None
        ):
            raise RuntimeError(
                "Close the write-mode SQL browser before reopening "
                "DotVectorDB."
            )
        self.conn = await aiosqlite.connect(self.path)
        def setup():
            try:
                # Get the raw connection to set up the database schema
                raw = self.conn._conn
                # Set PRAGMA settings for performance
                pragma_sql = self.__read_schema_file("settings.sql")
                raw.executescript(pragma_sql)
                # Loads the compiled vec1 extension for vector operations
                raw.enable_load_extension(True)
                raw.load_extension(
                str(BASE_DIR / "compiled_vec1" / "vec1.so")
                )
                raw.enable_load_extension(False)
                if self.verbose:
                    print("Database connection opened and vec1 extension loaded successfully.")
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the setup() function inside DotVectorDB's open function: {e}")
        await self.conn._execute(setup)
        await self.initialize_default_schema()
        replace_db_extension_with_underscore = self.path.replace(".db", "_")
        self.graph_path = f"""{replace_db_extension_with_underscore}_graph.pkl"""
        self.graph_retriever = GraphRetriever(graph_path=f"""{self.graph_path}""", 
                                              conn=self.conn, lock=self.lock, 
                                              get_num_nodes_edges_func=self.get_num_of_entities_and_relationships,
                                              vector_search=self.vector_search)
        self.DIM = await self.get_embedding_dim()
        if self.verbose:
            print("DotVectorDB initialized successfully.")
        return self
    async def close(self):
        if self.conn:
            if self.verbose:
                print("Closing database connection...")
            await self.conn.close()
            self.conn = None
        else:
            if self.verbose:
                print("Database connection is already closed.")
    ############################################################
    #   Functions For Manipulating DotVDB's Schema             #
    ############################################################
    async def initialize_default_schema(self) -> None:
        try:
            self.__require_open_connection()
            chunks_sql = self.__read_schema_file("chunk.sql")
            entities_sql = self.__read_schema_file("entity.sql")
            relationships_sql = self.__read_schema_file("relationship.sql")
            failure_log_sql = self.__read_schema_file("failure_log.sql")
            chat_history_sql = self.__read_schema_file("chat_history.sql")
            async with self.lock:
                await self.conn.executescript(chunks_sql)
                await self.conn.executescript(entities_sql)
                await self.conn.executescript(relationships_sql)
                await self.conn.executescript(failure_log_sql)
                await self.conn.executescript(chat_history_sql)
                await self.conn.commit()
            if self.verbose:
                print("Default schema initialized successfully.")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the initialize_default_schema function inside DotVectorDB: {e}")
    async def add_entity_tag(self,  tag):
        self.__require_open_connection()
        try:
            await self.conn.execute(
                """
                INSERT OR IGNORE INTO entity_tags(entity_tag)
                VALUES (?)
                """,
                (tag,),
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the add_entity_tags function inside DotVectorDB: {e}")

    async def add_entity_types(self, entity_types=None)-> None:
        self.__require_open_connection()
        try:
            existing_entity_types = await self.get_entity_types()
            if entity_types is None or await isEntityTypesFormatCorrect(entity_types) is False:
                if self.verbose:
                    print("No valid entity types provided. Using default entity types.")
                entity_types = DEFAULT_ENTITY_TYPES
            if self.verbose:
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
                if self.verbose:
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
    async def add_tables(self,schema_list)-> None:
        self.__require_open_connection()
        async with self.lock:
            for schema in schema_list:
                await self.conn.executescript(schema)
        await self.conn.commit()
    async def cleanup_temp_tables(self):
        self.__require_open_connection()
        async with self.conn.execute("""
            SELECT name
            FROM sqlite_temp_master
            WHERE type = 'table';
        """) as cursor:
            tables = await cursor.fetchall()
            for (table_name,) in tables:
                await self.conn.execute(
                    f'DROP TABLE IF EXISTS temp."{table_name}"'
                )
            await self.conn.commit()
    async def remove_tables(self, table_names)-> None:
        self.__require_open_connection()
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
                        if self.verbose:
                            print(f"Table {table_name} dropped.")
                    for table_name in table_names:
                        await drop_table(table_name)
                    await self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the remove_tables function inside DotVectorDB: {e}")
    async def display_tables(self)-> None:
        self.__require_open_connection()
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
    #   Accessor Functions                                     # 
    ############################################################
    async def get_entity_metadata(self, entity_id)-> dict:
        self.__require_open_connection()
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM entity WHERE entity_id = ?",
                (entity_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            return self.__transform_SQL_row_to_dict(row, cursor)
    async def get_relationship_metadata(self, relationship_id)-> dict:
        self.__require_open_connection()
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM relationship WHERE relation_id = ?",
                (relationship_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            return self.__transform_SQL_row_to_dict(row, cursor)
    async def get_chunk_metadata(self, chunk_id)-> dict:
        self.__require_open_connection()
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM chunk WHERE chunk_id = ?",
                (chunk_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            return self.__transform_SQL_row_to_dict(row, cursor)
    async def get_num_of_entities_and_relationships(self) -> tuple[int, int]:
        """Returns the number of entities and relationships in the database"""
        self.__require_open_connection()
        async with self.lock:
            cursor = await self.conn.execute("SELECT num_entities, num_relations FROM global_config WHERE id = 1")
            result = await cursor.fetchone()
            await cursor.close()
            num_entities = result[0] if result[0] is not None else 0
            num_relationships = result[1] if result[1] is not None else 0
        return num_entities, num_relationships
    async def get_entity_types(self, only_populated=False, include_counts=False)-> dict[str, str]:
            """Gets entity types from the database and returns them as a dictionary"""
            self.__require_open_connection()
            try: 
                async with self.lock:
                    if only_populated:
                        if include_counts:
                            cursor = await self.conn.execute(
                                """
                                SELECT entity_type, type_description, num_entities
                                FROM entity_types
                                WHERE num_entities > 0
                                """
                            )
                        else:
                            cursor = await self.conn.execute(
                                """
                                SELECT entity_type, type_description
                                FROM entity_types
                                WHERE num_entities > 0
                                """
                            )
                    else:
                        if include_counts:
                            cursor = await self.conn.execute(
                                """
                                SELECT entity_type, type_description, num_entities
                                FROM entity_types
                                """
                            )
                        else:
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
    async def get_specific_entity_types(self, entity_types):
        self.__require_open_connection()
        try:
            async with self.lock:
                placeholders = ', '.join('?' for _ in entity_types)
                query = f"""
                    SELECT entity_type, type_description
                    FROM entity_types
                    WHERE entity_type IN ({placeholders})
                """
                cursor = await self.conn.execute(query, tuple(entity_types))
                rows = await cursor.fetchall()
                await cursor.close()
                return {row[0]: row[1] for row in rows}
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the get_specific_entity_types function inside DotVectorDB: {e}")
    async def get_sources(self) -> list[dict]:
            """Gets sources from the database and returns them as a list of dictionaries"""
            self.__require_open_connection()
            try:
                async with self.lock:
                    cursor = await self.conn.execute(
                        "SELECT id, name FROM source"
                    )
                    sources = await cursor.fetchall()
                    await cursor.close()
                    return [
                        {"id": source[0], "name": source[1]}
                        for source in sources
                    ]
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the get_sources function inside DotVectorDB: {e}")
    async def get_embedding_dim(self) -> int:
        self.__require_open_connection()
        try:
            async with self.lock:
                cursor = await self.conn.execute(
                    "SELECT DIM FROM global_config WHERE id = 1"
                )
                result = await cursor.fetchone()
                await cursor.close()
                if result is None:
                    raise RuntimeError("Global configuration not found in the database. One should always exist!")
                dim = result[0]
            # We move outside the lock to avoid a potential deadlock. We call set_embedding_dim to set the missing dim.
            if dim is None:
                dim = await self.set_embedding_dim()
            return dim
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the get_embedding_dim function inside DotVectorDB: {e}")
    async def get_global_config(self) -> dict[str, int | str]:
            """Gets the global configuration metadata from the database"""
            self.__require_open_connection()
            try:
                async with self.lock:
                    cursor = await self.conn.execute(
                        """
                        SELECT current_version_count, description_version, description, DIM, embedding_model_id, num_chunks, num_entities, num_relations
                        FROM global_config
                        WHERE id = 1
                        """
                    )
                    global_config = await cursor.fetchone()
                    await cursor.close()
                    if global_config is None:
                        raise RuntimeError("Global configuration not found in the database. One should always exist!")
                    return {
                        "current_version_count": global_config[0],
                        "description_version": global_config[1],
                        "description": global_config[2],
                        "DIM": global_config[3],
                        "embedding_model_id": global_config[4],
                        "num_chunks": global_config[5],
                        "num_entities": global_config[6],
                        "num_relations": global_config[7]
                    }
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the get_global_config function inside DotVectorDB: {e}")
    async def get_chunks_associated_with_entity_id(self, entity_id):
        """Given an entity_id, this function retrieves all associated chunks from the database."""
        self.__require_open_connection()
        try:
            async with self.lock:
                cursor = await self.conn.execute(
                    """
                    SELECT chunk.*
                    FROM chunk
                    JOIN chunk_to_entity_mapping
                        ON chunk.chunk_id = chunk_to_entity_mapping.chunk_id
                    WHERE chunk_to_entity_mapping.entity_id = ?
                    """,
                    (entity_id,)
                )
                rows = await cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                await cursor.close()
                return [dict(zip(column_names, row)) for row in rows]
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the get_chunks_associated_with_entity_id function inside DotVectorDB: {e}")
    async def get_chunks_associated_with_two_entities(self, entity_id_one, entity_id_two):
        """Given two entity IDs, this function retrieves all associated chunks they have in common."""
        self.__require_open_connection()
        try:
            async with self.lock:
                cursor = await self.conn.execute(
                    """
                    SELECT chunk.*
                    FROM chunk
                    JOIN chunk_to_entity_mapping AS mapping_one
                        ON chunk.chunk_id = mapping_one.chunk_id
                    JOIN chunk_to_entity_mapping AS mapping_two
                        ON chunk.chunk_id = mapping_two.chunk_id
                    WHERE mapping_one.entity_id = ? AND mapping_two.entity_id = ?
                    """,
                    (entity_id_one, entity_id_two)
                )
                rows = await cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                await cursor.close()
                return [dict(zip(column_names, row)) for row in rows]
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the get_chunks_associated_with_entity_id function inside DotVectorDB: {e}")
    ############################################################
    #   Functions For Adding Text Content to the Database      # 
    ############################################################
    async def set_embedding_dim(self) -> int:
        self.__require_open_connection()
        text = "This is a test string to determine the embedding dimension."
        # First, we must add the embedding model to the embedding table if it does not already exist.
        embedding_model_id = await self.add_metadata_content(table="embedding_model", name=self.embedding_model.model_name)
        async with self.lock:
            try:
                # Then, we can proceed with the rest of the function.
                # Since it is a numpy array, we get the dimension by checking the last dimension of the shape.
                # e.g., (1, 1536) yields 1536 as the embedding dimension.
                embedding, tokens = await self.embedding_model.embed(text)
                dim = embedding.shape[-1]
                await self.conn.execute(
                    """
                    UPDATE global_config
                    SET DIM = ?, embedding_model_id = ?
                    WHERE id = 1
                    """,
                    (dim, embedding_model_id)
                )
                await self.conn.commit()
                return dim
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the set_embedding_dim function inside DotVectorDB: {e}")
    async def update_global_description(self, description, description_version) -> None:
                """Updates the global description and its database version"""
                self.__require_open_connection()
                try:
                    async with self.lock:
                        await self.conn.execute(
                            """
                            UPDATE global_config
                            SET description = ?, description_version = ?
                            WHERE id = 1
                            """,
                            (
                                json.dumps(description, ensure_ascii=False),
                                description_version,
                            )
                        )
                        await self.conn.commit()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the update_global_description function inside DotVectorDB: {e}")
    async def add_metadata_content(self, table, name) -> str:
        self.__require_open_connection()
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
            return source_hash
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the add_metadata_content function inside DotVectorDB: {e}")
    async def add_chunk(self, content, source_name, model_name) -> None:
        self.__require_open_connection()
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
            raise RuntimeError(f"Error while adding a chunk: {e}")
    ############################################################
    #   Functions For Ingesting Text Chunks                    # 
    ############################################################
    async def get_unprocessed_metadata(self, table) -> list[dict]:
        self.__require_open_connection()
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
            requests_per_minute=1000
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
            self.__require_open_connection()
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
                        print(f" Processing document: {document}")
                        document_chunks = self.chunking_func.chunk(value)
                        print(f"  Number of chunks created for {document}: {len(document_chunks)}")
                        for chunk in tqdm_asyncio(document_chunks, desc=f"Adding chunks from {document} into the DB"):
                            content = chunk["content"]
                            await self.add_chunk(
                                content=content,
                                source_name=document,
                                model_name=self.embedding_model.model_name
                            )
                        total_chunks.extend(document_chunks)
                    if self.verbose:
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
                            if self.verbose:
                                print(f"Embedding failed, retrying next batch: {error}")
                            return chunk         
                    if self.verbose:
                        print(f"Sending batch of {len(current_batch)} chunks to embedding model...")
                    batch_results = await tqdm_asyncio.gather(
                            *(embed_chunk(chunk) for chunk in current_batch),
                            desc="Embedding chunks and adding to the vector database",
                        )
                    failed_chunks.extend([
                        chunk for chunk in batch_results
                        if chunk is not None
                    ])
                    failed_chunks.extend(remaining_chunks)
                    if failed_chunks:
                        if self.verbose:
                            print(
                                f"Cooldown for {async_cooldown} seconds. "
                                f"{len(failed_chunks)} chunks remaining."
                            )
                        await asyncio.sleep(async_cooldown)
                if self.verbose:
                    print("All embeddings added successfully.")
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the ingest_documents function inside DotVectorDB for embedding chunks: {e}")
    ############################################################
    #   Function For Extracting and Embedding Entities        # 
    ############################################################
    async def extract_entity_tags(self, async_cooldown=60, requests_per_minute=1000) -> None:
        try:
            REL_indexer =RelationshipIndexer2(
                conn=self.conn,
                lock=self.lock,
                language_model=self.language_model.generate,
                hash_func=self.hashing_func.content_hash,
            )
            await REL_indexer.extract_relations_from_chunks(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=self.verbose,
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_entity_tags function inside DotVectorDB while initializing TypeIndexer: {e}")


    async def extract_and_embed_entities(self, entity_types=None, async_cooldown=60, requests_per_minute=1000) -> None:
        self.__require_open_connection()
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
                verbose=self.verbose
            )
            # Embeds all entities that were extracted by the entity extraction model in batches.
            await entity_indexer.embed_entities(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=self.verbose
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities function inside DotVectorDB: {e}")

    async def extract_and_embed_entities2(self, entity_types=None, async_cooldown=60, requests_per_minute=1000) -> None:
        self.__require_open_connection()
        # First, we add in any passed in routing types. If the database is new and no routing types are passed in, we use the default routing types.
        try:
            await self.add_entity_types(entity_types)
            entity_types = await self.get_entity_types()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities2 function inside DotVectorDB while adding routing types: {e}")
        # Second, we describe and classify the temporary entities from the extracted relationships.
        try:
            entity_indexer = EntityIndexer2(
                conn=self.conn,
                lock=self.lock,
                language_model=self.language_model.generate,
                embedding_model=self.embedding_model,
                hash_func=self.hashing_func.content_hash,
                entity_types=entity_types,
                add_embedding_func=self.add_embedding
            )
            await entity_indexer.extract_entities_from_ingested_chunks(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=self.verbose
            )
            await entity_indexer.embed_entities(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=self.verbose
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities2 function inside DotVectorDB: {e}")

    async def extract_and_embed_entities4(self, entity_types=None, async_cooldown=60, requests_per_minute=1000) -> None:
            self.__require_open_connection()
            try:
                await self.add_entity_types(entity_types)
                entity_types = await self.get_entity_types()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the extract_and_embed_entities3 function inside DotVectorDB while adding entity types: {e}")
    
            try:
                entity_indexer = EntityIndexer4(
                    conn=self.conn,
                    lock=self.lock,
                    language_model=self.language_model.generate,
                    embedding_model=self.embedding_model,
                    hash_func=self.hashing_func.content_hash,
                    entity_groups=entity_types,
                    add_embedding_func=self.add_embedding
                )
                await entity_indexer.extract_entities_from_ingested_chunks(
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                    verbose=self.verbose
                )

                await entity_indexer.fix_entities(
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                    verbose=self.verbose         
               )                                                              

                await entity_indexer.embed_entities(
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                    verbose=self.verbose
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the extract_and_embed_entities4 function inside DotVectorDB: {e}")
    ############################################################
    #   Function For Extracting and Embedding Relations        # 
    ############################################################
    async def extract_and_embed_relationships(self, async_cooldown=60, requests_per_minute=1000) -> None:
            self.__require_open_connection()
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
                    verbose=self.verbose
                )
                await relationship_indexer.embed_relationships(
                    async_cooldown=async_cooldown,
                    requests_per_minute=requests_per_minute,
                    verbose=self.verbose)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the extract_and_embed_relationships function inside DotVectorDB while extracting and embedding relationships: {e}")
    async def extract_and_embed_relationships2(self, async_cooldown=60, requests_per_minute=1000) -> None:
                self.__require_open_connection()
                try:
                    relationship_indexer = RelationshipIndexer3(
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
                        verbose=self.verbose
                    )
                    await relationship_indexer.fix_relationships(
                        async_cooldown=async_cooldown,
                        requests_per_minute=requests_per_minute,
                        verbose=self.verbose
                    )
                    await relationship_indexer.embed_relationships(
                        async_cooldown=async_cooldown,
                        requests_per_minute=requests_per_minute,
                        verbose=self.verbose)
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the extract_and_embed_relationships function inside DotVectorDB while extracting and embedding relationships: {e}")
    ############################################################
    #   Function For Generating Global Source Descriptions    #
    ############################################################
    async def generate_global_descriptions(self, sample_size=5, debug_mode=False) -> list[dict]:
        self.__require_open_connection()
        try:
            entity_types = await self.get_entity_types(only_populated=True)
            sources = await self.get_sources()
            global_config = await self.get_global_config()
            generator = GlobalDescriptionGenerator(
                conn=self.conn,
                lock=self.lock,
                language_model=self.language_model.generate,
                entity_types=entity_types,
                sources=sources,
                global_config=global_config,
                save_global_description_func=self.update_global_description,
                verbose=self.verbose,
                debug_mode=debug_mode
            )
            return await generator.generate(
                sample_size=sample_size,
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                "Error in the generate_global_descriptions function inside DotVectorDB: "
                f"{e}"
            ) 
    #####################################################################
    # Function For a Full Construction (Corpus-> Chunks -> Full Graph)  # 
    #####################################################################
    async def count_llm_tokens(self, stage):
                self.__require_open_connection()
                valid_stages = {"entity", "relation", "global_desc", "retrieval"}
                if stage not in valid_stages:
                    raise ValueError(
                        f"Invalid stage '{stage}'. "
                        f"Must be one of: {', '.join(sorted(valid_stages))}"
                    )
                cursor = await self.conn.execute(
                    """
                    SELECT COALESCE(SUM(total_tokens), 0)
                    FROM llm_token_usage
                    WHERE stage = ?
                    """,
                    (stage,)
                )
                row = await cursor.fetchone()
                return row[0]
    async def construct_GraphRAG(self, input_document_list=None, entity_types=None, debug_mode=False, async_cooldown=60, requests_per_minute=1000) -> None:
        # First, we ingest the documents into the database, which includes extracting text, chunking it, and adding embeddings.
        try:
            self.__graph_construction_print("STEP ONE: Chunking & Embedding a Given Corpus")
            await self.ingest_documents(input_document_list=input_document_list, 
                                        async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while ingesting text chunks: {e}")
        # Second, we extract entities from the chunks and add them to the database.
        try:
           self.__graph_construction_print("STEP TWO: Extracting Entities from the Ingested/Unprocessed Chunks")
           await self.extract_and_embed_entities(entity_types=entity_types, async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while adding in the entities: {e}")
        # Third, we extract relationships from the entities and add them to the database.
        try:
            self.__graph_construction_print("STEP THREE: Extracting Relationships from the Ingested/Unprocessed Chunks")
            await self.extract_and_embed_relationships(async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while adding in the relationships: {e}")
        # Fourth, generate and save a global description for each source document.
        try:
            self.__graph_construction_print("STEP FOUR: Generating a Global Description for the entity-relationship graph")
            await self.generate_global_descriptions(debug_mode=debug_mode)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while generating global descriptions: {e}")
        # Lastly, we need to generate the graph representation of the database.
        try:
            self.__graph_construction_print("STEP FIVE: Creating the Graph Representation of the Database")
            await self.graph_retriever.load_graph()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while generating the graph representation: {e}") 
        print("Graph construction completed successfully.")
    async def construct_GraphRAG2(self, input_document_list=None, entity_types=None, debug_mode=False, async_cooldown=60, requests_per_minute=1000) -> None:
            # First, we ingest the documents into the database, which includes extracting text, chunking it, and adding embeddings.
            try:
                self.__graph_construction_print("STEP ONE: Chunking & Embedding a Given Corpus")
                await self.ingest_documents(input_document_list=input_document_list,
                                            async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while ingesting text chunks: {e}")
            # Second, we extract, classify, and embed entities from the chunks.
            try:
               self.__graph_construction_print("STEP TWO: Extracting, Classifying, and Embedding Entities")
               await self.extract_and_embed_entities3(entity_types=entity_types, async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while adding in the entities: {e}")

             # Third, we extract relationships from the entities and add them to the database.
            try:
                self.__graph_construction_print("STEP THREE: Extracting Relationships from the Ingested/Unprocessed Chunks")
                await self.extract_and_embed_relationships(async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while adding in the relationships: {e}")


            # Fourth, generate and save a global description for each source document.
            try:
                self.__graph_construction_print("STEP FOUR: Generating a Global Description for the entity-relationship graph")
                await self.generate_global_descriptions(debug_mode=debug_mode)
            except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while generating global descriptions: {e}")
            # Fifth, we need to generate the graph representation of the database.
            try:
                self.__graph_construction_print("STEP FIVE: Creating the Graph Representation of the Database")
                await self.graph_retriever.load_graph()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while clustering entities: {e}")
            # Lastly, we construct the BM25 index
            try:
               self.__graph_construction_print("STEP SIX: Creating the BM25 Index")
               await self.create_BM25_index_for_table("chunk")
               await self.create_BM25_index_for_table("entity")
               await self.create_BM25_index_for_table("relationship")  
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while creating the BM25 index: {e}")
    async def construct_GraphRAG3(self, input_document_list=None, entity_types=None, debug_mode=False, async_cooldown=60, requests_per_minute=1000) -> None:
                # First, we ingest the documents into the database, which includes extracting text, chunking it, and adding embeddings.
                try:
                    self.__graph_construction_print("STEP ONE: Chunking & Embedding a Given Corpus")
                    await self.ingest_documents(input_document_list=input_document_list,
                                                async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while ingesting text chunks: {e}")
                # Second, we extract, classify, and embed entities from the chunks.
                try:
                   self.__graph_construction_print("STEP TWO: Extracting, Classifying, and Embedding Entities")
                   await self.extract_and_embed_entities4(entity_types=entity_types, async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while adding in the entities: {e}")
    
                 # Third, we extract relationships from the entities and add them to the database.
                try:
                    self.__graph_construction_print("STEP THREE: Extracting Relationships from the Ingested/Unprocessed Chunks")
                    await self.extract_and_embed_relationships2(async_cooldown=async_cooldown, requests_per_minute=requests_per_minute)
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while adding in the relationships: {e}")
    
    
                # Fourth, generate and save a global description for each source document.
                try:
                    self.__graph_construction_print("STEP FOUR: Generating a Global Description for the entity-relationship graph")
                    await self.generate_global_descriptions(debug_mode=debug_mode)
                except Exception as e:
                            traceback.print_exc()
                            raise RuntimeError(f"Error in the construct_GraphRAG function inside DotVectorDB while generating global descriptions: {e}")
                # Fifth, we need to generate the graph representation of the database.
                try:
                    self.__graph_construction_print("STEP FIVE: Creating the Graph Representation of the Database")
                    await self.graph_retriever.load_graph()
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while clustering entities: {e}")
                # Lastly, we construct the BM25 index
                try:
                   self.__graph_construction_print("STEP SIX: Creating the BM25 Index")
                   await self.create_BM25_index_for_table("chunk")
                   await self.create_BM25_index_for_table("entity")
                   await self.create_BM25_index_for_table("relationship")  
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while creating the BM25 index: {e}")
    
    ############################################################
    #   Functions For DotVDB's Vector Ingestion + Search       # 
    ############################################################
    async def add_embedding(self, table, content, pk_hash) -> None:
                # Check if the database connection is open
                self.__require_open_connection()
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
                try:
                     embedding_blob, tokens = await self.embedding_model.embed(content)
                     if embedding_blob.shape[-1] != self.DIM:
                        raise RuntimeError(f"Embedding dimension mismatch: expected {self.DIM}, got {embedding_blob.shape[-1]}")
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(f"Error in the add_embedding function inside DotVectorDB: {e}")
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
                        await self.conn.execute(INSERT_INTO_EMBEDDING_TABLE_SQL, (pk_hash, embedding_blob))
                        await self.conn.execute(UPDATE_TABLE_STATUS, (pk_hash,))
                        await self.conn.commit()
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the add_embedding function inside DotVectorDB: {e}")
    async def compare_two_entities(self, centroid_entity_id, comparsion_entity_id, distance_metric="l2") -> tuple[tuple[str, str], float]:
        self.__require_open_connection()
        distance_func = self.__get_distance_func(distance_metric)
        # Compare the similarity between two entities relative to the centroid's embedding
        SQL = f"""
            SELECT
                1.0 / (1.0 + {distance_func}(v1.embedding, v2.embedding))
                AS similarity_score,
                e.entity_name
            FROM entity_embeddings AS v1
            JOIN entity_embeddings AS v2
            JOIN entity AS e
            ON e.entity_id = v2.rowid
            WHERE v1.rowid = :centroid_entity_id
            AND v2.rowid = :comparison_entity_id;
            """
        try:
            async with self.conn.execute(
                SQL,
                {
                    "centroid_entity_id": centroid_entity_id,
                    "comparison_entity_id": comparsion_entity_id,
                },
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("No similarity score found for the given entity IDs.")
                similarity_score = row[0]
                comparsion_entity_name = row[1]
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the compare_two_entities function inside DotVectorDB: {e}")
        return (comparsion_entity_id, comparsion_entity_name), similarity_score
    async def vector_search(self, query, table, top_k=40, embedding_table=None, KNN=True, distance_metric="l2") -> tuple[list[dict], int]:
            self.__require_open_connection()
            query_embedding, query_tokens = await self.embedding_model.embed(query)
            distance_func = self.__get_distance_func(distance_metric)
            if embedding_table is None:
                embedding_table = f"{table}_embeddings"
            if KNN:
                SQL = f"""
                SELECT
                    v.rowid AS vector_rowid,
                    m.*,
                    1.0 / (1.0 + {distance_func}(:query_embedding, v.embedding)) AS similarity_score
                FROM {embedding_table}(:query_embedding, '{{k: {top_k}}}') AS v
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
                    FROM {embedding_table}(:query_embedding, '{{k: {rerank_k}, nprobe: 0.08}}') AS v
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
    async def construct_temp_graph(self, relationship_ids) -> igraph.Graph:
        self.__require_open_connection()
        self.__require_graph_retriever()
        self.__require_graph()
        try:
            temp_graph = await self.graph_retriever.create_temporary_relational_graph(relationship_ids=relationship_ids, verbose=True)
            return temp_graph
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the construct_temp_graph function inside DotVectorDB: {e}")
    async def subgraph_constrained_semantic_retrieval(self, query, custom_search_algo, DOTs, MAX_TOKENS, top_k=40, **kwargs):
            """
            In this code, you would open TempVectorSpace() and close tempvector space by the end of the function.
            It essentially creates a temporary vector space that is opened annd closed by the end of this function.
            between the two functions will be the custom function you want to run inside the temmp vector space. it should return whatever results you wan
            """
            self.__require_open_connection()
            gathered_results = None
            Manager = PathManager(tokenizer=self.tokenizer, max_tokens=MAX_TOKENS, top_k=top_k, query=query)
            async with AsyncTempVectorSpace(query=query, conn=self.conn,DOTs=DOTs, conn_check_func=self.__require_open_connection, lock=self.lock) as QueryWS:
                # The custom_search_algo should be some DOT-based retrieval algorithm
                # The search is performed concurrently for each DOT
                gathered_results = await asyncio.gather(
                *(
                    custom_search_algo(QueryWS, DOT, Manager, **kwargs)
                    for DOT in DOTs
                )
            )
            return gathered_results, Manager
    ############################################################
    #   Functions For DotVDB's BM25 Indexing + Search          # 
    ############################################################
    async def create_BM25_index_for_table(self, table_name: str):
            """Creates a BM25 index for the specified table ('chunk', 'entity', or 'relationship') using SQLite's FTS5 extension."""
            try:
                self.__require_open_connection()
                table_config = {
                    "chunk": {
                        "id_column": "chunk_id",
                        "columns": ["content"],
                    },
                    "entity": {
                        "id_column": "entity_id",
                        "columns": [
                            "entity_name",
                            "entity_description",
                            "questions",
                        ],
                    },
                    "relationship": {
                        "id_column": "relation_id",
                        "columns": [
                            "relation_description",
                            "questions",
                        ],
                    },
                }
                if table_name not in table_config:
                    raise ValueError(
                        "table_name must be 'chunk', 'entity', or 'relationship'"
                    )
                config = table_config[table_name]
                index_name = f"{table_name}_fts"
                columns = ", ".join(config["columns"])
                id_column = config["id_column"]
                async with self.lock:
                    await self.conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS {index_name}
                        USING fts5(
                            {columns},
                            content='{table_name}',
                            content_rowid='{id_column}'
                        )
                        """
                    )
                    # Populate the index using records already in the table.
                    await self.conn.execute(
                        f"""
                        INSERT INTO {index_name}({index_name})
                        VALUES ('rebuild')
                        """
                    )
                    await self.conn.commit()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(
                    "Error in the create_BM25_index_for_table "
                    f"function inside DotVectorDB: {e}"
                )
    async def search_BM25(self, table_name: str, query: str, top_k: int = 10) -> tuple[list[dict], int]:
        """Searches and ranks rows from a table's BM25 index."""
        try:
            self.__require_open_connection()
            if table_name not in {"chunk", "entity", "relationship"}:
                raise ValueError(
                    "table_name must be 'chunk', 'entity', or 'relationship'"
                )
            search_terms = query.split()
            if not search_terms:
                return [], 0
            fts_query = " OR ".join(
                f'"{term.replace(chr(34), chr(34) * 2)}"'
                for term in search_terms
            )
            index_name = f"{table_name}_fts"
            async with self.lock:
                cursor = await self.conn.execute(
                    f"""
                    SELECT
                        fts.rowid AS bm25_rowid,
                        records.*,
                        bm25({index_name}) AS bm25_score
                    FROM {index_name} AS fts
                    JOIN {table_name} AS records
                        ON records.rowid = fts.rowid
                    WHERE {index_name} MATCH ?
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    (fts_query, top_k)
                )
                rows = await cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                return [dict(zip(column_names, row)) for row in rows], 0
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"Error in the search_BM25 function inside DotVectorDB: {e}"
            )
    ############################################################
    #   Hybrid Vector and BM25 Search                          # 
    ############################################################
    async def hybrid_vector_BM25_search(self, query, table, top_k=40, embedding_table=None, KNN=True, distance_metric="l2") -> tuple[list[dict], int]:
            self.__require_open_connection()
            try:
                candidate_k = top_k * 2
                rrf_k = 60

                (vector_results, query_tokens), (bm25_results, _) = await asyncio.gather(
                    self.vector_search(
                        query=query,
                        table=table,
                        top_k=candidate_k,
                        embedding_table=embedding_table,
                        KNN=KNN,
                        distance_metric=distance_metric,
                    ),
                    self.search_BM25(
                        table_name=table,
                        query=query,
                        top_k=candidate_k,
                    ),
                )

                fused_results = {}

                for rank, result in enumerate(vector_results, start=1):
                    rowid = result["vector_rowid"]
                    fused_results[rowid] = {
                        "result": dict(result),
                        "rrf_score": 1.0 / (rrf_k + rank),
                    }

                for rank, result in enumerate(bm25_results, start=1):
                    rowid = result["bm25_rowid"]
                    rrf_score = 1.0 / (rrf_k + rank)

                    if rowid in fused_results:
                        fused_results[rowid]["rrf_score"] += rrf_score
                        fused_results[rowid]["result"]["bm25_rowid"] = rowid
                        fused_results[rowid]["result"]["bm25_score"] = result["bm25_score"]
                    else:
                        fused_results[rowid] = {
                            "result": dict(result),
                            "rrf_score": rrf_score,
                        }
                        fused_results[rowid]["result"]["vector_rowid"] = rowid
                        fused_results[rowid]["result"]["similarity_score"] = None

                ranked_results = sorted(
                    fused_results.values(),
                    key=lambda result: result["rrf_score"],
                    reverse=True,
                )[:top_k]

                results = []
                for ranked_result in ranked_results:
                    result = ranked_result["result"]
                    result.setdefault("bm25_rowid", result["vector_rowid"])
                    result.setdefault("bm25_score", None)
                    result["rrf_score"] = ranked_result["rrf_score"]
                    results.append(result)

                return results, query_tokens
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(
                    "Error in the hybrid_vector_BM25_search function "
                    f"inside DotVectorDB: {e}"
                )
    ############################################################
    #   Functions For Pre-Determined Graph Construction        # 
    ############################################################
    @staticmethod
    def __does_entity_name_overlap_chunk(
        entity_name: str,
        chunk_content: str,
    ) -> tuple[bool, list[str]]:
        """Require every normalized entity-name token to occur in the chunk."""
        entity_tokens = set(re.findall(r"\w+", entity_name.casefold()))
        chunk_tokens = set(re.findall(r"\w+", chunk_content.casefold()))
        missing_tokens = sorted(entity_tokens - chunk_tokens)
        return bool(entity_tokens) and not missing_tokens, missing_tokens

    def __verify_source_json(self, source_json: list[dict]) -> dict[str, str]:
            """
            Expected Schema for the source:
            [
                {
                    "source_name": "The name of the source.",
                    "source_description": "A brief description of the source.",
                }
            ]
            Returns a mapping of each unique source name to its generated source ID.
            """
            source_name_to_id = {}
            if not isinstance(source_json, list):
                raise TypeError("Source JSON must be a list of source objects.")
            for index, source in enumerate(source_json):
                if not isinstance(source, dict):
                    raise TypeError(
                        f"Source at index {index} must be an object (dictionary)."
                    )
                if "source_name" not in source:
                    raise ValueError(
                        f"Source at index {index} is missing the required "
                        "'source_name' key."
                    )
                source_name = source["source_name"]
                if not isinstance(source_name, str):
                    raise TypeError(
                        f"'source_name' for source at index {index} must be a string."
                    )
                if not source_name.strip():
                    raise ValueError(
                        f"'source_name' for source at index {index} cannot be empty."
                    )
                source_id = str(
                    self.hashing_func.content_hash(source_name)
                )
                source["source_id"] = source_id
                source_name_to_id[source_name] = source_id
            return source_name_to_id
    def __verify_prechunked_corpus(
            self,
            source_name_to_id: dict[str, str],
            chunk_json_list: list[dict]
        ) -> tuple[dict, list[dict]]:
            """
            Expected Schema for each chunk in the list:
            [
                {
                    "chunk_id": "Unique identifier for the chunk.",
                    "content": "The actual text content of the chunk.",
                    "source_name": "Unique identifier for the source.",
                }
            ]
            Returns the local-to-generated chunk ID mapping and the enriched chunk
            JSON list.
            """
            if not isinstance(source_name_to_id, dict):
                raise TypeError("source_name_to_id must be a dictionary.")
            if not isinstance(chunk_json_list, list):
                raise TypeError("Chunk JSON must be a list of chunk objects.")
    
            local_chunk_ids_to_new_chunk_ids = {}
            seen_chunk_ids = set()
            seen_content = set()
    
            # Validate the complete input before modifying any chunk dictionaries.
            for index, chunk in enumerate(chunk_json_list):
                if not isinstance(chunk, dict):
                    raise TypeError(
                        f"Chunk at index {index} must be an object (dictionary)."
                    )
                if "chunk_id" not in chunk:
                    raise ValueError(
                        f"Chunk at index {index} is missing the required 'chunk_id' key."
                    )
                local_chunk_id = chunk["chunk_id"]
                try:
                    if local_chunk_id in seen_chunk_ids:
                        raise ValueError(
                            f"Duplicate chunk_id '{local_chunk_id}' found at index {index}."
                        )
                    seen_chunk_ids.add(local_chunk_id)
                except TypeError as exc:
                    raise TypeError(
                        f"'chunk_id' for chunk at index {index} must be hashable."
                    ) from exc
                if "source_name" not in chunk:
                    raise ValueError(
                        f"Chunk '{local_chunk_id}' is missing the required "
                        "'source_name' key."
                    )
                source_name = chunk["source_name"]
                if not isinstance(source_name, str):
                    raise TypeError(
                        f"'source_name' for chunk '{local_chunk_id}' must be a string."
                    )
                if source_name not in source_name_to_id:
                    raise ValueError(
                        f"Chunk '{local_chunk_id}' references unknown source_name "
                        f"'{source_name}'."
                    )
                if "content" not in chunk:
                    raise ValueError(
                        f"Chunk '{local_chunk_id}' is missing the required 'content' key."
                    )
                content = chunk["content"]
                if not isinstance(content, str):
                    raise TypeError(
                        f"'content' for chunk '{local_chunk_id}' must be a string."
                    )
                if not content.strip():
                    raise ValueError(
                        f"'content' for chunk '{local_chunk_id}' cannot be empty."
                    )
                if content in seen_content:
                    raise ValueError(
                        f"Duplicate content found for chunk '{local_chunk_id}'."
                    )
                seen_content.add(content)
            for chunk in chunk_json_list:
                local_chunk_id = chunk["chunk_id"]
                new_chunk_id = str(
                    self.hashing_func.content_hash(chunk["content"])
                )
                local_chunk_ids_to_new_chunk_ids[local_chunk_id] = new_chunk_id
                chunk["chunk_id"] = new_chunk_id
                chunk["source_id"] = source_name_to_id[chunk["source_name"]]
            return local_chunk_ids_to_new_chunk_ids, chunk_json_list
    def __verify_prebuilt_entity_json(
            self,
            local_chunks_to_new_chunks_ids: dict[str, str],
            chunk_json: list[dict],
            entity_json: list[dict]
        ) -> tuple[dict, list[dict]]:
            """
            TODO: ADD TOKEN COUNTING TO ENSURE IT DOESN'T GO OVER EMBEDDING WINDOW
                    Expected Schema for each entity in the list:
                    [
                        {
                            "entity_id": "Unique identifier for the entity.",
                            "chunk_id": "Unique identifier for the chunk associated with the entity.",
                            "entity_name": "The actual name of the entity.",
                            "entity_description": "The description of the entity.",
                            "entity_types": ["List of types associated with the entity."],
                            "questions": ["List of questions related to the entity."],
                        }
                    ]
            Returns the local-to-generated entity ID mapping and the enriched
            entity JSON list.
            """
            if not isinstance(local_chunks_to_new_chunks_ids, dict):
                raise TypeError(
                    "local_chunks_to_new_chunks_ids must be a dictionary."
                )
            if not isinstance(chunk_json, list):
                raise TypeError("Chunk JSON must be a list of chunk objects.")
            if not isinstance(entity_json, list):
                raise TypeError("Entity JSON must be a list of entity objects.")
            chunk_content_by_id = {
                chunk["chunk_id"]: chunk["content"]
                for chunk in chunk_json
            }
            local_entity_ids_to_new_entity_ids = {}
            seen_local_entity_ids = set()
            prepared_entities = []
            entities_with_multiple_types = []
            # Validate the complete input before modifying any entity dictionaries.
            for index, entity in enumerate(entity_json):
                if not isinstance(entity, dict):
                    raise TypeError(
                        f"Entity at index {index} must be an object (dictionary)."
                    )
                missing_keys = [
                    key
                    for key in ("entity_id", "chunk_id", "entity_name")
                    if key not in entity
                ]
                if missing_keys:
                    raise ValueError(
                        f"Entity at index {index} is missing required key(s): "
                        f"{', '.join(missing_keys)}."
                    )
    
                local_entity_id = entity["entity_id"]
                try:
                    if local_entity_id in seen_local_entity_ids:
                        raise ValueError(
                            f"Duplicate entity_id '{local_entity_id}' found at "
                            f"index {index}."
                        )
                    seen_local_entity_ids.add(local_entity_id)
                except TypeError as exc:
                    raise TypeError(
                        f"'entity_id' for entity at index {index} must be hashable."
                    ) from exc
    
                local_chunk_id = entity["chunk_id"]
                try:
                    new_chunk_id = local_chunks_to_new_chunks_ids[local_chunk_id]
                except TypeError as exc:
                    raise TypeError(
                        f"'chunk_id' for entity '{local_entity_id}' must be hashable."
                    ) from exc
                except KeyError as exc:
                    raise ValueError(
                        f"Entity '{local_entity_id}' references unknown chunk_id "
                        f"'{local_chunk_id}'."
                    ) from exc
    
                entity_name = entity["entity_name"]
                if not isinstance(entity_name, str):
                    raise TypeError(
                        f"'entity_name' for entity '{local_entity_id}' must be a string."
                    )
                normalized_entity_name = entity_name.strip().lower()
                if not normalized_entity_name:
                    raise ValueError(
                        f"'entity_name' for entity '{local_entity_id}' cannot be empty."
                    )

                name_matches_chunk, missing_name_tokens = (
                    self.__does_entity_name_overlap_chunk(
                        normalized_entity_name,
                        chunk_content_by_id[new_chunk_id],
                    )
                )
                if not name_matches_chunk:
                    raise ValueError(
                        f"Entity '{local_entity_id}' claims entity_name "
                        f"'{entity_name}', but that name is not present in "
                        f"chunk '{local_chunk_id}'. Missing normalized token(s): "
                        f"{missing_name_tokens}."
                    )
    
                entity_description = entity.get("entity_description", "")
                if entity_description is None:
                    entity_description = ""
                if not isinstance(entity_description, str):
                    raise TypeError(
                        f"'entity_description' for entity '{local_entity_id}' "
                        "must be a string."
                    )
    
                entity_types = entity.get("entity_types", [])
                if entity_types is None:
                    entity_types = []
                if not isinstance(entity_types, list):
                    raise TypeError(
                        f"'entity_types' for entity '{local_entity_id}' must be a list."
                    )
                if any(not isinstance(entity_type, str) for entity_type in entity_types):
                    raise TypeError(
                        f"Every entity type for entity '{local_entity_id}' must be "
                        "a string."
                    )
    
                questions = entity.get("questions", [])
                if questions is None:
                    questions = []
                if not isinstance(questions, list):
                    raise TypeError(
                        f"'questions' for entity '{local_entity_id}' must be a list."
                    )
                if any(not isinstance(question, str) for question in questions):
                    raise TypeError(
                        f"Every question for entity '{local_entity_id}' must be a string."
                    )
    
                if len(entity_types) > 1:
                    entities_with_multiple_types.append(local_entity_id)
    
                new_entity_id = self.hashing_func.content_hash(
                    normalized_entity_name
                )
                prepared_entities.append(
                    (
                        entity,
                        local_entity_id,
                        new_entity_id,
                        new_chunk_id,
                        normalized_entity_name,
                        entity_description,
                        entity_types,
                        questions,
                    )
                )
    
            if entities_with_multiple_types:
                warnings.warn(
                    "Entities with local entity_id(s) "
                    f"{entities_with_multiple_types} have more than one entity type. "
                    "They are supported by DotRAG but may be incompatible with "
                    "GraphRAG, LightRAG, PathRAG, and other methods that require "
                    "exactly one type per node.",
                    UserWarning,
                    stacklevel=2,
                )
    
            for (
                entity,
                local_entity_id,
                new_entity_id,
                new_chunk_id,
                normalized_entity_name,
                entity_description,
                entity_types,
                questions,
            ) in prepared_entities:
                local_entity_ids_to_new_entity_ids[local_entity_id] = new_entity_id
                entity["entity_id"] = new_entity_id
                entity["chunk_id"] = new_chunk_id
                entity["entity_name"] = normalized_entity_name
                entity["entity_description"] = entity_description
                entity["entity_types"] = entity_types
                entity["questions"] = questions
                entity["_is_entity_enriched"] = bool(entity_description.strip())
                entity["does_entity_have_types"] = bool(entity_types)
                entity["does_entity_have_questions"] = bool(questions)
    
            return local_entity_ids_to_new_entity_ids, entity_json
    def __verify_entity_types(
            self,
            entity_json: list[dict],
            entity_types_json: dict[str, str] = None,
        ) -> dict[str, str]:
            """Verify definitions exist for every custom entity type in use."""
            if not isinstance(entity_json, list):
                raise TypeError("Entity JSON must be a list of entity objects.")

            used_entity_types = {
                entity_type
                for entity in entity_json
                for entity_type in entity.get("entity_types", [])
            }
            if not used_entity_types:
                if entity_types_json is None:
                    return {}
                if not isinstance(entity_types_json, dict):
                    raise TypeError(
                        "entity_types_json must be a dictionary mapping entity "
                        "type names to definitions."
                    )
                return entity_types_json

            if entity_types_json is None:
                raise ValueError(
                    "entity_types_json is required because the entity JSON "
                    "contains entity types."
                )
            if not isinstance(entity_types_json, dict):
                raise TypeError(
                    "entity_types_json must be a dictionary mapping entity "
                    "type names to definitions."
                )

            invalid_definitions = [
                entity_type
                for entity_type, definition in entity_types_json.items()
                if not isinstance(entity_type, str)
                or not entity_type.strip()
                or not isinstance(definition, str)
                or not definition.strip()
            ]
            if invalid_definitions:
                raise ValueError(
                    "Every entity type and definition in entity_types_json "
                    "must be a nonempty string. Invalid type(s): "
                    f"{invalid_definitions}."
                )

            missing_definitions = sorted(
                used_entity_types - set(entity_types_json)
            )
            if missing_definitions:
                raise ValueError(
                    "Missing definitions in entity_types_json for entity "
                    f"type(s): {missing_definitions}."
                )

            return entity_types_json

    def __verify_prebuilt_relationship_json(
            self,
            old_chunk_id_to_new_chunk_id: dict[str, str],
            old_entity_id_to_new_entity_id: dict[str, str],
            chunk_json: list[dict],
            entity_json: list[dict],
            relationship_json: list[dict]
        ) -> list[dict]:
            """
            TODO: ADD TOKEN COUNTING TO ENSURE IT DOESN'T GO OVER EMBEDDING WINDOW
            Expected Schema for each relationship in the list:
            [
                {
                    "chunk_id": "Unique identifier for the associated chunk.",
                    "src_entity_id": "Unique identifier for the entity.",
                    "dst_entity_id": "Unique identifier for the related entity.",
                    "relation_description": "The description of the entity.",
                    "keywords": ["List of types associated with the entity."],
                    "weight": "The weight of the relationship.",
                    "questions": ["List of questions related to the entity."],
                }
            ]
    
            The chunk ID and source and destination entity IDs are required.
            Optional metadata is defaulted and marked for later enrichment.
            """
            if not isinstance(old_chunk_id_to_new_chunk_id, dict):
                raise TypeError(
                    "old_chunk_id_to_new_chunk_id must be a dictionary."
                )
            if not isinstance(old_entity_id_to_new_entity_id, dict):
                raise TypeError(
                    "old_entity_id_to_new_entity_id must be a dictionary."
                )
            if not isinstance(chunk_json, list):
                raise TypeError("Chunk JSON must be a list of chunk objects.")
            if not isinstance(entity_json, list):
                raise TypeError("Entity JSON must be a list of entity objects.")
            if not isinstance(relationship_json, list):
                raise TypeError(
                    "Relationship JSON must be a list of relationship objects."
                )
            chunk_content_by_id = {
                chunk["chunk_id"]: chunk["content"]
                for chunk in chunk_json
            }
            entity_name_by_id = {
                entity["entity_id"]: entity["entity_name"]
                for entity in entity_json
            }
    
            prepared_relationships = []
    
            # Validate the complete input before modifying any relationships.
            for index, relationship in enumerate(relationship_json):
                if not isinstance(relationship, dict):
                    raise TypeError(
                        f"Relationship at index {index} must be an object "
                        "(dictionary)."
                    )
    
                missing_keys = [
                    key
                    for key in ("chunk_id", "src_entity_id", "dst_entity_id")
                    if key not in relationship
                ]
                if missing_keys:
                    raise ValueError(
                        f"Relationship at index {index} is missing required "
                        f"key(s): {', '.join(missing_keys)}."
                    )
    
                old_chunk_id = relationship["chunk_id"]
                try:
                    new_chunk_id = old_chunk_id_to_new_chunk_id[old_chunk_id]
                except TypeError as exc:
                    raise TypeError(
                        f"'chunk_id' for relationship at index {index} must "
                        "be hashable."
                    ) from exc
                except KeyError as exc:
                    raise ValueError(
                        f"Relationship at index {index} references unknown "
                        f"chunk_id '{old_chunk_id}'."
                    ) from exc

                old_src_entity_id = relationship["src_entity_id"]
                old_dst_entity_id = relationship["dst_entity_id"]
                try:
                    new_src_entity_id = old_entity_id_to_new_entity_id[
                        old_src_entity_id
                    ]
                except TypeError as exc:
                    raise TypeError(
                        f"'src_entity_id' for relationship at index {index} "
                        "must be hashable."
                    ) from exc
                except KeyError as exc:
                    raise ValueError(
                        f"Relationship at index {index} references unknown "
                        f"src_entity_id '{old_src_entity_id}'."
                    ) from exc
    
                try:
                    new_dst_entity_id = old_entity_id_to_new_entity_id[
                        old_dst_entity_id
                    ]
                except TypeError as exc:
                    raise TypeError(
                        f"'dst_entity_id' for relationship at index {index} "
                        "must be hashable."
                    ) from exc
                except KeyError as exc:
                    raise ValueError(
                        f"Relationship at index {index} references unknown "
                        f"dst_entity_id '{old_dst_entity_id}'."
                    ) from exc

                for endpoint_label, endpoint_id, endpoint_name in (
                    (
                        "src_entity_id",
                        old_src_entity_id,
                        entity_name_by_id[new_src_entity_id],
                    ),
                    (
                        "dst_entity_id",
                        old_dst_entity_id,
                        entity_name_by_id[new_dst_entity_id],
                    ),
                ):
                    name_matches_chunk, missing_name_tokens = (
                        self.__does_entity_name_overlap_chunk(
                            endpoint_name,
                            chunk_content_by_id[new_chunk_id],
                        )
                    )
                    if not name_matches_chunk:
                        raise ValueError(
                            f"Relationship at index {index} references "
                            f"{endpoint_label} '{endpoint_id}' with entity_name "
                            f"'{endpoint_name}', but that name is not present in "
                            f"chunk '{old_chunk_id}'. Missing normalized "
                            f"token(s): {missing_name_tokens}."
                        )
    
                relation_description = relationship.get("relation_description", "")
                if relation_description is None:
                    relation_description = ""
                if not isinstance(relation_description, str):
                    raise TypeError(
                        f"'relation_description' for relationship at index "
                        f"{index} must be a string."
                    )
    
                keywords = relationship.get("keywords", [])
                if keywords is None:
                    keywords = []
                if not isinstance(keywords, list):
                    raise TypeError(
                        f"'keywords' for relationship at index {index} must be a list."
                    )
                if any(not isinstance(keyword, str) for keyword in keywords):
                    raise TypeError(
                        f"Every keyword for relationship at index {index} must "
                        "be a string."
                    )
    
                questions = relationship.get("questions", [])
                if questions is None:
                    questions = []
                if not isinstance(questions, list):
                    raise TypeError(
                        f"'questions' for relationship at index {index} must be a list."
                    )
                if any(not isinstance(question, str) for question in questions):
                    raise TypeError(
                        f"Every question for relationship at index {index} must "
                        "be a string."
                    )
    
                weight = relationship.get("weight")
                if weight is not None and (
                    isinstance(weight, bool) or not isinstance(weight, (int, float))
                ):
                    raise TypeError(
                        f"'weight' for relationship at index {index} must be a number."
                    )
    
                relation_id = self.hashing_func.content_hash(
                    f"{new_src_entity_id}_{new_dst_entity_id}"
                )
                prepared_relationships.append(
                    (
                        relationship,
                        relation_id,
                        new_chunk_id,
                        new_src_entity_id,
                        new_dst_entity_id,
                        relation_description,
                        keywords,
                        weight,
                        questions,
                    )
                )
    
            for (
                relationship,
                relation_id,
                new_chunk_id,
                new_src_entity_id,
                new_dst_entity_id,
                relation_description,
                keywords,
                weight,
                questions,
            ) in prepared_relationships:
                relationship["relation_id"] = relation_id
                relationship["chunk_id"] = new_chunk_id
                relationship["src_entity_id"] = new_src_entity_id
                relationship["dst_entity_id"] = new_dst_entity_id
                relationship["relation_description"] = relation_description
                relationship["keywords"] = keywords
                relationship["weight"] = weight
                relationship["questions"] = questions
                relationship["_is_relationship_enriched"] = bool(
                    relation_description.strip()
                )
                relationship["does_relationship_have_keywords"] = bool(keywords)
                relationship["does_relationship_have_weight"] = weight is not None
                relationship["does_relationship_have_questions"] = bool(questions)
            return relationship_json
    def __verify_fixed_knowledge_graph(
        self,
        entity_json: list[dict],
        rel_json: list[dict]
    ) -> igraph.Graph:
        """Build a fixed knowledge graph from verified entity and relation JSON."""
        if not isinstance(entity_json, list):
            raise TypeError("Entity JSON must be a list of entity objects.")
        if not isinstance(rel_json, list):
            raise TypeError(
                "Relationship JSON must be a list of relationship objects."
            )

        # Entity extraction can associate the same normalized entity with more
        # than one chunk. Keep one graph vertex per generated entity ID, just as
        # the entity table and normal graph-construction path do.
        entities_by_id = {
            str(entity["entity_id"]): entity
            for entity in entity_json
        }
        relationships_by_id = {
            str(relationship["relation_id"]): relationship
            for relationship in rel_json
        }

        graph = igraph.Graph(directed=False)
        graph.add_vertices(
            list(entities_by_id),
            attributes={
                "entity_type": [
                    entity.get("entity_types", [])
                    for entity in entities_by_id.values()
                ],
                "entity_name": [
                    entity["entity_name"]
                    for entity in entities_by_id.values()
                ],
                "entity_description": [
                    entity.get("entity_description", "")
                    for entity in entities_by_id.values()
                ],
            },
        )

        graph.add_edges(
            [
                (
                    str(relationship["src_entity_id"]),
                    str(relationship["dst_entity_id"]),
                )
                for relationship in relationships_by_id.values()
            ],
            attributes={
                "relation_id": [
                    relationship["relation_id"]
                    for relationship in relationships_by_id.values()
                ],
                "relation_description": [
                    relationship.get("relation_description", "")
                    for relationship in relationships_by_id.values()
                ],
            },
        )
        return graph
    async def ingest_and_embed_custom_entities(self, entity_json, entity_types=None, async_cooldown=60, requests_per_minute=1000) -> None:
           self.__require_open_connection()
           try:
               await self.add_entity_types(entity_types)
               registered_entity_types = await self.get_entity_types()
               entity_indexer = EntityIndexer4(
                   conn=self.conn,
                   lock=self.lock,
                   language_model=self.language_model.generate,
                   embedding_model=self.embedding_model,
                   hash_func=self.hashing_func.content_hash,
                   entity_groups=registered_entity_types,
                   add_embedding_func=self.add_embedding,
               )
               await entity_indexer.ingest_custom_entities_from_ingested_chunks(
                   custom_entities_json=entity_json,
                   async_cooldown=async_cooldown,
                   requests_per_minute=requests_per_minute,
                   verbose=self.verbose,
               )
               await entity_indexer.fix_entities(
                   async_cooldown=async_cooldown,
                   requests_per_minute=requests_per_minute,
                   verbose=self.verbose,
               )
               await entity_indexer.embed_entities(
                   async_cooldown=async_cooldown,
                   requests_per_minute=requests_per_minute,
                   verbose=self.verbose,
               )
           except Exception as e:
               traceback.print_exc()
               raise RuntimeError(
                   "Error ingesting and embedding custom entities: "
                   f"{e}"
               ) from e

    async def ingest_and_embed_custom_relationships(self, relationship_json, async_cooldown=60, requests_per_minute=1000) -> None:
           self.__require_open_connection()
           try:
               relationship_indexer = RelationshipIndexer3(
                   conn=self.conn,
                   lock=self.lock,
                   language_model=self.language_model.generate,
                   hash_func=self.hashing_func.content_hash,
                   embedding_model=self.embedding_model,
                   add_embedding_func=self.add_embedding,
               )
               await relationship_indexer.ingest_custom_relationships_from_ingested_chunks(
                   custom_relationships_json=relationship_json,
                   async_cooldown=async_cooldown,
                   requests_per_minute=requests_per_minute,
                   verbose=self.verbose,
               )
               await relationship_indexer.fix_relationships(
                   async_cooldown=async_cooldown,
                   requests_per_minute=requests_per_minute,
                   verbose=self.verbose,
               )
               await relationship_indexer.embed_relationships(
                   async_cooldown=async_cooldown,
                   requests_per_minute=requests_per_minute,
                   verbose=self.verbose,
               )
           except Exception as e:
               traceback.print_exc()
               raise RuntimeError(
                   "Error ingesting and embedding custom relationships: "
                   f"{e}"
               ) from e
    async def construct_custom_GraphRAG(self, chunks: list[dict], graph: igraph.Graph = None,  entities: list[dict] = None, rels_json: list[dict] = None, entity_types_json: dict[str, str] = None, debug_mode=False, async_cooldown=60, requests_per_minute=1000) -> None:
                   # First, validate the provided information before we make any llm or embedding calls.
                   try: 
                       self.__graph_construction_print("STEP ONE: COLLECTING SOURCE DOCUMENTS AND VERIFYING THEIR FORMAT")
                       source_name_to_id = self.__verify_source_json(chunks)
                   except Exception as e:
                       traceback.print_exc()
                       raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the source JSON: {e}")
                   try: 
                        self.__graph_construction_print("STEP TWO: VERIFYING THE FORMAT OF PROVIDED CHUNKS")
                        old_chunk_ids_to_new_chunk_ids, chunk_json_list = self.__verify_prechunked_corpus(source_name_to_id, chunks)
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the chunk JSON: {e}")
                   try: 
                        self.__graph_construction_print("STEP THREE: VALIDATING PROVIDED ENTITY JSON FORMAT")
                        local_entity_ids_to_new_entity_ids, entity_json = self.__verify_prebuilt_entity_json(
                            old_chunk_ids_to_new_chunk_ids,
                            chunk_json_list,
                            entities,
                        )
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the entity JSON: {e}")
                   try:
                        self.__graph_construction_print("STEP THREE-B: VALIDATING PROVIDED ENTITY TYPE DEFINITIONS")
                        entity_types_json = self.__verify_entity_types(
                            entity_json,
                            entity_types_json,
                        )
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying entity type definitions: {e}")
                   try:
                        self.__graph_construction_print("STEP FOUR: VALIDATING PROVIDED RELATIONSHIP JSON FORMAT")
                        relationship_json = self.__verify_prebuilt_relationship_json(
                            old_chunk_ids_to_new_chunk_ids,
                            local_entity_ids_to_new_entity_ids,
                            chunk_json_list,
                            entity_json,
                            rels_json,
                        )
                   except Exception as e:
                          traceback.print_exc()
                          raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the relationship JSON: {e}")
                   try:
                       self.__graph_construction_print("STEP FIVE: CREATING THE INITIAL GRAPH REPRESENTATION OF THE DATABASE")
                       graph = self.__verify_fixed_knowledge_graph(
                           entity_json,
                           relationship_json,
                       )
                   except Exception as e:
                       traceback.print_exc()
                       raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while verifying the graph: {e}")
                    
                   # Second, we ingest the chunks into the database.
                   try:
                        total_chunks = []
                        self.__graph_construction_print("STEP SIX: INGESTING AND EMBEDDING THE CHUNKS")
                        for chunk in tqdm_asyncio(chunk_json_list, desc=f"Adding chunks from custom chunks into the DB"):
                            await self.add_chunk(
                                content=chunk["content"],
                                source_name=chunk["source_name"],
                                model_name=self.embedding_model.model_name
                            )
                            total_chunks.append(chunk)

                        failed_chunks = await self.get_unprocessed_metadata("chunk")
                        while failed_chunks:
                            current_batch = failed_chunks[:requests_per_minute]
                            remaining_chunks = failed_chunks[requests_per_minute:]

                            async def embed_custom_chunk(chunk):
                                try:
                                    await self.add_embedding(
                                        content=chunk["content"],
                                        table="chunk",
                                        pk_hash=chunk["chunk_id"],
                                    )
                                    return None
                                except Exception as error:
                                    if self.verbose:
                                        print(
                                            "Custom chunk embedding failed; "
                                            f"retrying: {error}"
                                        )
                                    return chunk

                            batch_results = await tqdm_asyncio.gather(
                                *(
                                    embed_custom_chunk(chunk)
                                    for chunk in current_batch
                                ),
                                desc="Embedding custom chunks",
                                total=len(current_batch),
                            )
                            failed_chunks = [
                                chunk
                                for chunk in batch_results
                                if chunk is not None
                            ]
                            failed_chunks.extend(remaining_chunks)
                            if failed_chunks:
                                await asyncio.sleep(async_cooldown)
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while adding chunks to the database: {e}")

                   # Third, construct the graph from the provided entity and relationship json
                   try:
                        self.__graph_construction_print("STEP SEVEN: INGESTING AND EMBEDDING CUSTOM ENTITIES")
                        await self.ingest_and_embed_custom_entities(
                            entity_json=entity_json,
                            entity_types=entity_types_json or None,
                            async_cooldown=async_cooldown,
                            requests_per_minute=requests_per_minute,
                        )
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(
                            "Error in construct_custom_GraphRAG while ingesting "
                            f"and embedding custom entities: {e}"
                        ) from e

                   try:
                        self.__graph_construction_print("STEP EIGHT: INGESTING AND EMBEDDING CUSTOM RELATIONSHIPS")
                        await self.ingest_and_embed_custom_relationships(
                            relationship_json=relationship_json,
                            async_cooldown=async_cooldown,
                            requests_per_minute=requests_per_minute,
                        )
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(
                            "Error in construct_custom_GraphRAG while ingesting "
                            f"and embedding custom relationships: {e}"
                        ) from e

                   # Fourth, generate and save a global description for each source document.
                   try:
                      self.__graph_construction_print("STEP FOUR: Generating a Global Description for the entity-relationship graph")
                      await self.generate_global_descriptions(debug_mode=debug_mode)
                   except Exception as e:
                      traceback.print_exc()
                      raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while generating global descriptions: {e}")
                    
                   # Fifth, we need to generate the graph representation of the database.
                   try:
                     self.__graph_construction_print("STEP FIVE: Creating the Graph Representation of the Database")
                     await self.graph_retriever.load_graph()
                   except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the construct_custom_GraphRAG function inside DotVectorDB while clustering entities: {e}")
                   # Lastly, we construct the BM25 index
                   try:
                      self.__graph_construction_print("STEP SIX: Creating the BM25 Index")
                      await self.create_BM25_index_for_table("chunk")
                      await self.create_BM25_index_for_table("entity")
                      await self.create_BM25_index_for_table("relationship")  
                   except Exception as e:
                      traceback.print_exc()
                      raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while creating the BM25 index: {e}")

    ############################################################
    #  Graph Convert Functions From DotRAG to Another          # 
    ############################################################
    async def export_DotRAG_to_LightRAG_PathRAG_Graph(
        self,
        output_file="lightrag_custom_kg.json",
    ) -> dict[str, list[dict]]:
        """Export DotRAG data in LightRAG/PathRAG custom-KG format.

        The returned dictionary can be passed directly to LightRAG's
        ``insert_custom_kg``/``ainsert_custom_kg`` methods. When
        ``output_file`` is not ``None``, the same payload is also written as
        JSON. Relationship weights are derived from the number of distinct
        chunks supporting each relationship, matching LightRAG's evidence
        weight semantics.
        """
        self.__require_open_connection()

        async with self.lock:
            chunk_cursor = await self.conn.execute(
                """
                SELECT chunk.chunk_id, chunk.content, chunk.source_id,
                       source.name
                FROM chunk
                JOIN source ON chunk.source_id = source.id
                ORDER BY chunk.source_id, chunk.chunk_id
                """
            )
            chunk_rows = await chunk_cursor.fetchall()
            await chunk_cursor.close()

            entity_cursor = await self.conn.execute(
                """
                SELECT entity_id, entity_name, entity_description
                FROM entity
                ORDER BY entity_id
                """
            )
            entity_rows = await entity_cursor.fetchall()
            await entity_cursor.close()

            entity_type_cursor = await self.conn.execute(
                """
                SELECT entity_id, entity_type
                FROM entity_to_entity_type_mapping
                ORDER BY entity_id, entity_type
                """
            )
            entity_type_rows = await entity_type_cursor.fetchall()
            await entity_type_cursor.close()

            entity_chunk_cursor = await self.conn.execute(
                """
                SELECT entity_id, chunk_id
                FROM chunk_to_entity_mapping
                ORDER BY entity_id, chunk_id
                """
            )
            entity_chunk_rows = await entity_chunk_cursor.fetchall()
            await entity_chunk_cursor.close()

            relationship_cursor = await self.conn.execute(
                """
                SELECT relationship.relation_id,
                       source_entity.entity_name,
                       destination_entity.entity_name,
                       relationship.relation_description,
                       relationship.keywords
                FROM relationship
                JOIN entity AS source_entity
                    ON relationship.src_entity_id = source_entity.entity_id
                JOIN entity AS destination_entity
                    ON relationship.dst_entity_id = destination_entity.entity_id
                ORDER BY relationship.relation_id
                """
            )
            relationship_rows = await relationship_cursor.fetchall()
            await relationship_cursor.close()

            relationship_chunk_cursor = await self.conn.execute(
                """
                SELECT relation_id, chunk_id
                FROM chunk_to_relationship_mapping
                ORDER BY relation_id, chunk_id
                """
            )
            relationship_chunk_rows = (
                await relationship_chunk_cursor.fetchall()
            )
            await relationship_chunk_cursor.close()

        chunk_source_names = {
            chunk_id: source_name
            for chunk_id, _, _, source_name in chunk_rows
        }

        entity_types_by_id = {}
        for entity_id, entity_type in entity_type_rows:
            entity_types_by_id.setdefault(entity_id, []).append(entity_type)

        entity_chunks_by_id = {}
        for entity_id, chunk_id in entity_chunk_rows:
            entity_chunks_by_id.setdefault(entity_id, set()).add(chunk_id)

        relationship_chunks_by_id = {}
        for relation_id, chunk_id in relationship_chunk_rows:
            relationship_chunks_by_id.setdefault(relation_id, set()).add(
                chunk_id
            )

        def provenance(chunk_ids):
            ordered_chunk_ids = sorted(chunk_ids)
            source_id = "<SEP>".join(
                str(chunk_id) for chunk_id in ordered_chunk_ids
            )
            file_paths = []
            seen_file_paths = set()
            for chunk_id in ordered_chunk_ids:
                file_path = chunk_source_names.get(chunk_id)
                if file_path and file_path not in seen_file_paths:
                    seen_file_paths.add(file_path)
                    file_paths.append(file_path)
            return source_id, "<SEP>".join(file_paths)

        chunks = []
        source_chunk_indexes = {}
        for chunk_id, content, source_id, source_name in chunk_rows:
            source_chunk_index = source_chunk_indexes.get(source_id, 0)
            chunks.append(
                {
                    "content": content,
                    "source_id": str(chunk_id),
                    "file_path": source_name,
                    "source_chunk_index": source_chunk_index,
                }
            )
            source_chunk_indexes[source_id] = source_chunk_index + 1

        entities = []
        for entity_id, entity_name, entity_description in entity_rows:
            source_id, file_path = provenance(
                entity_chunks_by_id.get(entity_id, set())
            )
            entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": ", ".join(
                        entity_types_by_id.get(entity_id, [])
                    ),
                    "description": entity_description,
                    "source_id": source_id,
                    "file_path": file_path,
                }
            )

        relationships = []
        for (
            relation_id,
            source_entity_name,
            destination_entity_name,
            relation_description,
            raw_keywords,
        ) in relationship_rows:
            chunk_ids = relationship_chunks_by_id.get(relation_id, set())
            source_id, file_path = provenance(chunk_ids)
            try:
                keywords = json.loads(raw_keywords)
            except (TypeError, json.JSONDecodeError):
                keywords = raw_keywords
            if isinstance(keywords, list):
                keywords = ", ".join(str(keyword) for keyword in keywords)
            elif keywords is None:
                keywords = ""
            else:
                keywords = str(keywords)

            relationships.append(
                {
                    "src_id": source_entity_name,
                    "tgt_id": destination_entity_name,
                    "description": relation_description,
                    "keywords": keywords,
                    "weight": float(len(chunk_ids)),
                    "source_id": source_id,
                    "file_path": file_path,
                }
            )

        custom_kg = {
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
        }

        if output_file is not None:
            output_path = Path(output_file).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(custom_kg, file, ensure_ascii=False, indent=2)

        return custom_kg

    async def export_DotRAG_to_HopRAG_Graph(
        self,
        output_file="hoprag_hotpotqa.jsonl",
    ) -> list[dict]:
        """Export the DotRAG corpus as HopRAG-compatible HotpotQA JSONL.

        HopRAG generates its own answerable and pending questions from the
        passage text. Its HotpotQA builder uses each record's ``context`` to
        determine which documents may be compared during edge construction.
        This exporter therefore groups DotRAG sources into connected
        components using their shared entities and relationships, while
        leaving all QA ground-truth fields empty.
        """
        self.__require_open_connection()

        async with self.lock:
            source_cursor = await self.conn.execute(
                """
                SELECT id, name
                FROM source
                ORDER BY id
                """
            )
            source_rows = await source_cursor.fetchall()
            await source_cursor.close()

            chunk_cursor = await self.conn.execute(
                """
                SELECT chunk_id, source_id, content
                FROM chunk
                ORDER BY source_id, chunk_id
                """
            )
            chunk_rows = await chunk_cursor.fetchall()
            await chunk_cursor.close()

            entity_source_cursor = await self.conn.execute(
                """
                SELECT DISTINCT mapping.entity_id, chunk.source_id
                FROM chunk_to_entity_mapping AS mapping
                JOIN chunk ON mapping.chunk_id = chunk.chunk_id
                ORDER BY mapping.entity_id, chunk.source_id
                """
            )
            entity_source_rows = await entity_source_cursor.fetchall()
            await entity_source_cursor.close()

            relationship_source_cursor = await self.conn.execute(
                """
                SELECT DISTINCT mapping.relation_id, chunk.source_id
                FROM chunk_to_relationship_mapping AS mapping
                JOIN chunk ON mapping.chunk_id = chunk.chunk_id
                ORDER BY mapping.relation_id, chunk.source_id
                """
            )
            relationship_source_rows = (
                await relationship_source_cursor.fetchall()
            )
            await relationship_source_cursor.close()

        source_names = dict(source_rows)
        chunks_by_source = {source_id: [] for source_id in source_names}
        for _, source_id, content in chunk_rows:
            chunks_by_source.setdefault(source_id, []).append(content)

        parent = {source_id: source_id for source_id in source_names}

        def find(source_id):
            while parent[source_id] != source_id:
                parent[source_id] = parent[parent[source_id]]
                source_id = parent[source_id]
            return source_id

        def union(first_source_id, second_source_id):
            first_root = find(first_source_id)
            second_root = find(second_source_id)
            if first_root != second_root:
                parent[second_root] = first_root

        def connect_sources(rows):
            sources_by_graph_item = {}
            for graph_item_id, source_id in rows:
                if source_id in parent:
                    sources_by_graph_item.setdefault(
                        graph_item_id,
                        [],
                    ).append(source_id)
            for source_ids in sources_by_graph_item.values():
                if len(source_ids) < 2:
                    continue
                first_source_id = source_ids[0]
                for source_id in source_ids[1:]:
                    union(first_source_id, source_id)

        connect_sources(entity_source_rows)
        connect_sources(relationship_source_rows)

        components = {}
        for source_id in source_names:
            if chunks_by_source.get(source_id):
                components.setdefault(find(source_id), []).append(source_id)

        name_counts = {}
        for source_name in source_names.values():
            name_counts[source_name] = name_counts.get(source_name, 0) + 1

        def hoprag_document_title(source_id):
            source_name = source_names[source_id]
            if name_counts[source_name] == 1:
                return source_name
            return f"{source_name} [{source_id}]"

        records = []
        ordered_components = sorted(
            components.values(),
            key=lambda source_ids: tuple(str(value) for value in source_ids),
        )
        for component_index, source_ids in enumerate(ordered_components, 1):
            context = [
                [
                    hoprag_document_title(source_id),
                    chunks_by_source[source_id],
                ]
                for source_id in source_ids
            ]
            records.append(
                {
                    "_id": f"dotrag-component-{component_index}",
                    "answer": "",
                    "question": "",
                    "supporting_facts": [],
                    "context": context,
                    "type": "dotrag",
                    "level": "unknown",
                }
            )

        if output_file is not None:
            output_path = Path(output_file).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                for record in records:
                    file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

        return records

    async def export_DotRAG_to_MS_GraphRAG_Graph(
        self,
        output_file="graphrag_output",
    ) -> dict[str, str]:
        """Export DotRAG data as Microsoft GraphRAG BYOG Parquet tables."""
        self.__require_open_connection()
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_path = Path(output_file).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        async with self.lock:
            entity_cursor = await self.conn.execute(
                """
                SELECT entity_id, entity_name, entity_description, degrees
                FROM entity
                ORDER BY entity_id
                """
            )
            entity_rows = await entity_cursor.fetchall()
            await entity_cursor.close()

            entity_type_cursor = await self.conn.execute(
                """
                SELECT entity_id, entity_type
                FROM entity_to_entity_type_mapping
                ORDER BY entity_id, entity_type
                """
            )
            entity_type_rows = await entity_type_cursor.fetchall()
            await entity_type_cursor.close()

            entity_chunk_cursor = await self.conn.execute(
                """
                SELECT entity_id, chunk_id
                FROM chunk_to_entity_mapping
                ORDER BY entity_id, chunk_id
                """
            )
            entity_chunk_rows = await entity_chunk_cursor.fetchall()
            await entity_chunk_cursor.close()

            relationship_cursor = await self.conn.execute(
                """
                SELECT
                    relationship.relation_id,
                    source.entity_name,
                    target.entity_name,
                    relationship.relation_description,
                    relationship.weight,
                    source.degrees,
                    target.degrees
                FROM relationship
                JOIN entity AS source
                    ON relationship.src_entity_id = source.entity_id
                JOIN entity AS target
                    ON relationship.dst_entity_id = target.entity_id
                ORDER BY relationship.relation_id
                """
            )
            relationship_rows = await relationship_cursor.fetchall()
            await relationship_cursor.close()

            relationship_chunk_cursor = await self.conn.execute(
                """
                SELECT relation_id, chunk_id
                FROM chunk_to_relationship_mapping
                ORDER BY relation_id, chunk_id
                """
            )
            relationship_chunk_rows = (
                await relationship_chunk_cursor.fetchall()
            )
            await relationship_chunk_cursor.close()

            chunk_cursor = await self.conn.execute(
                """
                SELECT chunk_id, content, source_id
                FROM chunk
                ORDER BY chunk_id
                """
            )
            chunk_rows = await chunk_cursor.fetchall()
            await chunk_cursor.close()

        entity_types_by_id = {}
        for entity_id, entity_type in entity_type_rows:
            entity_types_by_id.setdefault(entity_id, []).append(entity_type)

        entity_chunks_by_id = {}
        chunk_entities_by_id = {}
        for entity_id, chunk_id in entity_chunk_rows:
            entity_chunks_by_id.setdefault(entity_id, []).append(str(chunk_id))
            chunk_entities_by_id.setdefault(chunk_id, []).append(str(entity_id))

        relationship_chunks_by_id = {}
        chunk_relationships_by_id = {}
        for relation_id, chunk_id in relationship_chunk_rows:
            relationship_chunks_by_id.setdefault(relation_id, []).append(
                str(chunk_id)
            )
            chunk_relationships_by_id.setdefault(chunk_id, []).append(
                str(relation_id)
            )

        entity_records = []
        for human_readable_id, (
            entity_id,
            entity_name,
            entity_description,
            degree,
        ) in enumerate(entity_rows):
            entity_types = entity_types_by_id.get(entity_id, [])
            entity_text_unit_ids = entity_chunks_by_id.get(entity_id, [])
            entity_records.append(
                {
                    "id": str(entity_id),
                    "human_readable_id": human_readable_id,
                    "title": entity_name,
                    "type": ", ".join(entity_types),
                    "description": entity_description,
                    "text_unit_ids": entity_text_unit_ids,
                    "frequency": len(entity_text_unit_ids),
                    "degree": int(degree or 0),
                }
            )

        relationship_records = []
        for human_readable_id, (
            relation_id,
            source_name,
            target_name,
            description,
            weight,
            source_degree,
            target_degree,
        ) in enumerate(relationship_rows):
            relationship_records.append(
                {
                    "id": str(relation_id),
                    "human_readable_id": human_readable_id,
                    "source": source_name,
                    "target": target_name,
                    "description": description,
                    "weight": float(weight),
                    "combined_degree": int(source_degree or 0)
                    + int(target_degree or 0),
                    "text_unit_ids": relationship_chunks_by_id.get(
                        relation_id,
                        [],
                    ),
                }
            )

        text_unit_records = []
        for human_readable_id, (
            chunk_id,
            content,
            source_id,
        ) in enumerate(chunk_rows):
            text_unit_records.append(
                {
                    "id": str(chunk_id),
                    "human_readable_id": human_readable_id,
                    "text": content,
                    "n_tokens": len(self.embedding_model.tokenize(content)),
                    "document_id": str(source_id),
                    "entity_ids": chunk_entities_by_id.get(chunk_id, []),
                    "relationship_ids": chunk_relationships_by_id.get(
                        chunk_id,
                        [],
                    ),
                    "covariate_ids": [],
                }
            )

        entity_schema = pa.schema(
            [
                ("id", pa.string()),
                ("human_readable_id", pa.int64()),
                ("title", pa.string()),
                ("type", pa.string()),
                ("description", pa.string()),
                ("text_unit_ids", pa.list_(pa.string())),
                ("frequency", pa.int64()),
                ("degree", pa.int64()),
            ]
        )
        relationship_schema = pa.schema(
            [
                ("id", pa.string()),
                ("human_readable_id", pa.int64()),
                ("source", pa.string()),
                ("target", pa.string()),
                ("description", pa.string()),
                ("weight", pa.float64()),
                ("combined_degree", pa.int64()),
                ("text_unit_ids", pa.list_(pa.string())),
            ]
        )
        text_unit_schema = pa.schema(
            [
                ("id", pa.string()),
                ("human_readable_id", pa.int64()),
                ("text", pa.string()),
                ("n_tokens", pa.int64()),
                ("document_id", pa.string()),
                ("entity_ids", pa.list_(pa.string())),
                ("relationship_ids", pa.list_(pa.string())),
                ("covariate_ids", pa.list_(pa.string())),
            ]
        )

        output_files = {
            "entities": output_path / "entities.parquet",
            "relationships": output_path / "relationships.parquet",
            "text_units": output_path / "text_units.parquet",
        }
        pq.write_table(
            pa.Table.from_pylist(entity_records, schema=entity_schema),
            output_files["entities"],
        )
        pq.write_table(
            pa.Table.from_pylist(
                relationship_records,
                schema=relationship_schema,
            ),
            output_files["relationships"],
        )
        pq.write_table(
            pa.Table.from_pylist(text_unit_records, schema=text_unit_schema),
            output_files["text_units"],
        )

        return {
            table_name: str(file_path)
            for table_name, file_path in output_files.items()
        }
    ############################################################
    #   Functions For Visualuzing DotVDB's Graphs              # 
    ############################################################
    async def view_live_graph(self):
        """Launch the unified database browser with graph exploration available."""
        return await self.open_sql_browser(port=8765)

    async def open_sql_browser(
        self,
        write_mode=False,
        port=8766,
        host="127.0.0.1",
        open_browser=True,
        rag_functions=None,
    ):
        """Launch the unified SQL and graph browser."""
        async def route_chat_message(prompt):
            return await self.language_model.generate(
                prompt,
                structuredInput=ChatbotRetrievalDecision,
            )

        if write_mode and self.conn is not None:
            raise RuntimeError(
                "Close DotVectorDB before opening the SQL browser in write "
                "mode so two writers cannot use the database simultaneously."
            )
        if (
            self._sql_browser_server is not None
            and self._sql_browser_server.conn is not None
        ):
            raise RuntimeError("The SQL browser is already running.")
        self._sql_browser_server = await open_sql_browser(
            database_path=self.path,
            write_mode=write_mode,
            port=port,
            host=host,
            open_browser=open_browser,
            rag_functions=rag_functions,
            generate_function=self.language_model.generate,
            route_function=route_chat_message,
        )
        return self._sql_browser_server

    async def view_sql_database(
        self,
        rag_functions=None,
        write_mode=False,
        port=8766,
        host="127.0.0.1",
        open_browser=True,
    ):
        """Launch the database UI, optionally with named RAG chat functions."""
        return await self.open_sql_browser(
            write_mode=write_mode,
            port=port,
            host=host,
            open_browser=open_browser,
            rag_functions=rag_functions,
        )
