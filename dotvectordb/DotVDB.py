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
from dotvectordb.graph.dedupe_indexer import DedupeIndexer
from dotvectordb.graph.relatonship_indexer import RelationshipIndexer
from dotvectordb.graph.relationship_indexer2 import RelationshipIndexer2
from dotvectordb.graph.type_indexer import TypeIndexer
from dotvectordb.graph.graph_retriever import GraphRetriever
from dotvectordb.graph.generate_global_desc import GlobalDescriptionGenerator
from dotvectordb.graph.TempVectorSpace import AsyncTempVectorSpace
from dotvectordb.web_interface import open_graph_interface
import os
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import igraph
import json
import aiosqlite
import asyncio
import traceback
import tiktoken
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
                                              get_num_nodes_edges_func=self.get_num_of_entities_and_relationships)
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
            async with self.lock:
                await self.conn.executescript(chunks_sql)
                await self.conn.executescript(entities_sql)
                await self.conn.executescript(relationships_sql)
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

    async def extract_and_embed_entities3(self, entity_types=None, async_cooldown=60, requests_per_minute=1000) -> None:
        self.__require_open_connection()
        try:
            await self.add_entity_types(entity_types)
            entity_types = await self.get_entity_types()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities3 function inside DotVectorDB while adding entity types: {e}")

        try:
            entity_indexer = EntityIndexer3(
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
            await entity_indexer.embed_entities(
                async_cooldown=async_cooldown,
                requests_per_minute=requests_per_minute,
                verbose=self.verbose
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the extract_and_embed_entities3 function inside DotVectorDB: {e}")
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
            # Lastly, we need to generate the graph representation of the database.
            try:
                self.__graph_construction_print("STEP FIVE: Creating the Graph Representation of the Database")
                await self.graph_retriever.load_graph()
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the construct_GraphRAG2 function inside DotVectorDB while clustering entities: {e}")
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
                        if embedding_blob.shape[-1] != self.DIM:
                            raise RuntimeError(f"Embedding dimension mismatch: expected {self.DIM}, got {embedding_blob.shape[-1]}")
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
    async def subgraph_constrained_semantic_retrieval(self, query, custom_search_algo, DOTs, **kwargs):
            """
            In this code, you would open TempVectorSpace() and close tempvector space by the end of the function.
            It essentially creates a temporary vector space that is opened annd closed by the end of this function.
            between the two functions will be the custom function you want to run inside the temmp vector space. it should return whatever results you wan
            """
            self.__require_open_connection()
            gathered_results = None
            async with AsyncTempVectorSpace(query=query, conn=self.conn,DOTs=DOTs, conn_check_func=self.__require_open_connection, lock=self.lock) as QueryWS:
                # The custom_search_algo should be some DOT-based retrieval algorithm
                # The search is performed concurrently for each DOT
                gathered_results = await asyncio.gather(
                *(
                    custom_search_algo(QueryWS, DOT, **kwargs)
                    for DOT in DOTs
                )
            )
            return gathered_results
    async def view_live_graph(self):
        await open_graph_interface(self)
