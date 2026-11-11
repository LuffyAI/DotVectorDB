from dotvectordb.interfaces.EmbeddingModel import EmbeddingModel
from dotvectordb.interfaces.LanguageModel import LanguageModel
from dotvectordb.interfaces.ChunkingFunc import ChunkingFunc
from dotvectordb.interfaces.HashingFunc import HashingFunc
from dotvectordb.plugins.embedding_providers.openai_embeddings import OpenAIEmbeddingModel
from dotvectordb.plugins.llm_providers.openai_llms import OpenAiAgent
from dotvectordb.plugins.chunk_functions.FixedSizeChunker import FixedSizeChunker
from dotvectordb.plugins.hash_functions.hashfuncs import SHA256HashingFunc
from dotvectordb.construction_prompts.entity_prompts import DEFAULT_ENTITY_TYPES, isEntityTypesFormatCorrect
import os
from pathlib import Path
import json
import aiosqlite
import asyncio
import traceback
import tiktoken
import numpy as np
import subprocess
from itertools import islice
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
        requested_path = Path(path).expanduser()
        if requested_path.suffix:
            database_name = requested_path.stem
            database_filename = requested_path.name
            if requested_path.parent.name == database_name:
                database_dir = requested_path.parent
            else:
                database_dir = requested_path.parent / database_name
        else:
            database_name = requested_path.name
            if not database_name:
                raise ValueError("Database path must include a database name.")
            database_dir = requested_path
            database_filename = f"{database_name}.db"

        self.requested_path = str(requested_path.resolve())
        self.database_dir = database_dir.resolve()
        self.path = str(
            (self.database_dir / database_filename).resolve()
        )
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.chunking_func = chunking_func
        self.hashing_func = hashing_func
        self.verbose = verbose
        self.conn: aiosqlite.Connection | None = None
        self.lock = asyncio.Lock()
        self._query_embedding_tasks = {}
        self._query_embedding_tasks_lock = asyncio.Lock()
        self._sql_browser_server = None
    ############################################################
    #   Private Helper Functions For DotVDB                    # 
    ############################################################
    def _is_vec1_installed(self) -> bool:
            """Helper function to check if the vec1 extension is installed."""
            return (BASE_DIR / "compiled_vec1" / "vec1.so").is_file()
    def _compile_vec1_extension(self):
            """If Vec1 is not installed, compile it using the make command. This requires that make and a C compiler are available on the system."""
            extension_dir = BASE_DIR / "compiled_vec1"
            try:
                subprocess.run(
                    ["make", f"OUTPUT_DIR={extension_dir}", "vec1"],
                    cwd=extension_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Cannot compile Vec1: make or the compiled_vec1 directory is missing."
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile Vec1. Ensure make and a C compiler are installed.\n"
                    f"{exc.stdout}\n{exc.stderr}"
                ) from exc
    def _read_schema_file(self, filename: str) -> str:
            """Helper function to read a schema file and return its contents as a string."""
            file = (BASE_DIR / "schema" / filename).read_text(encoding="utf-8")
            return file
    def _require_open_connection(self) -> None:
        """Ensure that the database connection is open."""
        if not self.conn:
            raise RuntimeError("Database connection is not open. Please call the 'open' method first.")
    def _require_distance_func(self, distance_metric: str) -> str:
            """Helper function to determine the appropriate distance function based on the specified distance metric."""
            if distance_metric == "cos":
                distance_func = "vec1_cos_distance"
            else:
                distance_func = "vec1_l2_distance"
            return distance_func
    def _require_size_positive(self, item, name) -> None:
        """Ensure that a given size parameter is positive."""
        if not isinstance(item, int) or item <= 0:
            raise ValueError(f"{name} must be a positive integer.")  
    def _graph_construction_print(self, message):
            """Print a graph-construction phase heading."""
            print("==================================================================")
            print(message)
            print("==================================================================")
    def _transform_SQL_row_to_dict(self, row, cursor) -> list[dict]:
        """Helper function to transform SQL query results into a list of dictionaries."""
        if row is None:
            return None
        column_names = [description[0] for description in cursor.description]
        return dict(zip(column_names, row))
    ############################################################
    #   Functions For Opening and Closing DotVDB               # 
    ############################################################
    async def __aenter__(self):
        """Upon openin DotVectorDB, we open the database connection and initialize the default schema if it doesn't exist."""
        if self.conn is not None:
            raise RuntimeError("Database connection is already open.")
        try:
            await self._open()
        except BaseException:
            await self._close()
            raise
        return self
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Once the context manager is exited, we close the database connection."""
        await self.close()
        return False
    async def _open(self):
        """Open the database connection, set up the schema, and load the vec1 extension."""
        if (
            self._sql_browser_server is not None
            and self._sql_browser_server.write_mode
            and self._sql_browser_server.conn is not None
        ):
            raise RuntimeError(
                "Close the write-mode SQL browser before reopening "
                "DotVectorDB."
            )
        # Every database owns one directory containing its SQLite file,
        # SQLite side files, graph artifacts, and reachability sidecars.
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.path)
        def setup():
            try:
                # Get the raw connection to set up the database schema
                raw = self.conn._conn
                # Set PRAGMA settings for performance
                pragma_sql = self._read_schema_file("settings.sql")
                raw.executescript(pragma_sql)
                # Loads the compiled vec1 extension for vector operations
                raw.enable_load_extension(True)
                if not self._is_vec1_installed():
                    self._compile_vec1_extension()
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
        await self._initialize_default_schema()
        database_stem_path = Path(self.path).with_suffix("")
        self.graph_path = f"{database_stem_path}_graph.pkl"
        self.DIM = await self.get_embedding_dim()
        if self.verbose:
            print("DotVectorDB initialized successfully.")
        return self
    async def _close(self):
        """Close the database connection if it's open."""
        if self.conn:
            if self.verbose:
                print("Closing database connection...")
            await self.conn.close()
            self.conn = None
        else:
            if self.verbose:
                print("Database connection is already closed.")
    ############################################################
    #  Private Functions For Manipulating DotVDB's Schema      #
    ############################################################
    async def _initialize_default_schema(self) -> None:
        """Initialize the default schema for the database, including tables for chunks, entities, relationships, failure logs, chat history, and query embeddings."""
        try:
            self._require_open_connection()
            chunks_sql = self._read_schema_file("chunk.sql")
            entities_sql = self._read_schema_file("entity.sql")
            relationships_sql = self._read_schema_file("relationship.sql")
            failure_log_sql = self._read_schema_file("failure_log.sql")
            chat_history_sql = self._read_schema_file("chat_history.sql")
            query_embeddings_sql = self._read_schema_file(
                "query_embeddings.sql"
            )
            async with self.lock:
                await self.conn.executescript(chunks_sql)
                await self.conn.executescript(entities_sql)
                await self.conn.executescript(relationships_sql)
                await self.conn.executescript(failure_log_sql)
                await self.conn.executescript(chat_history_sql)
                await self.conn.executescript(query_embeddings_sql)
                await self.conn.commit()
            if self.verbose:
                print("Default schema initialized successfully.")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error in the initialize_default_schema function inside DotVectorDB: {e}")     
    async def add_entity_types(self, entity_types=None)-> None:
        """Add entity types to the database. If no entity types are provided, default entity types will be used."""
        self._require_open_connection()
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
    ############################################################
    #   Accessor Functions                                     # 
    ############################################################
    async def get_entities_that_need_updated_dedupe(self):
        """Return tracked updated entity IDs and their types."""
        async with self.conn.execute(
            """
            SELECT tracked.entity_id, mapping.entity_type
            FROM new_and_updated_entity AS tracked
            LEFT JOIN entity_to_entity_type_mapping AS mapping
                ON mapping.entity_id = tracked.entity_id
            WHERE tracked.is_new = FALSE
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return rows
    async def get_new_entities_before_last_completed_version(self):
        """Return tracked new entity IDs and their types."""
        async with self.conn.execute(
            """
            SELECT tracked.entity_id, mapping.entity_type
            FROM new_and_updated_entity AS tracked
            LEFT JOIN entity_to_entity_type_mapping AS mapping
                ON mapping.entity_id = tracked.entity_id
            WHERE tracked.is_new = TRUE
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return rows
    async def get_new_relationships(self):
        """Return tracked new relationships with their IDs and endpoints."""
        async with self.conn.execute(
            """
            SELECT tracked.relation_id, relationship.src_entity_id, relationship.dst_entity_id
            FROM new_and_updated_relationship AS tracked
            JOIN relationship ON relationship.relation_id = tracked.relation_id
            WHERE tracked.is_new = TRUE
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {"relation_id": row[0], "src_entity_id": row[1], "dst_entity_id": row[2]}
            for row in rows
        ]
    async def get_updated_relationships(self):
        """Return tracked updated relationships with their IDs and endpoints."""
        async with self.conn.execute(
            """
            SELECT tracked.relation_id, relationship.src_entity_id, relationship.dst_entity_id
            FROM new_and_updated_relationship AS tracked
            JOIN relationship ON relationship.relation_id = tracked.relation_id
            WHERE tracked.is_new = FALSE
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {"relation_id": row[0], "src_entity_id": row[1], "dst_entity_id": row[2]}
            for row in rows
        ]
    async def get_entity_metadata(self, entity_id)-> dict:
        """Return metadata for a specific entity by its ID."""
        self._require_open_connection()
        cursor = await self.conn.execute(
            "SELECT * FROM entity WHERE entity_id = ?",
            (entity_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._transform_SQL_row_to_dict(row, cursor)
    async def get_relationship_metadata(self, relationship_id)-> dict:
        """Return metadata for a specific relationship by its ID."""
        self._require_open_connection()
        cursor = await self.conn.execute(
            "SELECT * FROM relationship WHERE relation_id = ?",
            (relationship_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._transform_SQL_row_to_dict(row, cursor)
    async def get_relationship_types(self, relationship_id) -> dict:
        """Return a relationship's endpoint IDs and their entity types."""
        self._require_open_connection()
        cursor = await self.conn.execute(
            """
            SELECT src_entity_id, dst_entity_id
            FROM relationship
            WHERE relation_id = ?
            """,
            (relationship_id,),
        )
        relationship = await cursor.fetchone()
        await cursor.close()
        if relationship is None:
            return None
        src_entity_id, dst_entity_id = relationship
        cursor = await self.conn.execute(
            """
            SELECT entity_id, entity_type
            FROM entity_to_entity_type_mapping
            WHERE entity_id IN (?, ?)
            ORDER BY entity_id, entity_type
            """,
            (src_entity_id, dst_entity_id),
        )
        type_rows = await cursor.fetchall()
        await cursor.close()

        types_by_entity_id = {
            src_entity_id: [],
            dst_entity_id: [],
        }
        for entity_id, entity_type in type_rows:
            types_by_entity_id[entity_id].append(entity_type)

        return {
            "src_entity_id": src_entity_id,
            "src_entity_types": types_by_entity_id[src_entity_id],
            "dst_entity_id": dst_entity_id,
            "dst_entity_types": types_by_entity_id[dst_entity_id],
        }
    async def batch_get_relationship_types(
        self,
        relationship_ids,
        batch_size=10_000,
    ) -> list[dict]:
        """Return endpoint IDs and types for multiple relationships."""
        self._require_open_connection()
        if not isinstance(relationship_ids, list):
            raise TypeError("relationship_ids must be a list")
        if any(
            not isinstance(relationship_id, int)
            or isinstance(relationship_id, bool)
            for relationship_id in relationship_ids
        ):
            raise TypeError("every relationship_id must be an integer")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not relationship_ids:
            return []

        unique_relationship_ids = list(dict.fromkeys(relationship_ids))
        relationships_by_id = {}
        for start in range(0, len(unique_relationship_ids), batch_size):
            batch = unique_relationship_ids[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            cursor = await self.conn.execute(
                f"""
                SELECT relation_id, src_entity_id, dst_entity_id
                FROM relationship
                WHERE relation_id IN ({placeholders})
                """,
                batch,
            )
            for relation_id, src_entity_id, dst_entity_id in (
                await cursor.fetchall()
            ):
                relationships_by_id[relation_id] = (
                    src_entity_id,
                    dst_entity_id,
                )
            await cursor.close()

        endpoint_ids = list(dict.fromkeys(
            entity_id
            for endpoints in relationships_by_id.values()
            for entity_id in endpoints
        ))
        types_by_entity_id = {
            entity_id: [] for entity_id in endpoint_ids
        }
        for start in range(0, len(endpoint_ids), batch_size):
            batch = endpoint_ids[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            cursor = await self.conn.execute(
                f"""
                SELECT entity_id, entity_type
                FROM entity_to_entity_type_mapping
                WHERE entity_id IN ({placeholders})
                ORDER BY entity_id, entity_type
                """,
                batch,
            )
            for entity_id, entity_type in await cursor.fetchall():
                types_by_entity_id[entity_id].append(entity_type)
            await cursor.close()

        results = []
        for relationship_id in unique_relationship_ids:
            endpoints = relationships_by_id.get(relationship_id)
            if endpoints is None:
                continue
            src_entity_id, dst_entity_id = endpoints
            results.append({
                "relation_id": relationship_id,
                "src_entity_id": src_entity_id,
                "src_entity_types": types_by_entity_id[src_entity_id],
                "dst_entity_id": dst_entity_id,
                "dst_entity_types": types_by_entity_id[dst_entity_id],
            })
        return results
    async def get_chunk_metadata(self, chunk_id)-> dict:
        self._require_open_connection()
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT * FROM chunk WHERE chunk_id = ?",
                (chunk_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            return self._transform_SQL_row_to_dict(row, cursor)
    async def get_num_of_entities_and_relationships(self) -> tuple[int, int]:
        """Returns the number of entities and relationships in the database"""
        self._require_open_connection()
        async with self.lock:
            cursor = await self.conn.execute("SELECT num_entities, num_relations FROM global_config WHERE id = 1")
            result = await cursor.fetchone()
            await cursor.close()
            num_entities = result[0] if result[0] is not None else 0
            num_relationships = result[1] if result[1] is not None else 0
        return num_entities, num_relationships
    async def get_entity_types(self, only_populated=False, include_counts=False)-> dict[str, str]:
            """Gets entity types from the database and returns them as a dictionary"""
            self._require_open_connection()
            try: 
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
        self._require_open_connection()
        try:
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
            self._require_open_connection()
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
        self._require_open_connection()
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
    async def is_Database_new_or_old(self) -> bool:
        """Save the current version for later phases; return True if it is zero."""
        global_config = await self.get_global_config()
        self.initial_database_version = global_config["current_version_count"]
        return self.initial_database_version == 0
    async def get_global_config(self) -> dict[str, int | str]:
            """Gets the global configuration metadata from the database"""
            self._require_open_connection()
            try:
                cursor = await self.conn.execute(
                    """
                    SELECT
                        current_version_count,
                        description_version,
                        description,
                        DIM,
                        embedding_model_id,
                        num_chunks,
                        num_entities,
                        num_relations,
                        avg_entity_tokens,
                        avg_relationship_tokens,
                        average_chunk_tokens,
                        total_chunk_tokens,
                        total_entity_tokens,
                        total_relationship_tokens,
                        avg_degree,
                        max_degree,
                        min_degree,
                        bitmap_in_progress,
                        bitmap_phase_checkpoints,
                        bitmap_pending_inputs,
                        entity_bm25_in_progress,
                        relationship_bm25_in_progress,
                        new_entities,
                        updated_entities
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
                    "num_relations": global_config[7],
                    "avg_entity_tokens": global_config[8],
                    "avg_relationship_tokens": global_config[9],
                    "average_chunk_tokens": global_config[10],
                    "total_chunk_tokens": global_config[11],
                    "total_entity_tokens": global_config[12],
                    "total_relationship_tokens": global_config[13],
                    "avg_degree": global_config[14],
                    "max_degree": global_config[15],
                    "min_degree": global_config[16],
                    "bitmap_in_progress": bool(global_config[17]),
                    "bitmap_phase_checkpoints": json.loads(global_config[18]),
                    "bitmap_pending_inputs": json.loads(global_config[19]),
                    "entity_bm25_in_progress": bool(global_config[20]),
                    "relationship_bm25_in_progress": bool(global_config[21]),
                    "new_entities": json.loads(global_config[22]),
                    "updated_entities": json.loads(global_config[23]),
                }
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Error in the get_global_config function inside DotVectorDB: {e}")
    async def get_chunks_associated_with_entity_id(self, entity_id):
        """Given an entity_id, this function retrieves all associated chunks from the database."""
        self._require_open_connection()
        try:
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
    async def batch_get_entity_degrees(self, entity_ids, batch_size=10_000):
        """Return {entity_id: degree} for the supplied set; omit IDs not found."""
        self._require_open_connection()
        if not isinstance(entity_ids, set):
            raise TypeError("entity_ids must be a set")
        self._require_size_positive(batch_size, "batch_size")
        entity_ids = list(entity_ids)
        degrees = {}
        for start in range(0, len(entity_ids), batch_size):
            batch = entity_ids[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            async with self.conn.execute(
                f"SELECT entity_id, degrees FROM entity "
                f"WHERE entity_id IN ({placeholders})",
                batch,
            ) as cursor:
                rows = await cursor.fetchall()
            degrees.update(rows)
        return degrees
    async def batch_entity_embeddings_given_ids(self, entity_ids, batch_size=10_000):
        """Return {entity_id: float32 NumPy embedding}; omit IDs without embeddings."""
        self._require_open_connection()
        if not isinstance(entity_ids, list):
            raise TypeError("entity_ids must be a list")
        if any(
            not isinstance(entity_id, int) or isinstance(entity_id, bool)
            for entity_id in entity_ids
        ):
            raise TypeError("every entity_id must be an integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        unique_entity_ids = list(dict.fromkeys(entity_ids))
        embeddings = {}
        for start in range(0, len(unique_entity_ids), batch_size):
            batch = unique_entity_ids[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            async with self.lock:
                async with self.conn.execute(
                    f"SELECT rowid, embedding FROM entity_embeddings "
                    f"WHERE rowid IN ({placeholders})",
                    batch,
                ) as cursor:
                    rows = await cursor.fetchall()
            for entity_id, embedding in rows:
                embeddings[entity_id] = np.frombuffer(
                    embedding, dtype=np.float32
                ).copy()
        return embeddings
    async def get_chunks_associated_with_entity_ids(self, entity_ids):
        """Return the unique chunks associated with any supplied entity ID."""
        self._require_open_connection()
        if not isinstance(entity_ids, list):
            raise TypeError("entity_ids must be a list")
        if any(
            not isinstance(entity_id, int) or isinstance(entity_id, bool)
            for entity_id in entity_ids
        ):
            raise TypeError("every entity_id must be an integer")
        if not entity_ids:
            return []
        unique_entity_ids = list(dict.fromkeys(entity_ids))
        chunks_by_id = {}
        batch_size = 10_000
        try:
            for start in range(0, len(unique_entity_ids), batch_size):
                batch = unique_entity_ids[start:start + batch_size]
                placeholders = ", ".join("?" for _ in batch)
                cursor = await self.conn.execute(
                    f"""
                    SELECT DISTINCT chunk.*
                    FROM chunk
                    JOIN chunk_to_entity_mapping AS mapping
                        ON chunk.chunk_id = mapping.chunk_id
                    WHERE mapping.entity_id IN ({placeholders})
                    ORDER BY chunk.chunk_id
                    """,
                    batch,
                )
                rows = await cursor.fetchall()
                column_names = [
                    description[0] for description in cursor.description
                ]
                await cursor.close()
                for row in rows:
                    chunk = dict(zip(column_names, row))
                    chunks_by_id[chunk["chunk_id"]] = chunk
            return [
                chunks_by_id[chunk_id]
                for chunk_id in sorted(chunks_by_id)
            ]
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                "Error in the get_chunks_associated_with_entity_ids "
                f"function inside DotVectorDB: {e}"
            ) from e
    async def get_chunks_associated_with_relationship_ids(
        self,
        relationship_ids,
    ):
        """Return the unique chunks associated with any relationship ID."""
        self._require_open_connection()
        if not isinstance(relationship_ids, list):
            raise TypeError("relationship_ids must be a list")
        if any(
            not isinstance(relationship_id, int)
            or isinstance(relationship_id, bool)
            for relationship_id in relationship_ids
        ):
            raise TypeError("every relationship_id must be an integer")
        if not relationship_ids:
            return []
        unique_relationship_ids = list(dict.fromkeys(relationship_ids))
        chunks_by_id = {}
        batch_size = 10_000
        try:
            for start in range(0, len(unique_relationship_ids), batch_size):
                batch = unique_relationship_ids[start:start + batch_size]
                placeholders = ", ".join("?" for _ in batch)
                cursor = await self.conn.execute(
                    f"""
                    SELECT DISTINCT chunk.*
                    FROM chunk
                    JOIN chunk_to_relationship_mapping AS mapping
                        ON chunk.chunk_id = mapping.chunk_id
                    WHERE mapping.relation_id IN ({placeholders})
                    ORDER BY chunk.chunk_id
                    """,
                    batch,
                )
                rows = await cursor.fetchall()
                column_names = [
                    description[0] for description in cursor.description
                ]
                await cursor.close()
                for row in rows:
                    chunk = dict(zip(column_names, row))
                    chunks_by_id[chunk["chunk_id"]] = chunk
            return [
                chunks_by_id[chunk_id]
                for chunk_id in sorted(chunks_by_id)
            ]
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                "Error in the get_chunks_associated_with_relationship_ids "
                f"function inside DotVectorDB: {e}"
            ) from e
    async def get_chunks_associated_with_two_entities(self, entity_id_one, entity_id_two):
        """Given two entity IDs, this function retrieves all associated chunks they have in common."""
        self._require_open_connection()
        try:
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
    async def get_entities_and_rels_associated_with_chunk_id(self, chunk_id):
        """Return all entities and relationships mapped to a chunk."""
        self._require_open_connection()
        try:
            entity_cursor = await self.conn.execute(
                """
                SELECT entity.*
                FROM entity
                JOIN chunk_to_entity_mapping
                    ON entity.entity_id = chunk_to_entity_mapping.entity_id
                WHERE chunk_to_entity_mapping.chunk_id = ?
                """,
                (chunk_id,),
            )
            entity_rows = await entity_cursor.fetchall()
            entity_columns = [
                description[0] for description in entity_cursor.description
            ]
            await entity_cursor.close()

            relationship_cursor = await self.conn.execute(
                """
                SELECT relationship.*
                FROM relationship
                JOIN chunk_to_relationship_mapping
                    ON relationship.relation_id =
                        chunk_to_relationship_mapping.relation_id
                WHERE chunk_to_relationship_mapping.chunk_id = ?
                """,
                (chunk_id,),
            )
            relationship_rows = await relationship_cursor.fetchall()
            relationship_columns = [
                description[0]
                for description in relationship_cursor.description
            ]
            await relationship_cursor.close()

            return {
                "entities": [
                    dict(zip(entity_columns, row)) for row in entity_rows
                ],
                "relationships": [
                    dict(zip(relationship_columns, row))
                    for row in relationship_rows
                ],
            }
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                "Error in the get_entities_and_rels_associated_with_chunk_id "
                f"function inside DotVectorDB: {e}"
            ) from e
    ############################################################
    #   Functions For Adding Text Content to the Database      # 
    ############################################################
    async def set_embedding_dim(self) -> int:
        self._require_open_connection()
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
                self._require_open_connection()
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
        self._require_open_connection()
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
    async def add_chunk(
        self,
        content,
        source_name,
        model_name,
        tokens,
        numbered_line_dict
    ) -> None:
        self._require_open_connection()
        if tokens is None:
            tokens = len(self.chunking_func.tokenize(content))
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise TypeError("Chunk tokens must be an integer.")
        if tokens < 0:
            raise ValueError("Chunk tokens cannot be negative.")

        content_hash = self.hashing_func.content_hash(content)
        sentence_rows = [
            (content_hash, sentence_number, sentence_text)
            for sentence_number, sentence_text in numbered_line_dict.items()
        ]
        source_hash = f"""{self.hashing_func.content_hash(source_name)}"""
        model_hash = f"""{self.hashing_func.content_hash(model_name)}"""
        await self.add_metadata_content(table="source", name=source_name)
        await self.add_metadata_content(table="embedding_model", name=model_name)
        async with self.lock:
            try:
                await self.conn.execute(
                    """
                    INSERT INTO chunk (
                        chunk_id,
                        source_id,
                        content,
                        embedding_model_id,
                        tokens
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        tokens = excluded.tokens
                    """,
                    (
                        content_hash,
                        source_hash,
                        content,
                        model_hash,
                        tokens,
                    )
                )
                await self.conn.executemany(
                    """
                    INSERT INTO chunk_to_sentence_mapping (
                        chunk_id, sentence_number, sentence_text
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id, sentence_number) DO UPDATE SET
                        sentence_text = excluded.sentence_text
                    """,
                    sentence_rows,
                )
                await self.conn.commit()
            except Exception as e:
                await self.conn.rollback()
                traceback.print_exc()
                raise RuntimeError(f"Error while adding a chunk: {e}") from e

    async def add_chunk_batch(self, chunks, source_name, model_name, batch_size=10_000) -> None:
        """Insert chunker output and numbered lines in transactions of up to batch_size chunks."""
        self._require_open_connection()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        source_hash = str(self.hashing_func.content_hash(source_name))
        model_hash = str(self.hashing_func.content_hash(model_name))
        remaining = iter(chunks)
        while batch := list(islice(remaining, batch_size)):
            chunk_rows = []
            sentence_rows = []
            for chunk in batch:
                content = chunk["content"]
                tokens = chunk["tokens"]
                if tokens is None:
                    tokens = len(self.chunking_func.tokenize(content))
                if isinstance(tokens, bool) or not isinstance(tokens, int):
                    raise TypeError("Chunk tokens must be an integer.")
                if tokens < 0:
                    raise ValueError("Chunk tokens cannot be negative.")
                content_hash = self.hashing_func.content_hash(content)
                chunk_rows.append((content_hash, source_hash, content, model_hash, tokens))
                sentence_rows.extend(
                    (content_hash, number, text)
                    for number, text in chunk["numbered_lines"].items()
                )
            async with self.lock:
                try:
                    await self.conn.execute(
                        "INSERT OR IGNORE INTO source(id, name) VALUES (?, ?)",
                        (source_hash, source_name),
                    )
                    await self.conn.execute(
                        "INSERT OR IGNORE INTO embedding_model(id, name) VALUES (?, ?)",
                        (model_hash, model_name),
                    )
                    await self.conn.executemany(
                        """
                        INSERT INTO chunk(chunk_id, source_id, content, embedding_model_id, tokens)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET tokens = excluded.tokens
                        """,
                        chunk_rows,
                    )
                    await self.conn.executemany(
                        """
                        INSERT INTO chunk_to_sentence_mapping(chunk_id, sentence_number, sentence_text)
                        VALUES (?, ?, ?)
                        ON CONFLICT(chunk_id, sentence_number) DO UPDATE SET
                            sentence_text = excluded.sentence_text
                        """,
                        sentence_rows,
                    )
                    await self.conn.commit()
                except BaseException:
                    await self.conn.rollback()
                    raise
    #####################################################################
    # Function For a Full Construction (Corpus-> Chunks -> Full Graph)  # 
    #####################################################################
    async def count_llm_tokens(self, stage):
                self._require_open_connection()
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
    #   Functions For DotVDB's Vector Ingestion + Search       # 
    ############################################################
    async def add_embedding(self, table, content, pk_hash, status="new") -> None:
                # Check if the database connection is open
                self._require_open_connection()
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
                CREATE_EMBEDDING_DELETE_TRIGGER_SQL = f"""
                CREATE TRIGGER IF NOT EXISTS delete_{table}_embedding
                AFTER DELETE ON {table}
                BEGIN
                    DELETE FROM {table}_embeddings WHERE rowid = OLD.rowid;
                END;
                """
                INSERT_INTO_EMBEDDING_TABLE_SQL = f"""
                INSERT INTO {table}_embeddings(rowid, embedding)
                VALUES (?, ?)
                """

                UPDATE_EMBEDDING_TABLE_SQL = f"""
                UPDATE {table}_embeddings
                SET embedding = ?
                WHERE rowid = ?
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
                    await self.conn.execute(CREATE_EMBEDDING_DELETE_TRIGGER_SQL)
                    # Third, we insert the embedding into the embedding table
                    try:
                        if status == "new":
                            await self.conn.execute(INSERT_INTO_EMBEDDING_TABLE_SQL, (pk_hash, embedding_blob))
                        elif status == "updated":
                            await self.conn.execute(UPDATE_EMBEDDING_TABLE_SQL, (embedding_blob, pk_hash))
                        else:
                            raise RuntimeError(f"Invalid status '{status}' provided to add_embedding. Must be 'new' or 'updated'.")
                        await self.conn.execute(UPDATE_TABLE_STATUS, (pk_hash,))
                        await self.conn.commit()
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(f"Error in the add_embedding function inside DotVectorDB: {e}")
    async def _get_cached_query_embedding(
        self,
        query,
        query_hash,
        embedding_model_id,
    ):
        """Return a cached query embedding, or None when no valid row exists."""
        async with self.conn.execute(
            """
            SELECT embedding, dimensions
            FROM query_embeddings
            WHERE query_hash = ?
              AND embedding_model_id = ?
              AND query = ?
            """,
            (query_hash, embedding_model_id, query),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None

        embedding = np.frombuffer(row[0], dtype=np.float32).copy()
        stored_dimensions = row[1]
        if (
            embedding.shape[-1] != stored_dimensions
            or embedding.shape[-1] != self.DIM
        ):
            return None
        return embedding

    async def _create_and_cache_query_embedding(
        self,
        query,
        query_hash,
        embedding_model_id,
    ):
        """Embed one cache miss and persist the result."""
        cached_embedding = await self._get_cached_query_embedding(
            query=query,
            query_hash=query_hash,
            embedding_model_id=embedding_model_id,
        )
        if cached_embedding is not None:
            return cached_embedding, 0

        embedding, query_tokens = await self.embedding_model.embed(query)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape[-1] != self.DIM:
            raise RuntimeError(
                "Query embedding dimension mismatch: "
                f"expected {self.DIM}, got {embedding.shape[-1]}"
            )

        async with self.lock:
            await self.conn.execute(
                """
                INSERT OR REPLACE INTO query_embeddings (
                    query_hash,
                    embedding_model_id,
                    query,
                    embedding,
                    dimensions,
                    tokens
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_hash,
                    embedding_model_id,
                    query,
                    embedding.tobytes(),
                    embedding.shape[-1],
                    query_tokens,
                ),
            )
            await self.conn.commit()
        return embedding, query_tokens

    async def _get_or_create_query_embedding(self, query):
        """Reuse an exact query embedding and coalesce concurrent cache misses."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query_hash = self.hashing_func.content_hash(query)
        embedding_model_id = str(
            self.hashing_func.content_hash(self.embedding_model.model_name)
        )
        cached_embedding = await self._get_cached_query_embedding(
            query=query,
            query_hash=query_hash,
            embedding_model_id=embedding_model_id,
        )
        if cached_embedding is not None:
            return cached_embedding, 0

        cache_key = (query_hash, embedding_model_id, query)
        created_task = False
        async with self._query_embedding_tasks_lock:
            task = self._query_embedding_tasks.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._create_and_cache_query_embedding(
                        query=query,
                        query_hash=query_hash,
                        embedding_model_id=embedding_model_id,
                    )
                )
                self._query_embedding_tasks[cache_key] = task
                created_task = True

        try:
            embedding, query_tokens = await task
            if not created_task:
                query_tokens = 0
            return embedding, query_tokens
        finally:
            async with self._query_embedding_tasks_lock:
                if self._query_embedding_tasks.get(cache_key) is task:
                    del self._query_embedding_tasks[cache_key]

    
