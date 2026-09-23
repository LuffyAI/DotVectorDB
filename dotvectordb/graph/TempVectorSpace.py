import uuid
class AsyncTempVectorSpace:
    def __init__(self, query, conn, lock, DOTs, conn_check_func, verbose=False, **kwargs):
        self.query = query
        self.conn = conn
        self.wid = uuid.uuid4().hex[:8]            
        self.DOTs = DOTs
        self.lock = lock
        self.conn_check_func = conn_check_func
        self.verbose = verbose
    ##############################################################################
    #   Functions that prepares and closes temporary vector stores               # 
    ##############################################################################
    async def __aenter__(self):
        async with self.lock:
            self.conn_check_func()
            for DOT in self.DOTs:
                vector_store_query = DOT.prepare_temp_vector_space_sql_query(self.wid)
                await self.conn.execute(vector_store_query)
                await self._copy_and_move_embeddings_to_temp(DOT)
                if self.verbose:
                 print(f"Created temporary vector search tables for tid {DOT.centroid_id} with workspace id {self.wid}")
                await self.conn.commit()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    ############################################################
    #   Helper Functions For Temp Vector Spaces                # 
    ############################################################
    async def _copy_and_move_embeddings_to_temp(self, DOT, from_table="relationship_embeddings"):
                await self.conn.execute(
                f"""
                INSERT INTO temp.{DOT.sql_table_name}_{self.wid} (rowid, embedding)
                SELECT rowid, embedding
                FROM {from_table}
                WHERE rowid IN ({','.join(['?'] * len(DOT.vector_indices))});
                """,
                DOT.vector_indices,
                )
                if self.verbose:
                 print(f"Copied embeddings from {from_table} to temporary table {DOT.sql_table_name} for tid {DOT.centroid_id}")
    async def remove_embeddings_from_temp_store(self, DOT, indices_to_remove_list):
       async with self.lock:
            await self.conn.execute(
                f"""
                DELETE FROM temp.{DOT.sql_table_name}_{self.wid}
                WHERE rowid IN ({','.join(['?'] * len(indices_to_remove_list))});
                """,
                indices_to_remove_list,
                )
            if self.conn:
                print(f"Removed embeddings from temporary table {DOT.sql_table_name} for tid {DOT.centroid_id}")


class TempVectorSpacePathWeights:
    def __init__(
        self,
        conn,
        lock,
        relationship_ids,
        conn_check_func,
        query=None,
        verbose=False,
        **kwargs,
    ):
        self.conn = conn
        self.lock = lock
        self.relationship_ids = list(dict.fromkeys(relationship_ids))
        self.conn_check_func = conn_check_func
        self.query = query
        self.verbose = verbose
        self.wid = uuid.uuid4().hex[:8]
        self.sql_table_name = f"temp_vector_path_weights_{self.wid}"
        self.embedding_table = f"temp.{self.sql_table_name}"

    async def __aenter__(self):
        async with self.lock:
            self.conn_check_func()
            await self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.embedding_table}
                USING vec1(embedding);
                """
            )
            await self._copy_relationship_embeddings_to_temp()
            await self.conn.commit()
            if self.verbose:
                print(
                    f"Created temporary path-weight vector table "
                    f"{self.sql_table_name} with "
                    f"{len(self.relationship_ids)} relationship IDs"
                )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def _copy_relationship_embeddings_to_temp(
        self,
        from_table="relationship_embeddings",
    ):
        if not self.relationship_ids:
            return

        placeholders = ",".join("?" for _ in self.relationship_ids)
        await self.conn.execute(
            f"""
            INSERT INTO {self.embedding_table} (rowid, embedding)
            SELECT rowid, embedding
            FROM {from_table}
            WHERE rowid IN ({placeholders});
            """,
            self.relationship_ids,
        )
