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
        async with self.lock:
             if self.verbose:
              print(f"Cleaning up temporary vector search tables for workspace id {self.wid}...")
             for DOT in self.DOTs:
                    vector_store_query = DOT.close_temp_vector_space_sql_query(self.wid)
                    await self.conn.execute(vector_store_query)
                    if self.verbose:
                     print(f"Dropped temporary vector search tables for tid {DOT.centroid_id} with workspace id {self.wid}")
             await self.conn.commit()
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
            

            