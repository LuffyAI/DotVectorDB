import uuid

class AsyncTempVectorSpace:
    def __init__(self, conn, lock, tids, memory=None, verbose=False, **kwargs):
        self.conn = conn
        self.wid = uuid.uuid4().hex[:8]            
        self.memory = memory  
        self.lock = lock
        self.verbose = verbose
        self.tids = tids

    async def __aenter__(self):
        async with self.lock:
            for tid in self.tids:
                table_name = f"temp_vector_search_{self.wid}_{tid}"
                await self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS temp.{table_name}
                USING vec1(embedding);
                """)
                if self.verbose:
                 print(f"Created temporary vector search tables for tid {tid} with workspace id {self.wid}")
            return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self.lock:
             if self.verbose:
              print(f"Cleaning up temporary vector search tables for workspace id {self.wid}...")

             for tid in self.tids:
                 table_name = f"temp_vector_search_{self.wid}_{tid}"
                 await self.conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
                 if self.verbose:
                  print(f"Dropped temporary vector search tables for tid {tid} with workspace id {self.wid}")
             await self.conn.commit()

    async def copy_and_move_embeddings_to_temp(self, from_table, tid, ids):
       async with self.lock:
                table_name = f"temp_vector_search_{self.wid}_{tid}"
                await self.conn.execute(
                f"""
                INSERT INTO temp.{table_name} (rowid, embedding)
                SELECT rowid, embedding
                FROM {from_table}
                WHERE rowid IN ({','.join(['?'] * len(ids))});
                """,
                ids,
                )
                if self.verbose:
                 print(f"Copied embeddings from {from_table} to temporary table {table_name} for tid {tid}")
          

    async def remove_embeddings_from_temp():
       pass