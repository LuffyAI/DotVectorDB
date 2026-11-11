from .UnstructuredGraphConstructor import UnstructuredGraphConstructor


class NaiveVDBConstructor(UnstructuredGraphConstructor):
    async def construct_document_bm25_index(
        self,
        input_document_list=None,
        async_cooldown=60,
        requests_per_minute=1000,
    ) -> None:
        """Ingest documents and build the chunk BM25 index."""
        await self.ingest_documents(
            input_document_list=input_document_list,
            async_cooldown=async_cooldown,
            requests_per_minute=requests_per_minute,
            skip_embeddings=True,
            skip_bm25=False,
        )

    async def construct_document_vectordatabase(
        self,
        input_document_list=None,
        async_cooldown=60,
        requests_per_minute=1000,
    ) -> None:
        """Ingest documents and embed their chunks."""
        await self.ingest_documents(
            input_document_list=input_document_list,
            async_cooldown=async_cooldown,
            requests_per_minute=requests_per_minute,
            skip_embeddings=False,
            skip_bm25=True,
        )

    async def construct_bm25_and_vdb(
        self,
        input_document_list=None,
        async_cooldown=60,
        requests_per_minute=1000,
    ) -> None:
        """Ingest documents, embed their chunks, and build the BM25 index."""
        await self.ingest_documents(
            input_document_list=input_document_list,
            async_cooldown=async_cooldown,
            requests_per_minute=requests_per_minute,
            skip_embeddings=False,
            skip_bm25=False,
        )
