from .DotVDB import DotVectorDB
from .construction_prompts.chatbot_prompts import ChatbotRetrievalDecision
from .sql_browser import open_sql_browser as launch_sql_browser


class DatabaseViewer(DotVectorDB):
    """Open browser-based views for an existing DotVectorDB database."""

    async def view_live_graph(self):
        """Launch the unified browser with graph exploration selected."""
        return await self.open_sql_browser(port=8765)
    
    async def open_sql_browser(
        self,
        write_mode=False,
        port=8766,
        host="127.0.0.1",
        open_browser=True,
        rag_functions=None,
    ):
        """Launch the unified SQL, graph, and RAG-chat browser."""
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
        self._sql_browser_server = await launch_sql_browser(
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
        """Launch the database UI, optionally with RAG chat functions."""
        return await self.open_sql_browser(
            write_mode=write_mode,
            port=port,
            host=host,
            open_browser=open_browser,
            rag_functions=rag_functions,
        )
    
