import asyncio
import inspect
import json
import re
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import aiosqlite
from aiohttp import web

from dotvectordb.web_interface.server import GraphInterfaceServer
from dotvectordb.prompts.chatbot_prompts import (
    CHATBOT_RESPONSE_PROMPT,
    CHATBOT_ROUTING_PROMPT,
)


class SQLBrowserServer:
    """Small browser-based SQLite inspector for a DotVectorDB database."""

    def __init__(
        self,
        database_path,
        write_mode=False,
        host="127.0.0.1",
        port=8766,
        rag_functions=None,
        generate_function=None,
        route_function=None,
    ):
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.exists():
            raise FileNotFoundError(
                f"SQLite database does not exist: {self.database_path}"
            )
        self.write_mode = bool(write_mode)
        self.host = host
        self.port = port
        self.rag_functions = self._normalize_rag_functions(rag_functions)
        if generate_function is not None and not callable(generate_function):
            raise TypeError("generate_function must be callable")
        self.generate_function = generate_function
        if route_function is not None and not callable(route_function):
            raise TypeError("route_function must be callable")
        self.route_function = route_function
        self.conn = None
        self._history_conn = None
        self._lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        self._runner = None
        self._site = None
        self._graph_interface = None

    @staticmethod
    def _normalize_rag_functions(rag_functions):
        if rag_functions is None:
            return {}
        if callable(rag_functions):
            name = getattr(rag_functions, "__name__", "RAG")
            return {name: rag_functions}
        if not isinstance(rag_functions, dict):
            raise TypeError("rag_functions must be a callable or a name-to-callable dictionary")
        invalid = [name for name, function in rag_functions.items() if not isinstance(name, str) or not callable(function)]
        if invalid:
            raise TypeError("Every rag_functions entry must map a string name to a callable")
        return dict(rag_functions)

    async def _connect(self):
        if self.write_mode:
            self.conn = await aiosqlite.connect(str(self.database_path))
        else:
            encoded_path = quote(str(self.database_path), safe="/")
            self.conn = await aiosqlite.connect(
                f"file:{encoded_path}?mode=ro",
                uri=True,
            )
            await self.conn.execute("PRAGMA query_only = ON")
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA busy_timeout = 5000")
        self._history_conn = await aiosqlite.connect(str(self.database_path))
        self._history_conn.row_factory = aiosqlite.Row
        await self._history_conn.execute("PRAGMA busy_timeout = 5000")
        await self._history_conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                rag_function TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS chat_history_created_at_index
            ON chat_history (created_at, message_id);
            """
        )
        await self._history_conn.commit()

    @staticmethod
    def _quote_identifier(identifier):
        return '"' + identifier.replace('"', '""') + '"'

    async def _object_exists(self, name):
        cursor = await self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE name = ? AND type IN ('table', 'view')
            """,
            (name,),
        )
        exists = await cursor.fetchone() is not None
        await cursor.close()
        return exists

    @staticmethod
    def _json_value(value):
        if isinstance(value, bytes):
            return {"type": "blob", "size": len(value)}
        return value

    async def _status(self, request):
        return web.json_response(
            {
                "database": str(self.database_path),
                "mode": "write" if self.write_mode else "read-only",
            }
        )

    async def _tables(self, request):
        async with self._lock:
            cursor = await self.conn.execute(
                """
                SELECT name, type
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                  AND name NOT LIKE 'sqlite_%'
                  AND LOWER(name) NOT LIKE '%embedding%'
                ORDER BY type, name
                """
            )
            objects = [dict(row) for row in await cursor.fetchall()]
            await cursor.close()
        return web.json_response({"objects": objects})

    async def _table(self, request):
        name = request.match_info["name"]
        limit = min(max(int(request.query.get("limit", 100)), 1), 500)
        offset = max(int(request.query.get("offset", 0)), 0)
        search = request.query.get("search", "").strip()

        async with self._lock:
            if not await self._object_exists(name):
                raise web.HTTPNotFound(text="Table or view not found")
            quoted_name = self._quote_identifier(name)
            schema_cursor = await self.conn.execute(
                f"PRAGMA table_info({quoted_name})"
            )
            schema = [dict(row) for row in await schema_cursor.fetchall()]
            await schema_cursor.close()

            searchable_columns = [column["name"] for column in schema]
            where_sql = ""
            parameters = []
            if search and searchable_columns:
                where_sql = " WHERE " + " OR ".join(
                    f"CAST({self._quote_identifier(column)} AS TEXT) LIKE ?"
                    for column in searchable_columns
                )
                parameters = [f"%{search}%"] * len(searchable_columns)

            count_cursor = await self.conn.execute(
                f"SELECT COUNT(*) AS count FROM {quoted_name}{where_sql}",
                parameters,
            )
            total = (await count_cursor.fetchone())["count"]
            await count_cursor.close()

            row_cursor = await self.conn.execute(
                f"SELECT * FROM {quoted_name}{where_sql} LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            )
            columns = [description[0] for description in row_cursor.description]
            rows = [
                [self._json_value(value) for value in row]
                for row in await row_cursor.fetchall()
            ]
            await row_cursor.close()

        return web.json_response(
            {
                "name": name,
                "columns": columns,
                "rows": rows,
                "schema": schema,
                "total": total,
                "limit": limit,
                "offset": offset,
                "search": search,
            }
        )

    @staticmethod
    def _is_single_statement(sql):
        stripped = sql.strip()
        if not stripped:
            return False
        without_final_semicolon = stripped[:-1] if stripped.endswith(";") else stripped
        return ";" not in without_final_semicolon

    async def _query(self, request):
        payload = await request.json()
        sql = payload.get("sql", "").strip()
        if not self._is_single_statement(sql):
            raise web.HTTPBadRequest(text="Submit exactly one SQL statement.")

        if not self.write_mode:
            first_word_match = re.match(r"^\s*([A-Za-z]+)", sql)
            first_word = first_word_match.group(1).upper() if first_word_match else ""
            if first_word not in {"SELECT", "WITH", "EXPLAIN"}:
                raise web.HTTPForbidden(
                    text="Read-only mode accepts SELECT, WITH, and EXPLAIN statements only."
                )

        try:
            async with self._lock:
                cursor = await self.conn.execute(sql)
                if cursor.description:
                    columns = [item[0] for item in cursor.description]
                    fetched_rows = await cursor.fetchmany(501)
                    truncated = len(fetched_rows) > 500
                    rows = [
                        [self._json_value(value) for value in row]
                        for row in fetched_rows[:500]
                    ]
                    affected = None
                else:
                    columns = []
                    rows = []
                    truncated = False
                    affected = cursor.rowcount
                    if self.write_mode:
                        await self.conn.commit()
                await cursor.close()
        except Exception as error:
            if self.write_mode:
                await self.conn.rollback()
            raise web.HTTPBadRequest(text=str(error))

        return web.json_response(
            {
                "columns": columns,
                "rows": rows,
                "truncated": truncated,
                "affected": affected,
            }
        )

    async def _home(self, request):
        return web.FileResponse(Path(__file__).with_name("home.html"))

    async def _index(self, request):
        return web.FileResponse(Path(__file__).with_name("index.html"))

    async def _graph_index(self, request):
        return web.FileResponse(
            Path(__file__).parent.parent / "web_interface" / "index.html"
        )

    async def _chat_index(self, request):
        return web.FileResponse(Path(__file__).with_name("chat.html"))

    async def _michigan_logo(self, request):
        return web.FileResponse(
            Path(__file__).with_name("university-of-michigan-vector-logo.svg")
        )

    async def _michigan_logo_white(self, request):
        return web.FileResponse(
            Path(__file__).with_name(
                "university-of-michigan-vector-logo-white.svg"
            )
        )

    async def _chat_status(self, request):
        return web.json_response(
            {
                "functions": list(self.rag_functions),
                "generation_enabled": self.generate_function is not None,
            }
        )

    async def _chat_history(self, request):
        async with self._history_lock:
            cursor = await self._history_conn.execute(
                """
                SELECT message_id, role, content, rag_function, created_at
                FROM chat_history
                ORDER BY message_id
                """
            )
            messages = [dict(row) for row in await cursor.fetchall()]
            await cursor.close()
        return web.json_response({"messages": messages})

    async def _clear_chat_history(self, request):
        async with self._history_lock:
            await self._history_conn.execute("DELETE FROM chat_history")
            await self._history_conn.commit()
        return web.json_response({"cleared": True})

    @staticmethod
    def _chat_answer_value(value):
        if isinstance(value, str):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        if isinstance(value, dict) and set(value) == {"answer"}:
            return str(value["answer"])
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            return str(value)

    async def _call_chat_function(self, function, value):
        result = function(value)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, tuple):
            result = result[0]
        return result

    @staticmethod
    def _conversation_transcript(history):
        return "\n".join(
            f"{item['role'].capitalize()}: {item['content']}"
            for item in history
        )

    async def _chat(self, request):
        payload = await request.json()
        message = payload.get("message", "").strip()
        function_name = payload.get("function", "__none__")
        if not message:
            raise web.HTTPBadRequest(text="Message cannot be empty.")
        if self.generate_function is None:
            raise web.HTTPServiceUnavailable(
                text="No language-model generate function is configured."
            )
        if function_name != "__none__" and function_name not in self.rag_functions:
            raise web.HTTPBadRequest(text="Select a configured RAG function.")
        try:
            async with self._history_lock:
                cursor = await self._history_conn.execute(
                    """
                    SELECT role, content
                    FROM chat_history
                    ORDER BY message_id
                    """
                )
                history = [dict(row) for row in await cursor.fetchall()]
                await cursor.close()

            transcript = self._conversation_transcript(history)
            retrieval_context = ""
            needs_retrieval = False
            retrieval_performed = False
            retrieval_query = None
            if function_name != "__none__":
                routing_prompt = CHATBOT_ROUTING_PROMPT.format(
                    conversation_history=(
                        transcript or "(no previous messages)"
                    ),
                    user_message=message,
                )
                router = self.route_function or self.generate_function
                decision = await self._call_chat_function(
                    router,
                    routing_prompt,
                )
                if hasattr(decision, "model_dump"):
                    decision = decision.model_dump()
                elif hasattr(decision, "dict"):
                    decision = decision.dict()

                if isinstance(decision, dict):
                    needs_retrieval = bool(decision.get("needs_retrieval"))
                    retrieval_query = decision.get("search_query")
                else:
                    needs_retrieval = True
                    retrieval_query = str(decision).strip()

                if needs_retrieval and retrieval_query:
                    retrieval_result = await self._call_chat_function(
                        self.rag_functions[function_name],
                        retrieval_query,
                    )
                    retrieval_context = self._chat_answer_value(
                        retrieval_result
                    )
                    retrieval_performed = True

            final_prompt = CHATBOT_RESPONSE_PROMPT.format(
                conversation_history=(transcript or "(no previous messages)"),
                user_message=message,
                retrieved_context=(
                    retrieval_context or "(no retrieval used)"
                ),
            )
            result = await self._call_chat_function(
                self.generate_function,
                final_prompt,
            )
        except Exception as error:
            raise web.HTTPInternalServerError(text=str(error))
        answer = self._chat_answer_value(result)
        async with self._history_lock:
            await self._history_conn.executemany(
                """
                INSERT INTO chat_history (role, content, rag_function)
                VALUES (?, ?, ?)
                """,
                (
                    ("user", message, None if function_name == "__none__" else function_name),
                    ("assistant", answer, None if function_name == "__none__" else function_name),
                ),
            )
            await self._history_conn.commit()
        return web.json_response(
            {
                "answer": answer,
                "needs_retrieval": needs_retrieval,
                "retrieval_performed": retrieval_performed,
                "retrieval_query": retrieval_query,
                "rag_function": (
                    None if function_name == "__none__" else function_name
                ),
            }
        )

    async def start(self, open_browser=True):
        await self._connect()
        graph_database = SimpleNamespace(
            conn=self.conn,
            lock=self._lock,
            path=self.database_path,
        )
        self._graph_interface = GraphInterfaceServer(graph_database)
        app = web.Application()
        app.router.add_get("/", self._home)
        app.router.add_get("/sql", self._index)
        app.router.add_get("/graph", self._graph_index)
        app.router.add_get("/chat", self._chat_index)
        app.router.add_get("/assets/michigan-logo.svg", self._michigan_logo)
        app.router.add_get(
            "/assets/michigan-logo-white.svg",
            self._michigan_logo_white,
        )
        app.router.add_get("/api/status", self._status)
        app.router.add_get("/api/tables", self._tables)
        app.router.add_get("/api/table/{name}", self._table)
        app.router.add_post("/api/query", self._query)
        app.router.add_get("/api/chat/status", self._chat_status)
        app.router.add_post("/api/chat", self._chat)
        app.router.add_get("/api/chat/history", self._chat_history)
        app.router.add_delete("/api/chat/history", self._clear_chat_history)
        app.router.add_get("/graph-api/status", self._graph_interface._status)
        app.router.add_get("/graph-api/graph", self._graph_interface._graph_data)
        app.router.add_get(
            "/graph-api/neighbors/{entity_id}",
            self._graph_interface._neighbors,
        )
        app.router.add_get(
            "/graph-api/subgraph/{entity_id}",
            self._graph_interface._subgraph,
        )
        app.router.add_get(
            "/graph-api/shortest-paths/{entity_id}",
            self._graph_interface._shortest_paths,
        )

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        url = f"http://{self.host}:{self.port}"
        print(
            f"DotVectorDB browser ({'write' if self.write_mode else 'read-only'}): {url}"
        )
        if open_browser:
            await asyncio.to_thread(webbrowser.open, url)
        return self

    async def close(self):
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        if self.conn is not None:
            await self.conn.close()
            self.conn = None
        if self._history_conn is not None:
            await self._history_conn.close()
            self._history_conn = None


async def open_sql_browser(
    database_path,
    write_mode=False,
    port=8766,
    host="127.0.0.1",
    open_browser=True,
    rag_functions=None,
    generate_function=None,
    route_function=None,
):
    server = SQLBrowserServer(
        database_path=database_path,
        write_mode=write_mode,
        host=host,
        port=port,
        rag_functions=rag_functions,
        generate_function=generate_function,
        route_function=route_function,
    )
    return await server.start(open_browser=open_browser)
