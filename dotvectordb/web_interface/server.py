import asyncio
import json
import webbrowser
from pathlib import Path

from aiohttp import web
import igraph as ig


class GraphInterfaceServer:
    def __init__(self, dotvdb, host="127.0.0.1", port=8765):
        if dotvdb.conn is None:
            raise ValueError("DotVectorDB must be open before starting the graph interface")
        self.dotvdb = dotvdb
        self.host = host
        self.port = port
        self._runner = None
        self._site = None
        self._igraph = None
        self._igraph_version = None

    async def _get_igraph(self):
        cursor = await self.dotvdb.conn.execute(
            "SELECT current_version_count FROM global_config WHERE id = 1"
        )
        row = await cursor.fetchone()
        await cursor.close()
        version = row[0]
        if self._igraph is not None and self._igraph_version == version:
            return self._igraph

        cursor = await self.dotvdb.conn.execute("SELECT entity_id FROM entity")
        node_ids = [str(row[0]) for row in await cursor.fetchall()]
        await cursor.close()
        cursor = await self.dotvdb.conn.execute(
            "SELECT relation_id, src_entity_id, dst_entity_id FROM relationship"
        )
        edge_rows = await cursor.fetchall()
        await cursor.close()

        graph = ig.Graph(directed=False)
        graph.add_vertices(node_ids)
        known = set(node_ids)
        graph.add_edges(
            (str(source), str(target))
            for _, source, target in edge_rows
            if str(source) in known and str(target) in known
        )
        graph.es["relation_id"] = [
            str(relation_id)
            for relation_id, source, target in edge_rows
            if str(source) in known and str(target) in known
        ]
        self._igraph = graph
        self._igraph_version = version
        return graph

    def _serialize_nodes(self, rows):
        return [
            {
                "id": str(row[0]),
                "name": row[1],
                "description": row[2],
                "degree": row[3] or 0,
                "types": json.loads(row[4]),
            }
            for row in rows
        ]

    def _serialize_edges(self, rows):
        return [
            {
                "id": str(row[0]),
                "source": str(row[1]),
                "target": str(row[2]),
                "description": row[3],
            }
            for row in rows
        ]

    async def _status(self, request):
        async with self.dotvdb.lock:
            cursor = await self.dotvdb.conn.execute(
                """
                SELECT current_version_count, num_entities, num_relations
                FROM global_config
                WHERE id = 1
                """
            )
            row = await cursor.fetchone()
            await cursor.close()
        return web.json_response(
            {
                "database": str(getattr(self.dotvdb, "path", "dotvectordb")),
                "version": row[0],
                "entities": row[1] or 0,
                "relationships": row[2] or 0,
            }
        )

    async def _graph_data(self, request):
        limit = min(max(int(request.query.get("limit", 500)), 1), 1000)
        node_offset = max(int(request.query.get("node_offset", 0)), 0)
        edge_offset = max(int(request.query.get("edge_offset", 0)), 0)
        async with self.dotvdb.lock:
            node_cursor = await self.dotvdb.conn.execute(
                """
                SELECT
                    entity.entity_id,
                    entity.entity_name,
                    entity.entity_description,
                    entity.degrees,
                    COALESCE(
                        (
                            SELECT json_group_array(entity_type)
                            FROM entity_to_entity_type_mapping
                            WHERE entity_to_entity_type_mapping.entity_id = entity.entity_id
                        ),
                        '[]'
                    ) AS entity_types
                FROM entity
                ORDER BY entity.entity_id
                LIMIT ? OFFSET ?
                """,
                (limit, node_offset),
            )
            node_rows = await node_cursor.fetchall()
            await node_cursor.close()

            edge_cursor = await self.dotvdb.conn.execute(
                """
                SELECT
                    relation_id,
                    src_entity_id,
                    dst_entity_id,
                    relation_description
                FROM relationship
                ORDER BY relation_id
                LIMIT ? OFFSET ?
                """
                , (limit, edge_offset)
            )
            edge_rows = await edge_cursor.fetchall()
            await edge_cursor.close()

        return web.json_response(
            {
                "nodes": self._serialize_nodes(node_rows),
                "edges": self._serialize_edges(edge_rows),
                "next_node_offset": node_offset + len(node_rows),
                "next_edge_offset": edge_offset + len(edge_rows),
                "has_more_nodes": len(node_rows) == limit,
                "has_more_edges": len(edge_rows) == limit,
            }
        )

    async def _neighbors(self, request):
        entity_id = int(request.match_info["entity_id"])
        limit = min(max(int(request.query.get("limit", 100)), 1), 500)
        async with self.dotvdb.lock:
            edge_cursor = await self.dotvdb.conn.execute(
                """
                SELECT relation_id, src_entity_id, dst_entity_id, relation_description
                FROM relationship
                WHERE src_entity_id = ? OR dst_entity_id = ?
                LIMIT ?
                """,
                (entity_id, entity_id, limit),
            )
            edge_rows = await edge_cursor.fetchall()
            await edge_cursor.close()

            node_ids = {entity_id}
            for row in edge_rows:
                node_ids.add(row[1])
                node_ids.add(row[2])
            placeholders = ",".join("?" for _ in node_ids)
            node_cursor = await self.dotvdb.conn.execute(
                f"""
                SELECT
                    entity.entity_id,
                    entity.entity_name,
                    entity.entity_description,
                    entity.degrees,
                    COALESCE(
                        (
                            SELECT json_group_array(entity_type)
                            FROM entity_to_entity_type_mapping
                            WHERE entity_to_entity_type_mapping.entity_id = entity.entity_id
                        ),
                        '[]'
                    ) AS entity_types
                FROM entity
                WHERE entity.entity_id IN ({placeholders})
                """,
                tuple(node_ids),
            )
            node_rows = await node_cursor.fetchall()
            await node_cursor.close()

        return web.json_response(
            {
                "nodes": self._serialize_nodes(node_rows),
                "edges": self._serialize_edges(edge_rows),
            }
        )

    async def _subgraph(self, request):
        entity_id = str(int(request.match_info["entity_id"]))
        hops = max(int(request.query.get("hops", 1)), 1)
        requested_types = {
            value for value in request.query.getall("type", []) if value
        }

        async with self.dotvdb.lock:
            graph = await self._get_igraph()
            try:
                center = graph.vs.find(name=entity_id).index
            except ValueError:
                raise web.HTTPNotFound(text="Entity not found")
            node_ids = {
                graph.vs[index]["name"]
                for index in graph.neighborhood(center, order=hops)
            }

            placeholders = ",".join("?" for _ in node_ids)
            node_cursor = await self.dotvdb.conn.execute(
                f"""
                SELECT
                    entity.entity_id,
                    entity.entity_name,
                    entity.entity_description,
                    entity.degrees,
                    COALESCE(
                        (
                            SELECT json_group_array(entity_type)
                            FROM entity_to_entity_type_mapping
                            WHERE entity_to_entity_type_mapping.entity_id = entity.entity_id
                        ),
                        '[]'
                    ) AS entity_types
                FROM entity
                WHERE entity.entity_id IN ({placeholders})
                """,
                tuple(node_ids),
            )
            node_rows = await node_cursor.fetchall()
            await node_cursor.close()
            available_types = sorted(
                {
                    entity_type
                    for row in node_rows
                    for entity_type in json.loads(row[4])
                }
            )

            if requested_types:
                node_rows = [
                    row for row in node_rows
                    if str(row[0]) == entity_id
                    or requested_types.intersection(json.loads(row[4]))
                ]
            included_ids = {str(row[0]) for row in node_rows}

            included_placeholders = ",".join("?" for _ in included_ids)
            edge_cursor = await self.dotvdb.conn.execute(
                f"""
                SELECT relation_id, src_entity_id, dst_entity_id, relation_description
                FROM relationship
                WHERE src_entity_id IN ({included_placeholders})
                  AND dst_entity_id IN ({included_placeholders})
                """,
                tuple(included_ids) + tuple(included_ids),
            )
            edge_rows = await edge_cursor.fetchall()
            await edge_cursor.close()

        return web.json_response(
            {
                "nodes": self._serialize_nodes(node_rows),
                "edges": self._serialize_edges(edge_rows),
                "hops": hops,
                "available_types": available_types,
            }
        )

    async def _shortest_paths(self, request):
        source_id = str(int(request.match_info["entity_id"]))
        target_id = request.query.get("target_id")
        target_name = request.query.get("target_name", "").strip()
        k = max(int(request.query.get("k", 1)), 1)

        async with self.dotvdb.lock:
            if target_id:
                target_id = str(int(target_id))
            elif target_name:
                cursor = await self.dotvdb.conn.execute(
                    """
                    SELECT entity_id
                    FROM entity
                    WHERE entity_name = ? COLLATE NOCASE
                    ORDER BY entity_id
                    LIMIT 1
                    """,
                    (target_name,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise web.HTTPNotFound(text="Destination entity name not found")
                target_id = str(row[0])
            else:
                raise web.HTTPBadRequest(text="A destination entity is required")

            graph = await self._get_igraph()
            try:
                source = graph.vs.find(name=source_id).index
                target = graph.vs.find(name=target_id).index
            except ValueError:
                raise web.HTTPNotFound(text="Source or destination entity not found")

            vertex_paths = graph.get_k_shortest_paths(
                source, to=target, k=k, mode="all", output="vpath"
            )
            paths = []
            all_node_ids = set()
            all_edge_ids = set()
            for rank, vertex_path in enumerate(vertex_paths, 1):
                if not vertex_path:
                    continue
                path_node_ids = [graph.vs[index]["name"] for index in vertex_path]
                path_edge_ids = []
                for left, right in zip(vertex_path, vertex_path[1:]):
                    edge_index = graph.get_eid(left, right, directed=False)
                    path_edge_ids.append(graph.es[edge_index]["relation_id"])
                all_node_ids.update(path_node_ids)
                all_edge_ids.update(path_edge_ids)
                paths.append(
                    {
                        "rank": rank,
                        "nodes": path_node_ids,
                        "edges": path_edge_ids,
                    }
                )

            node_rows = []
            edge_rows = []
            if all_node_ids:
                placeholders = ",".join("?" for _ in all_node_ids)
                cursor = await self.dotvdb.conn.execute(
                    f"""
                    SELECT
                        entity.entity_id,
                        entity.entity_name,
                        entity.entity_description,
                        entity.degrees,
                        COALESCE(
                            (
                                SELECT json_group_array(entity_type)
                                FROM entity_to_entity_type_mapping
                                WHERE entity_to_entity_type_mapping.entity_id = entity.entity_id
                            ),
                            '[]'
                        )
                    FROM entity
                    WHERE entity.entity_id IN ({placeholders})
                    """,
                    tuple(all_node_ids),
                )
                node_rows = await cursor.fetchall()
                await cursor.close()
            if all_edge_ids:
                placeholders = ",".join("?" for _ in all_edge_ids)
                cursor = await self.dotvdb.conn.execute(
                    f"""
                    SELECT relation_id, src_entity_id, dst_entity_id, relation_description
                    FROM relationship
                    WHERE relation_id IN ({placeholders})
                    """,
                    tuple(all_edge_ids),
                )
                edge_rows = await cursor.fetchall()
                await cursor.close()

        return web.json_response(
            {
                "source": source_id,
                "target": target_id,
                "paths": paths,
                "nodes": self._serialize_nodes(node_rows),
                "edges": self._serialize_edges(edge_rows),
            }
        )

    async def _index(self, request):
        html_path = Path(__file__).with_name("index.html")
        return web.FileResponse(html_path)

    async def _michigan_logo(self, request):
        return web.FileResponse(
            Path(__file__).parent.parent
            / "sql_browser"
            / "university-of-michigan-vector-logo.svg"
        )

    async def _michigan_logo_white(self, request):
        return web.FileResponse(
            Path(__file__).parent.parent
            / "sql_browser"
            / "university-of-michigan-vector-logo-white.svg"
        )

    async def start(self, open_browser=True):
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/assets/michigan-logo.svg", self._michigan_logo)
        app.router.add_get(
            "/assets/michigan-logo-white.svg",
            self._michigan_logo_white,
        )
        app.router.add_get("/api/status", self._status)
        app.router.add_get("/api/graph", self._graph_data)
        app.router.add_get("/api/neighbors/{entity_id}", self._neighbors)
        app.router.add_get("/api/subgraph/{entity_id}", self._subgraph)
        app.router.add_get("/api/shortest-paths/{entity_id}", self._shortest_paths)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        url = f"http://{self.host}:{self.port}"
        print(f"Live graph interface: {url}")
        if open_browser:
            await asyncio.to_thread(webbrowser.open, url)
        return self

    async def close(self):
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None


async def open_graph_interface(dotvdb, port=8765, host="127.0.0.1", open_browser=True):
    server = GraphInterfaceServer(dotvdb=dotvdb, host=host, port=port)
    return await server.start(open_browser=open_browser)
