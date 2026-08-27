from aiosqlite import cursor
import igraph as ig
from collections import defaultdict, deque
import traceback
import os

class GraphRetriever():
    def __init__(self, graph_path, conn, lock, get_num_nodes_edges_func):
        self.conn = conn
        self.graph_path = graph_path
        self.graph = None
        self.lock = lock
        self.get_num_nodes_edges_func = get_num_nodes_edges_func
    
    ############################################################
    #   Loads Persistent Graph                                 # 
    ############################################################
    async def load_graph(self, verbose=True):
        if self.conn is None:
            raise ValueError("Database connection is not initialized. Please open the database first.")
        # Get the number of nodes and edges from the SQL database
        num_nodes, num_edges = await self.get_num_nodes_edges_func()
        try:
            # Step One: Attempt to load the graph from the specified path. If it fails, it raises and exception, which creates a new graph.
            self.graph = ig.Graph.Read_Pickle(self.graph_path)
            if verbose:
             print(f"Graph loaded with {self.graph.summary()} nodes and edges.")
             print(f"Graph loaded from {self.graph_path}.")
            # Step Two: Validate the loaded graph against the database counts. If they don't match, we delete the loaded graph and create a new one.
            if self.graph.vcount() != num_nodes or self.graph.ecount() != num_edges:
                print(f"Graph validation failed: Expected {num_nodes} nodes and {num_edges} edges, but got {self.graph.vcount()} nodes and {self.graph.ecount()} edges.")
                print(f"Deleting the graph at {self.graph_path} and creating a new one.")
                await self.delete_graph(verbose=verbose)
                await self.create_graph(verbose=verbose)
            return self.graph
        except Exception as e:
            # If loading the graph fails, print the traceback and create a new graph.
            print(f"Graph loading failed in load_graph: {e}. Creating new graph.")
            print(f"Graph file not found at {self.graph_path}. Creating new graph.")
            await self.create_graph(verbose=verbose)

    ############################################################
    #   Deletes Persistent Graph                               # 
    ############################################################
    async def delete_graph(self, verbose=True): 
        try:
            # Set the graph attribute to None to indicate that there is no graph currently loaded.
            self.graph = None
            if os.path.exists(self.graph_path):
                os.remove(self.graph_path)
                if verbose:
                    print(f"Graph file {self.graph_path} deleted.")
            else:
                if verbose:
                    print(f"Graph file {self.graph_path} does not exist. Nothing to delete.")
        except Exception as e:
            traceback.print_exc()
            print(f"Graph deletion failed in delete_graph: {e}.")
            raise e

    ############################################################
    #   Persistent & Temporary Graph Creation Operations       # 
    ############################################################
    async def create_graph(self, verbose=True):
        try:
            async with self.lock:
                cursor = await self.conn.execute(
                    "SELECT entity_id, entity_type FROM entity"
                )
                nodes = await cursor.fetchall()

                cursor = await self.conn.execute(
                    "SELECT relation_id, src_entity_id, dst_entity_id FROM relationship"
                )
                edges = await cursor.fetchall()
                g = ig.Graph(directed=False)
                g.add_vertices([str(node[0]) for node in nodes], attributes={"entity_type": [node[1] for node in nodes]})
                g.add_edges([(str(edge[1]), str(edge[2])) for edge in edges], attributes={"relation_id": [edge[0] for edge in edges]})
                g.write_pickle(self.graph_path)
                self.graph = await self.load_graph(verbose=verbose)
        except Exception as e:
            traceback.print_exc()
            print(f"Graph creation failed in create_graph: {e}.")
            raise e

    async def create_temporary_relational_graph(self, relationship_ids, verbose=True):
        if relationship_ids is None:
            raise ValueError("relationship_ids cannot be None")
        
        if relationship_ids == []:
            raise ValueError("relationship_ids cannot be an empty list")
        
        if self.graph is None:
            raise ValueError("Original graph is not loaded")
        
        try: 
            requested_ids = {str(relationship_id) for relationship_id in relationship_ids}

            selected_edge_indexes = [
                edge.index
                for edge in self.graph.es
                if str(edge["relation_id"]) in requested_ids
            ]

            if not selected_edge_indexes:
                raise ValueError("None of the provided relationship IDs exist in the graph")

            # Create a subgraph containing only the selected edges and their connected vertices
            # delete_vertices=True removes vertices that are not connected to any selected edge.
            temporary_graph = self.graph.subgraph_edges(
                selected_edge_indexes,
                delete_vertices=True,
            )

            if verbose:
                print(temporary_graph.summary())

            return temporary_graph
        except Exception as e:
            traceback.print_exc()
            print(f"Temporary graph creation failed in create_temporary_graph: {e}.")
            raise e


    ############################################################
    #            Graph Retrieval Algorithms                    # 
    ############################################################
    async def shortest_k_paths(self, graph, src, dst, k=10):
         return graph.get_k_shortest_paths(
            src,
            to=dst,
            k=k,
            weights=None  
        )
    
    async def low_level_search_lightrag_style(self, graph, node_ids, k=10):
        """Retrieve the adjacent neighbors of the specified node IDs in the graph."""
        pass

    async def bfs_weight_propagation(
        self,
        starting_node_id,
        starting_weight=1.0,
        banned_node_types=None,
        degree_exponent=1.0
    ):
        banned_node_types = set(banned_node_types or [])
        graph = self.graph

        try:
            starting_vertex = graph.vs.find(
                name=str(starting_node_id)
            ).index
        except ValueError as exc:
            raise KeyError(
                f"Entity does not exist: {starting_node_id}"
            ) from exc

        queue = deque([
            (starting_vertex, float(starting_weight))
        ])

        visited_edges = set()
        edge_entity_ids = []
        surviving_relationship_ids = []

        while queue:
            vertex_id, weight = queue.popleft()

            if weight <= 0:
                continue

            if graph.vs[vertex_id]["entity_type"] in banned_node_types:
                continue

            eligible_edges = []

            for edge_id in graph.incident(vertex_id, mode="all"):
                if edge_id in visited_edges:
                    continue

                edge = graph.es[edge_id]
                source_id, target_id = edge.tuple

                neighbor_id = (
                    target_id
                    if source_id == vertex_id
                    else source_id
                )

                if (
                    graph.vs[neighbor_id]["entity_type"]
                    in banned_node_types
                ):
                    continue

                eligible_edges.append((edge_id, edge, neighbor_id))

            if not eligible_edges:
                continue

            propagated_weight = (weight / (len(eligible_edges) ** degree_exponent)
)

            # Mark edges as visited and add their relationship IDs to the surviving list
            for edge_id, edge, neighbor_id in eligible_edges:
                visited_edges.add(edge_id)

                # These are primary keys in the relationship table, allowing for direct retrieval of the relationship data from the database.
                surviving_relationship_ids.append(
                    edge["relation_id"]
                )

                # These tell us the exact primary keyts of entities connected by the edge, allowing for direct retrieval of the entity data from the database.
                edge_entity_ids.append(
                    (
                        graph.vs[edge.source]["name"],
                        graph.vs[edge.target]["name"],
                    )
                )

                queue.append(
                    (neighbor_id, propagated_weight)
                )

        return surviving_relationship_ids, edge_entity_ids

