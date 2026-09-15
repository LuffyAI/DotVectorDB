from aiosqlite import cursor
from dotvectordb.data_structures.path import Node, Edge, Path
from dotvectordb.data_structures.trie import PathTrie
import igraph as ig
from collections import defaultdict, deque
import traceback
import os
import json

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
            print(f"Graph file for current database version not found at {self.graph_path}. Attempting to create a new graph.")
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
                    """
                    SELECT
                        entity.entity_id,
                        COALESCE(
                            (
                                SELECT json_group_array(entity_type)
                                FROM entity_to_entity_type_mapping
                                WHERE entity_to_entity_type_mapping.entity_id = entity.entity_id
                            ),
                            '[]'
                        ) AS entity_types,
                        entity.entity_name
                    FROM entity
                    """
                )
                nodes = await cursor.fetchall()

                cursor = await self.conn.execute(
                    "SELECT relation_id, src_entity_id, dst_entity_id, relation_description FROM relationship"
                )
                edges = await cursor.fetchall()
                g = ig.Graph(directed=False)
                g.add_vertices([str(node[0]) for node in nodes], attributes={"entity_type": [json.loads(node[1]) for node in nodes], "entity_name": [node[2] for node in nodes]})
                g.add_edges([(str(edge[1]), str(edge[2])) for edge in edges], attributes={"relation_id": [edge[0] for edge in edges], "relation_description": [edge[3] for edge in edges]})
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
    #            Graph Retrieval Helper Functions              # 
    ############################################################
    async def get_igraph_index(self, entity_id, graph=None):
        """Retrieve the igraph index of a given entity ID."""
        try:
            if graph == None:
                graph = self.graph
            vertex = graph.vs.find(name=str(entity_id))
            return vertex.index
        except ValueError:
            raise KeyError(f"Entity does not exist: {entity_id}")
    async def is_entity_an_island(self, entity_id, graph=None):
        """Check if a given entity is an island (i.e., has no edges) in the graph."""
        try:
            if graph == None:
                graph = self.graph
            vertex = await self.get_igraph_index(entity_id=entity_id, graph=graph)
            return len(graph.incident(vertex, mode="all")) == 0
        except ValueError:
            raise KeyError(f"Entity does not exist: {entity_id}")
    async def get_entity_neighbor_types(
        self,
        hops=10,
        entity_id=None,
        graph=None,
    ):
        """Retrieve neighboring entity types grouped by exact hop."""
        try:
            if graph is None:
                graph = self.graph

            vertex = await self.get_igraph_index(
                entity_id=entity_id,
                graph=graph,
            )

            neighbor_types_by_hop = {}

            for hop in range(1, hops + 1):
                neighbors = graph.neighborhood(
                    vertex,
                    order=hop,
                    mindist=hop,
                    mode="all",
                )

                neighbor_types_by_hop[hop] = {
                    entity_type
                    for neighbor in neighbors
                    for entity_type in graph.vs[neighbor]["entity_type"]
                }

            return neighbor_types_by_hop

        except ValueError:
            raise KeyError(f"Entity does not exist: {entity_id}")
            
    # async def get_shortest_k_paths(self, graph, source, destination, prev_path_trie=None, k=10):
    #     """Retrieve the k shortest paths between two nodes in the graph."""
    #     # We have a root entity and we're finding the shortest to a specific edge. First, we find the shortest paths from root to the source node of the edge.
    #     if prev_path_trie is None:
    #         prev_path_trie = PathTrie()
    #     found_paths = []
    #     source_index = await self.get_igraph_index(entity_id=source)
    #     dst_index = await self.get_igraph_index(entity_id=destination)
    #     shortest_paths = graph.get_k_shortest_paths(
    #         source_index,
    #         to=dst_index,
    #         k=k,
    #         weights=None  
    #     )
    #     # If no paths are found, we return an empty list. This can happen if the root entity is not connected to the source node.
    #     if shortest_paths is None or len(shortest_paths) == 0:
    #         return found_paths, prev_path_trie
    #     # If paths are found, we process each one.
    #     for path in shortest_paths: 
    #         # First, we check if the path is already in the prev_path_set. If it is, we skip it. If not, we add it to the prev_path_set and continue processing.
    #         current_entities = []
    #         current_relationships = []
    #         path_text = ""
    #         if path in prev_path_set:
    #             continue
    #         else:
    #             prev_path_set.add(tuple(path))
    #         for vertex_index in path:
    #             # We find the metadata associated with the current entity
    #             current_node = graph.vs[vertex_index]
    #             entity = Node(entity_id=current_node["name"], entity_name=current_node["entity_name"], entity_type=current_node["entity_type"])
    #             current_entities.append(entity)
    #             # After the first entity, we need to look one back to find the relationship that connects the previous entity to the current one.
    #             if len(current_entities) > 1:
    #                 prev_entity = current_entities[len(current_entities) - 2]
    #                 rel = graph.es.find(_between=([prev_entity.entity_id], [entity.entity_id]))
    #                 relationship = Edge(rel_id=rel["relation_id"], src_entity=prev_entity, dst_entity=entity, description=rel["relation_description"])
    #                 current_relationships.append(relationship)
    #                 path_text += f"**{prev_entity.entity_name} ({prev_entity.entity_type})** is related to **{entity.entity_name} ({entity.entity_type})** because [{relationship.description}]."
    #         new_path = Path(entities=current_entities, relationships=current_relationships, description=path_text)
    #         found_paths.append(new_path)
    #     return found_paths, prev_path_set
    # async def get_shortest_k_paths(
    #     self,
    #     graph,
    #     source,
    #     destination,
    #     end_point,
    #     PathManager,
    #     k=10,
    # ):
    #     """
    #     Precondition:
    #         - The graph is already loaded and is an igraph.Graph object.
    #     Retrieve the k shortest paths between two nodes in the graph.
    #     """
    #     # First, we need the corresponding igraph indices for the source, destination, and end_index ids.
    #     # Source index is the query's centroid
    #     # Dst index is the source node of the edge we're trying to reach
    #     # end_index is the other node of the edge we're trying to reach.
    #     # We use top-k yen shortest path to move for source to dst then append the end_index to the path, if not already in the path 
    #     source_index = await self.get_igraph_index(entity_id=source, graph=graph)
    #     dst_index = await self.get_igraph_index(entity_id=destination, graph=graph)
    #     end_index = await self.get_igraph_index(entity_id=end_point, graph=graph)
    #     shortest_paths = graph.get_k_shortest_paths(
    #         source_index,
    #         to=dst_index,
    #         k=k,
    #         weights=None,
    #     )
    #     # If no paths are found, then this means there were no paths from the centroid to the edge we wanted to reach.
    #     if not shortest_paths:
    #         return False

    #     for path in shortest_paths:

    #         # Skip if this path is already represented by a previously seen path.
    #         #
    #         # Stored: [1, 2, 3]
    #         # Current: [1, 2] -> skip
    #         #
    #         # Stored: [1, 2, 3]
    #         # Current: [1, 2, 4] -> process
    #         isEndPointInPath = False

    #         if path in prev_path_trie:
    #             continue

    #         # Add the new path.
    #         prev_path_trie.add(path)

    #         current_entities = []
    #         current_relationships = []
    #         path_text = ""

    #         for vertex_index in path:
    #             current_node = graph.vs[vertex_index]

    #             entity = Node(
    #                 entity_id=current_node["name"],
    #                 entity_name=current_node["entity_name"],
    #                 entity_type=current_node["entity_type"],
    #             )
    #             current_entities.append(entity)

    #             # We need to manually append it to the end
    #             if isEndPointInPath == False:
    #                 if path[-1] != end_index:
    #                  path.append(end_index)
    #                 isEndPointInPath = True

    #             # We always look one back to find the relationship that connects the previous entity to the current one.
    #             if len(current_entities) > 1:
    #                 prev_entity = current_entities[-2]
    #                 prev_vertex = await self.get_igraph_index(entity_id=prev_entity.entity_id, graph=graph)

    #                 edge_id = graph.get_eid(
    #                     prev_vertex,
    #                     vertex_index,
    #                     directed=True,
    #                     error=False,
    #                 )

    #                 if edge_id == -1:
    #                     edge_id = graph.get_eid(
    #                         vertex_index,
    #                         prev_vertex,
    #                         directed=True,
    #                         error=False,
    #                     )

    #                 rel = graph.es[edge_id]

    #                 relationship = Edge(
    #                     rel_id=rel["relation_id"],
    #                     src_entity=prev_entity,
    #                     dst_entity=entity,
    #                     description=rel["relation_description"],
    #                 )

    #                 current_relationships.append(relationship)

    #                 path_text += (
    #                     f"**{prev_entity.entity_name} ({prev_entity.entity_type})** "
    #                     f"is related to "
    #                     f"**{entity.entity_name} ({entity.entity_type})** "
    #                     f"because [{relationship.description}]."
    #                 )

    

    #         new_path = Path(
    #             entities=current_entities,
    #             relationships=current_relationships,
    #             description=path_text,
    #         )

    #         found_paths.append(new_path)

    #     return True

    async def get_shortest_k_paths(
        self,
        graph,
        source,
        destination,
        end_point,
        PathManager,
        k=10,
    ):
        """
        Precondition:
            - The graph is already loaded and is an igraph.Graph object.
        Retrieve the k shortest paths from the source node to a target edge.
        """
        # First, we need the corresponding igraph indices for the source, destination, and end_index ids.
        # Source index is the query's centroid
        # Dst index is the source node of the edge we're trying to reach
        # end_index is the other node of the edge we're trying to reach.
        # We use top-k yen shortest path to move for source to dst then append the end_index to the path, if not already in the path
        source_index = await self.get_igraph_index(entity_id=source, graph=graph)
        dst_index = await self.get_igraph_index(entity_id=destination, graph=graph)
        end_index = await self.get_igraph_index(entity_id=end_point, graph=graph)
        shortest_paths = graph.get_k_shortest_paths(
            source_index,
            to=dst_index,
            k=k,
            weights=None,
        )
        # If no paths are found, then this means there were no paths from the centroid to the edge we wanted to reach.
        if not shortest_paths:
            return False
        for path in shortest_paths:
            complete_path = list(path)
            # We need to manually append the other endpoint so the path includes the edge we're trying to reach.
            if end_index not in complete_path:
                complete_path.append(end_index)
            PathManager.build_and_add_path(graph, complete_path)
        return True

#     async def bfs_weight_propagation(
#         self,
#         starting_node_id,
#         starting_weight=1.0,
#         banned_node_types=None,
#         degree_exponent=1.0,
#         hop_limit=None,
#     ):
#         banned_node_types = set(banned_node_types or [])
#         graph = self.graph

#         try:
#             starting_vertex = graph.vs.find(
#                 name=str(starting_node_id)
#             ).index
#         except ValueError as exc:
#             raise KeyError(
#                 f"Entity does not exist: {starting_node_id}"
#             ) from exc

#         queue = deque([
#             (starting_vertex, float(starting_weight))
#         ])

#         visited_edges = set()
#         edge_entity_ids = []
#         surviving_relationship_ids = []

#         while queue:
#             vertex_id, weight = queue.popleft()

#             if weight <= 0:
#                 continue

#             if graph.vs[vertex_id]["entity_type"] in banned_node_types:
#                 continue

#             eligible_edges = []

#             for edge_id in graph.incident(vertex_id, mode="all"):
#                 if edge_id in visited_edges:
#                     continue

#                 edge = graph.es[edge_id]
#                 source_id, target_id = edge.tuple

#                 neighbor_id = (
#                     target_id
#                     if source_id == vertex_id
#                     else source_id
#                 )

#                 if (
#                     graph.vs[neighbor_id]["entity_type"]
#                     in banned_node_types
#                 ):
#                     continue

#                 eligible_edges.append((edge_id, edge, neighbor_id))

#             if not eligible_edges:
#                 continue

#             propagated_weight = (weight / (len(eligible_edges) ** degree_exponent)
# )

#             # Mark edges as visited and add their relationship IDs to the surviving list
#             for edge_id, edge, neighbor_id in eligible_edges:
#                 visited_edges.add(edge_id)

#                 # These are primary keys in the relationship table, allowing for direct retrieval of the relationship data from the database.
#                 surviving_relationship_ids.append(
#                     edge["relation_id"]
#                 )

#                 # These tell us the exact primary keyts of entities connected by the edge, allowing for direct retrieval of the entity data from the database.
#                 edge_entity_ids.append(
#                     (
#                         graph.vs[edge.source]["name"],
#                         graph.vs[edge.target]["name"],
#                     )
#                 )

#                 queue.append(
#                     (neighbor_id, propagated_weight)
#                 )

#         return surviving_relationship_ids, edge_entity_ids

    async def bfs_weight_propagation(
        self,
        starting_node_id,
        starting_weight=1.0,
        banned_node_types=None,
        degree_exponent=1.0,
        hop_limit=None,
    ):
        """
        Traverse outward from a starting node.

        Two traversal modes:

        1. hop_limit is None or 0:
        - Use degree-based weight propagation.
        - Weight is divided across eligible neighboring edges.
        - Traversal continues while propagated weight is positive.

        2. hop_limit is greater than 0:
        - Do not use weight propagation.
        - Perform ordinary BFS expansion for at most `hop_limit` rounds.

        In both modes:
        - The starting node is never filtered by entity type.
        - A neighboring node is skipped only when all its entity types are banned.
        - Banned neighbors are not traversed and their edges are not returned.
        - Each edge is processed at most once.
        """

        banned_node_types = set(banned_node_types or [])
        graph = self.graph

        def is_node_banned(vertex_id):
            return all(
                entity_type in banned_node_types
                for entity_type in graph.vs[vertex_id]["entity_type"]
            )

        starting_vertex = await self.get_igraph_index(
            entity_id=starting_node_id
        )

        visited_edges = set()
        edge_entity_ids = []
        surviving_relationship_ids = []

        # ---------------------------------------------------------
        # MODE 1: No hop limit -> use weight propagation.
        # ---------------------------------------------------------
        if hop_limit is None or hop_limit == 0:
            queue = deque([
                (starting_vertex, float(starting_weight))
            ])

            while queue:
                vertex_id, weight = queue.popleft()

                if weight <= 0.01:
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

                    if is_node_banned(neighbor_id):
                        continue

                    eligible_edges.append(
                        (edge_id, edge, neighbor_id)
                    )

                if not eligible_edges:
                    continue

                propagated_weight = (
                    weight
                    / (len(eligible_edges) ** degree_exponent)
                )

                for edge_id, edge, neighbor_id in eligible_edges:
                    visited_edges.add(edge_id)

                    surviving_relationship_ids.append(
                        edge["relation_id"]
                    )

                    edge_entity_ids.append(
                        (
                            graph.vs[edge.source]["name"],
                            graph.vs[edge.target]["name"],
                        )
                    )

                    queue.append(
                        (neighbor_id, propagated_weight)
                    )

        # ---------------------------------------------------------
        # MODE 2: Hop limit provided -> ordinary bounded BFS.
        # ---------------------------------------------------------
        else:
            queue = deque([
                (starting_vertex, 0)
            ])

            while queue:
                vertex_id, hops = queue.popleft()

                if hops >= hop_limit:
                    continue

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

                    if is_node_banned(neighbor_id):
                        continue

                    visited_edges.add(edge_id)

                    surviving_relationship_ids.append(
                        edge["relation_id"]
                    )

                    edge_entity_ids.append(
                        (
                            graph.vs[edge.source]["name"],
                            graph.vs[edge.target]["name"],
                        )
                    )

                    queue.append(
                        (neighbor_id, hops + 1)
                    )

        return surviving_relationship_ids, edge_entity_ids