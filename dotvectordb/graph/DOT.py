import uuid
from dotvectordb.data_structures.path import PathManager
class DOT:
    def __init__(self, mode, centroid_id, semantic_neighborhood, vector_indices, temp_graph):
        """
        Initializes a DOT (Division of Thought) instance with the provided parameters.
        A DOT (“Division of Thought”) packages everything needed to explore one promising region of a knowledge graph and decide which reasoning paths are valid.
        - centroid_id: The ID of the centroid entity anchoring this DOT. The centroid represents one semantic region predicted to be important for answering the query.
        - semantic_neighborhood: A list of entities found to be semantically similar to our centroid entity. This covers possible aliases or related entities that may be relevant to the query.
        - vector_indices: Each relationship in the graph has a vector index. This is a list of the vector indices for the edges in the temporary graph that will be used to construct the DOT's temporary vector store, thus letting us perform semantic retrieval over relationships in that subgraph.
        - temp_graph: A pruned, query-specific graph rather than searching the entire knowledge graph, containing only relevant entities and edges that can be traversed to the centroid or an entity in the related entity cluster. We use this to apply pathfinding algorithms.
        - llm_judge_rubric_dict: A dictionary containing the rubric for judging the validity of retrieved paths from the graph. This is used to determine which paths should be considered for the final context and which not.
        - memory_dict: Mutable retrieval state containing evidence, paths, decisions, or other information accumulated while exploring this DOT.
        """
        self.centroid_id = centroid_id
        self.sql_table_name = self.prepare_table_name()
        self.semantic_neighborhood = semantic_neighborhood
        self.vector_indices = vector_indices
        self.temp_graph = temp_graph
        self.mode = mode
    def prepare_table_name(self):
        if self.centroid_id < 0:
            sql_table_name = f"temp_vector_search_neg_{abs(self.centroid_id)}_{uuid.uuid4().hex[:8]}"
        else:
            sql_table_name = f"temp_vector_search_{self.centroid_id}_{uuid.uuid4().hex[:8]}"
        return sql_table_name
    def prepare_temp_vector_space_sql_query(self, workspace_id):
        return f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS temp.{self.sql_table_name}_{workspace_id}
                USING vec1(embedding);
                """
    def close_temp_vector_space_sql_query(self, workspace_id):
        return f"""
                DROP TABLE IF EXISTS temp.{self.sql_table_name}_{workspace_id};
                """
    
