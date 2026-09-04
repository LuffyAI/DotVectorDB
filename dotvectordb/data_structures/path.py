class Node:
    __slots__ = ("entity_id", "entity_name", "entity_type", "path_count")
    def __init__(self, entity_id, entity_name, entity_type):
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.path_count = 0
class Edge:
    __slots__ = ("rel_id", "src_entity", "dst_entity", "description")
    def __init__(self, rel_id, src_entity, dst_entity, description):
        self.rel_id = rel_id
        self.src_entity = src_entity
        self.dst_entity = dst_entity
        self.description = description
class Path:
    __slots__ = ("entities","relationships", "relational_description", "path_text", "usefulness_score")
    def __init__(self, entities, relationships, description):
        self.entities = entities
        self.relationships = relationships
        self.relational_description = description
        self.path_text = " -> ".join(str(e.entity_name) for e in entities)
        self.usefulness_score = 0.0