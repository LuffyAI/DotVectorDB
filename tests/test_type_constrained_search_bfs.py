import asyncio

import igraph as ig

from dotvectordb.graph.graph_retriever import GraphRetriever


def make_retriever():
    graph = ig.Graph(
        edges=[
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (3, 5),
        ],
        directed=False,
    )
    graph.vs["name"] = [
        "Anchor",
        "Branching Person",
        "Dead Person",
        "Continuing Organization",
        "Dead Organization",
        "Destination City",
    ]
    graph.vs["entity_type"] = [
        ["anchor"],
        ["person"],
        ["person"],
        ["organization"],
        ["organization"],
        ["city"],
    ]
    graph.es["relation_id"] = ["r1", "r2", "r3", "r4", "r5"]

    retriever = GraphRetriever(
        graph_path="unused",
        conn=None,
        lock=None,
        get_num_nodes_edges_func=None,
        vector_search=None,
    )
    retriever.graph = graph

    async def get_anchor_index(entity_id, graph):
        assert entity_id == "anchor-id"
        return 0

    retriever.get_igraph_index = get_anchor_index
    return retriever


def test_type_constrained_bfs_returns_only_deepest_completed_hop():
    retriever = make_retriever()

    entity_paths, relationship_paths = asyncio.run(
        retriever.type_constrained_search_bfs(
            anchor_node="anchor-id",
            ordered_entity_types=[
                ["person"],
                ["organization"],
                ["city"],
            ],
        )
    )

    assert entity_paths == {
        "3_hop": [
            [
                "Anchor",
                "Branching Person",
                "Continuing Organization",
                "Destination City",
            ]
        ],
    }
    assert relationship_paths == {
        "3_hop": [["r1", "r3", "r5"]],
    }


def test_type_constrained_bfs_returns_last_frontier_when_plan_stops():
    retriever = make_retriever()

    entity_paths, relationship_paths = asyncio.run(
        retriever.type_constrained_search_bfs(
            anchor_node="anchor-id",
            ordered_entity_types=[
                ["person"],
                ["organization"],
                ["missing-type"],
            ],
        )
    )

    assert entity_paths == {
        "2_hop": [
            ["Anchor", "Branching Person", "Continuing Organization"],
            ["Anchor", "Branching Person", "Dead Organization"],
        ]
    }
    assert relationship_paths == {
        "2_hop": [["r1", "r3"], ["r1", "r4"]]
    }


def test_type_constrained_bfs_omits_anchor_only_failure():
    retriever = make_retriever()

    result = asyncio.run(
        retriever.type_constrained_search_bfs(
            anchor_node="anchor-id",
            ordered_entity_types=[["missing-type"]],
        )
    )

    assert result == ({}, {})
