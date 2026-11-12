"""Layered layout for the concept graph, rendered as SVG.

The graph is laid out in topological layers: prerequisites on the left,
dependent concepts on the right. Nodes are coloured by mastery level so the
progress is readable at a glance. Nodes are not clickable yet; the layout
exposes their ids so that interaction can be added later.
"""

from __future__ import annotations

from typing import Any


NODE_WIDTH = 120
NODE_HEIGHT = 50
LAYER_GAP = 220
ROW_GAP = 100
LEFT_MARGIN = 90
TOP_MARGIN = 70


def _level_color(level: float) -> str:
    if level >= 0.75:
        return "#43a047"
    if level >= 0.5:
        return "#fb8c00"
    if level >= 0.25:
        return "#f4511e"
    return "#c62828"


def layout_graph(
    concepts: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    mastery: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return node positions and edge endpoints for an SVG rendering."""
    mastery = mastery or {}
    concept_ids = [concept["concept_id"] for concept in concepts]
    label_by_id = {concept["concept_id"]: concept["label"] for concept in concepts}
    prerequisites = {concept_id: set() for concept_id in concept_ids}
    children = {concept_id: [] for concept_id in concept_ids}
    for edge in edges:
        prerequisite_id = edge["prerequisite_id"]
        concept_id = edge["concept_id"]
        if prerequisite_id in prerequisites and concept_id in prerequisites:
            prerequisites[concept_id].add(prerequisite_id)
            children[prerequisite_id].append(concept_id)

    indegree = {concept_id: len(prerequisites[concept_id]) for concept_id in concept_ids}
    queue = [concept_id for concept_id in concept_ids if indegree[concept_id] == 0]
    order: list[str] = []
    while queue:
        concept_id = queue.pop(0)
        order.append(concept_id)
        for child in children[concept_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    layer: dict[str, int] = {}
    for concept_id in order:
        if not prerequisites[concept_id]:
            layer[concept_id] = 0
        else:
            layer[concept_id] = max(layer[item] for item in prerequisites[concept_id]) + 1

    layers: dict[int, list[str]] = {}
    for concept_id in concept_ids:
        layers.setdefault(layer[concept_id], []).append(concept_id)

    nodes_by_id: dict[str, dict[str, Any]] = {}
    ordered_nodes: list[dict[str, Any]] = []
    max_nodes = 0
    max_layer = 0
    for layer_index in sorted(layers):
        layer_ids = sorted(layers[layer_index], key=lambda item: label_by_id[item])
        max_nodes = max(max_nodes, len(layer_ids))
        max_layer = max(max_layer, layer_index)
        for index, concept_id in enumerate(layer_ids):
            level = mastery.get(concept_id, 0.0)
            node = {
                "concept_id": concept_id,
                "label": label_by_id[concept_id],
                "level": level,
                "percent": round(level * 100),
                "color": _level_color(level),
                "x": LEFT_MARGIN + layer_index * LAYER_GAP,
                "y": TOP_MARGIN + index * ROW_GAP,
            }
            nodes_by_id[concept_id] = node
            ordered_nodes.append(node)

    rendered_edges: list[dict[str, Any]] = []
    for edge in edges:
        prerequisite_id = edge["prerequisite_id"]
        concept_id = edge["concept_id"]
        if prerequisite_id in nodes_by_id and concept_id in nodes_by_id:
            rendered_edges.append(
                {
                    "prerequisite_id": prerequisite_id,
                    "concept_id": concept_id,
                    "x1": nodes_by_id[prerequisite_id]["x"] + NODE_WIDTH / 2,
                    "y1": nodes_by_id[prerequisite_id]["y"],
                    "x2": nodes_by_id[concept_id]["x"] - NODE_WIDTH / 2,
                    "y2": nodes_by_id[concept_id]["y"],
                }
            )

    return {
        "nodes": ordered_nodes,
        "edges": rendered_edges,
        "width": LEFT_MARGIN * 2 + max_layer * LAYER_GAP,
        "height": TOP_MARGIN * 2 + max(0, max_nodes - 1) * ROW_GAP,
    }
