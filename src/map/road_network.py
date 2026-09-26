"""Построение направленного графа из автомобильных OSM-дорог."""

from __future__ import annotations

from typing import Any

from src.map.geometry import road_width_m, segment_heading_rad, segment_length_m
from src.map.preprocessing import is_drivable_way, lane_count, speed_limit_mps, travel_directions
from src.map.projection import LocalProjection


def build_road_network(
    elements: list[dict[str, Any]],
    nodes: dict[int, dict[str, Any]],
    projection: LocalProjection,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Возвращает локальные узлы, направленные рёбра и ограничения поворотов."""
    local_nodes = {
        str(node_id): {"x_m": projection.project(node["lat"], node["lon"])[0], "y_m": projection.project(node["lat"], node["lon"])[1]}
        for node_id, node in nodes.items()
    }
    edges: list[dict[str, Any]] = []
    known_way_ids: set[int] = set()

    for way in (element for element in elements if is_drivable_way(element)):
        way_id = way["id"]
        way_nodes = way["nodes"]
        if any(node_id not in nodes for node_id in way_nodes):
            continue
        known_way_ids.add(way_id)
        tags = way.get("tags", {})
        allow_forward, allow_backward = travel_directions(tags)
        lanes = lane_count(tags)
        for index, (start_id, end_id) in enumerate(zip(way_nodes, way_nodes[1:])):
            start = (local_nodes[str(start_id)]["x_m"], local_nodes[str(start_id)]["y_m"])
            end = (local_nodes[str(end_id)]["x_m"], local_nodes[str(end_id)]["y_m"])
            length = segment_length_m(start, end)
            if length == 0:
                continue
            common = {
                "way_id": way_id,
                "segment_index": index,
                "highway": tags["highway"],
                "length_m": length,
                "speed_limit_mps": speed_limit_mps(tags),
                "lane_count": lanes,
                "road_width_m": road_width_m(lanes),
                "name": tags.get("name") or tags.get("name:ru"),
            }
            if allow_forward:
                edges.append({
                    **common,
                    "id": f"{way_id}:{index}:forward",
                    "from_node_id": start_id,
                    "to_node_id": end_id,
                    "geometry": [[start[0], start[1]], [end[0], end[1]]],
                    "heading_rad": segment_heading_rad(start, end),
                })
            if allow_backward:
                edges.append({
                    **common,
                    "id": f"{way_id}:{index}:backward",
                    "from_node_id": end_id,
                    "to_node_id": start_id,
                    "geometry": [[end[0], end[1]], [start[0], start[1]]],
                    "heading_rad": segment_heading_rad(end, start),
                })

    restrictions = _extract_turn_restrictions(elements, known_way_ids)
    return local_nodes, edges, restrictions


def _extract_turn_restrictions(
    elements: list[dict[str, Any]], known_way_ids: set[int]
) -> list[dict[str, Any]]:
    restrictions: list[dict[str, Any]] = []
    for relation in elements:
        tags = relation.get("tags", {})
        if relation.get("type") != "relation" or tags.get("type") != "restriction":
            continue
        members = relation.get("members", [])
        from_way = next((member.get("ref") for member in members if member.get("role") == "from" and member.get("type") == "way"), None)
        to_way = next((member.get("ref") for member in members if member.get("role") == "to" and member.get("type") == "way"), None)
        via_node = next((member.get("ref") for member in members if member.get("role") == "via" and member.get("type") == "node"), None)
        if from_way in known_way_ids and to_way in known_way_ids and isinstance(via_node, int):
            restrictions.append({
                "relation_id": relation["id"],
                "kind": tags.get("restriction"),
                "from_way_id": from_way,
                "to_way_id": to_way,
                "via_node_id": via_node,
            })
    return restrictions
