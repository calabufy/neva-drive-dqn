#!/usr/bin/env python3
"""Подготавливает неизменяемый MapData для маршрутизации и среды."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.map.json_loader import index_nodes, load_osm, load_trafficlights
from src.map.projection import projection_from_nodes
from src.map.road_network import build_road_network
from src.map.preprocessing import is_drivable_way


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roads", type=Path, default=PROJECT_ROOT / "data/roads.json")
    parser.add_argument("--buildings", type=Path, default=PROJECT_ROOT / "data/buildings.json")
    parser.add_argument("--trafficlights", type=Path, default=PROJECT_ROOT / "data/trafficlights.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/prepared/map_data.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_buildings(elements: list[dict[str, Any]], projection: Any) -> list[dict[str, Any]]:
    """Сохраняет только замкнутые контуры зданий с готовой Overpass-геометрией."""
    buildings: list[dict[str, Any]] = []
    for element in elements:
        if element.get("type") != "way" or "building" not in element.get("tags", {}):
            continue
        geometry = element.get("geometry", [])
        if len(geometry) < 3:
            continue
        polygon = [list(projection.project(point["lat"], point["lon"])) for point in geometry]
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        buildings.append({
            "osm_id": element["id"],
            "building_type": element.get("tags", {}).get("building"),
            "name": element.get("tags", {}).get("name"),
            "polygon": polygon,
        })
    for element in elements:
        if element.get('type') != 'relation' or 'building' not in element.get('tags', {}):
            continue
        rings = {}
        for role in ('outer', 'inner'):
            pieces = [[(p['lat'], p['lon']) for p in m['geometry']]
                      for m in element.get('members', [])
                      if m.get('role') == role and m.get('geometry')]
            closed = []
            while pieces:
                ring = pieces.pop()
                while ring[0] != ring[-1]:
                    for i, piece in enumerate(pieces):
                        if piece[0] == ring[-1] or piece[-1] == ring[-1]:
                            if piece[-1] == ring[-1]:
                                piece = piece[::-1]
                            ring.extend(piece[1:])
                            pieces.pop(i)
                            break
                    else:
                        raise ValueError(f"Незамкнутый контур relation {element['id']}")
                closed.append([list(projection.project(*p)) for p in ring])
            rings[role] = closed
        if not rings['outer']:
            continue
        buildings.append({
            'osm_id': element['id'], 'osm_type': 'relation',
            'building_type': element['tags']['building'],
            'name': element['tags'].get('name'),
            'polygon': rings['outer'][0],
            'outers': rings['outer'], 'holes': rings['inner'],
        })
    return buildings


def prepare_signals(
    nodes: dict[int, dict[str, Any]], projection: Any,
    programs: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Готовит светофоры из YAML без привязки к направлению движения."""
    signals: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        if node.get("tags", {}).get("highway") != "traffic_signals" or node_id not in programs:
            continue
        x_m, y_m = projection.project(node["lat"], node["lon"])
        signals.append({
            "osm_id": node_id,
            "x_m": x_m,
            "y_m": y_m,
            "phases": programs[node_id],
            "is_programmed": True,
        })
    return signals


def prepare_signs(
    elements: list[dict[str, Any]], nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]], projection: Any,
) -> list[dict[str, Any]]:
    """Привязывает STOP и GIVE WAY к подъездным рёбрам по направлению OSM."""
    edge_ids = {edge["id"] for edge in edges}
    approaches: dict[int, set[str]] = {}
    for way in elements:
        if not is_drivable_way(way):
            continue
        way_nodes = way["nodes"]
        for index, node_id in enumerate(way_nodes):
            if index and way.get("id") is not None:
                edge_id = f"{way['id']}:{index - 1}:forward"
                if edge_id in edge_ids:
                    approaches.setdefault(node_id, set()).add(edge_id)
            if index < len(way_nodes) - 1:
                edge_id = f"{way['id']}:{index}:backward"
                if edge_id in edge_ids:
                    approaches.setdefault(node_id, set()).add(edge_id)

    signs = []
    for node_id, node in sorted(nodes.items()):
        kind = node.get("tags", {}).get("highway")
        if kind not in {"stop", "give_way"}:
            continue
        direction = node["tags"].get("direction")
        attached = sorted(
            edge_id for edge_id in approaches.get(node_id, set())
            if direction not in {"forward", "backward"} or edge_id.endswith(f":{direction}")
        )
        x_m, y_m = projection.project(node["lat"], node["lon"])
        signs.append({
            "osm_id": node_id,
            "type": kind,
            "x_m": x_m,
            "y_m": y_m,
            "direction": direction,
            "approach_edge_ids": attached,
        })
    return signs


def main() -> None:
    args = parse_args()
    road_elements = load_osm(args.roads)
    road_nodes = index_nodes(road_elements)
    projection = projection_from_nodes(road_nodes)
    nodes, edges, restrictions = build_road_network(road_elements, road_nodes, projection)
    building_elements = load_osm(args.buildings)
    relations_path = args.buildings.with_name('building_relations.json')
    if relations_path.exists():
        # Заменяем неполные relations из старой выгрузки версиями с geometry.
        relations = load_osm(relations_path)
        ids = {e['id'] for e in relations}
        building_elements = [e for e in building_elements
                             if not (e['type'] == 'relation' and e['id'] in ids)] + relations
    buildings = prepare_buildings(building_elements, projection)
    signals = prepare_signals(road_nodes, projection, load_trafficlights(args.trafficlights))
    signs = prepare_signs(road_elements, road_nodes, edges, projection)

    payload = {
        "version": 1,
        "coordinate_system": {
            "source": "EPSG:4326",
            "projected": "EPSG:32636",
            "local_origin_utm_m": [projection.origin_easting_m, projection.origin_northing_m],
            "units": "m",
        },
        "sources": {
            "roads": str(args.roads),
            "roads_sha256": file_sha256(args.roads),
            "buildings": str(args.buildings),
            "buildings_sha256": file_sha256(args.buildings),
            "trafficlights": str(args.trafficlights),
            "trafficlights_sha256": file_sha256(args.trafficlights),
        },
        "nodes": nodes,
        "edges": edges,
        "turn_restrictions": restrictions,
        "buildings": buildings,
        "signals": signals,
        "signs": signs,
        "stop_rule": {
            "type": "stop_before_sign",
            "description": "Перед знаком STOP автомобиль должен остановиться; точка контроля совпадает с координатой знака.",
        },
        "signal_stop_rule": {
            "type": "distance_before_signal",
            "description": "Среда проверяет остановку на настраиваемом расстоянии перед сигналом; стоп-линии нет.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if relations_path.exists():
        payload['sources']['building_relations'] = str(relations_path)
        payload['sources']['building_relations_sha256'] = file_sha256(relations_path)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"MapData создан: {args.output}\n"
        f"узлов: {len(nodes)}, направленных рёбер: {len(edges)}, "
        f"ограничений поворота: {len(restrictions)}, зданий: {len(buildings)}, "
        f"сигналов: {len(signals)}, знаков: {len(signs)}"
    )


if __name__ == "__main__":
    main()
