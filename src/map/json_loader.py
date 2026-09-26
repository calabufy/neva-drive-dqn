"""Загрузка исходных OSM-выгрузок и программ светофоров."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_osm(path: Path) -> list[dict[str, Any]]:
    """Возвращает объекты OSM из JSON-ответа Overpass."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError(f"{path} не содержит список elements из Overpass JSON")
    return elements


def index_nodes(elements: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Индексирует OSM-узлы с координатами по их идентификаторам."""
    return {
        element["id"]: element
        for element in elements
        if element.get("type") == "node"
        and isinstance(element.get("id"), int)
        and isinstance(element.get("lat"), (int, float))
        and isinstance(element.get("lon"), (int, float))
    }


def load_trafficlights(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Читает фазы светофоров, сопоставляя их с OSM node ID."""
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    result: dict[int, list[dict[str, Any]]] = {}
    for signal in payload.get("trafficlights", []):
        if signal.get("osm_type") != "node":
            continue
        osm_id = signal.get("osm_id")
        phases = signal.get("phases")
        if not isinstance(osm_id, int) or not isinstance(phases, list):
            continue
        result[osm_id] = phases
    return result
