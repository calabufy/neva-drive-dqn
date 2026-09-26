"""Отбор и нормализация OSM-объектов для дорожной карты."""

from __future__ import annotations

import re
from typing import Any


DRIVABLE_HIGHWAYS = {
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}


def is_drivable_way(element: dict[str, Any]) -> bool:
    tags = element.get("tags", {})
    return (
        element.get("type") == "way"
        and tags.get("highway") in DRIVABLE_HIGHWAYS
        and len(element.get("nodes", [])) >= 2
        and tags.get("access") not in {"no", "private"}
        and tags.get("motor_vehicle") not in {"no", "private"}
    )


def lane_count(tags: dict[str, str]) -> int:
    raw_value = tags.get("lanes")
    if raw_value:
        match = re.search(r"\d+", raw_value)
        if match:
            return max(1, int(match.group()))
    return 1 if tags.get("oneway") in {"yes", "1", "-1"} else 2


def speed_limit_mps(tags: dict[str, str]) -> float:
    raw_value = tags.get("maxspeed", "")
    match = re.search(r"\d+(?:\.\d+)?", raw_value)
    if match:
        return float(match.group()) / 3.6
    # RU:urban и отсутствие maxspeed в первой версии трактуются как 60 км/ч.
    return 60.0 / 3.6


def travel_directions(tags: dict[str, str]) -> tuple[bool, bool]:
    """Возвращает допустимость прямого и обратного направлений OSM way."""
    oneway = tags.get("oneway", "").lower()
    if oneway in {"yes", "1", "true"}:
        return True, False
    if oneway == "-1":
        return False, True
    return True, True
