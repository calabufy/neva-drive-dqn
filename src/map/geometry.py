"""Геометрические функции для подготовленной дорожной карты."""

from __future__ import annotations

from math import atan2, hypot


def segment_length_m(start: tuple[float, float], end: tuple[float, float]) -> float:
    return hypot(end[0] - start[0], end[1] - start[1])


def segment_heading_rad(start: tuple[float, float], end: tuple[float, float]) -> float:
    return atan2(end[1] - start[1], end[0] - start[0])


def road_width_m(lane_count: int) -> float:
    """Оценка ширины дорожного коридора без выдумывания разметки полос."""
    return max(1, lane_count) * 3.25
