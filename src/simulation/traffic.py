"""Детерминированные фазы светофоров по модельному времени."""

from math import ceil
from typing import Any


def signal_phase(phases: list[dict[str, Any]], elapsed_s: float) -> tuple[str, int]:
    """Текущий цвет и целое число секунд до следующей фазы."""
    if not phases:
        raise ValueError("У светофора нет фаз")
    if elapsed_s < 0:
        raise ValueError("Время не может быть отрицательным")
    durations = [float(phase["duration_s"]) for phase in phases]
    if any(duration <= 0 for duration in durations):
        raise ValueError("Длительность фазы должна быть положительной")
    within_cycle = elapsed_s % sum(durations)
    for phase, duration in zip(phases, durations):
        if within_cycle < duration:
            return str(phase["state"]), ceil(duration - within_cycle)
        within_cycle -= duration
    raise RuntimeError("Не удалось определить фазу светофора")
