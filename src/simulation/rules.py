# Checks compliance with the rules of the road in a simulation.
"""Проверки дорожных правил, независимые от визуализации."""

from dataclasses import dataclass


@dataclass
class StopApproach:
    """Состояние одного подъезда к STOP на направленном ребре.

    distance_to_sign_m измеряется от передней части машины вдоль движения.
    Положительное значение означает, что машина ещё перед знаком.
    """

    stop_zone_m: float = 4.0
    stopped_speed_mps: float = 0.1
    previous_distance_m: float | None = None
    stopped_before_sign: bool = False
    crossed: bool = False

    def update(self, distance_to_sign_m: float, speed_mps: float) -> bool | None:
        """Возвращает True при нарушении, False при законном проезде, иначе None."""
        if speed_mps < 0:
            raise ValueError("Скорость должна быть неотрицательной")
        if self.crossed and distance_to_sign_m > self.stop_zone_m:
            self.stopped_before_sign = False
            self.crossed = False
        if 0 <= distance_to_sign_m <= self.stop_zone_m and speed_mps <= self.stopped_speed_mps:
            self.stopped_before_sign = True
        violation = None
        if (not self.crossed and self.previous_distance_m is not None
                and self.previous_distance_m > 0 >= distance_to_sign_m):
            violation = not self.stopped_before_sign
            self.crossed = True
        self.previous_distance_m = distance_to_sign_m
        return violation
