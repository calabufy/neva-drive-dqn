"""Преобразование WGS 84 в локальную метрическую систему карты."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer


@dataclass(frozen=True)
class LocalProjection:
    """EPSG:32636 с началом координат в юго-западной точке карты."""

    origin_easting_m: float
    origin_northing_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_transformer",
            Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True),
        )

    def project_utm(self, latitude: float, longitude: float) -> tuple[float, float]:
        easting, northing = self._transformer.transform(longitude, latitude)
        return float(easting), float(northing)

    def project(self, latitude: float, longitude: float) -> tuple[float, float]:
        easting, northing = self.project_utm(latitude, longitude)
        return easting - self.origin_easting_m, northing - self.origin_northing_m


def projection_from_nodes(nodes: dict[int, dict]) -> LocalProjection:
    """Создаёт детерминированное начало локальной системы по минимумам UTM."""
    if not nodes:
        raise ValueError("Невозможно создать проекцию без OSM-узлов")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    projected = [
        transformer.transform(node["lon"], node["lat"])
        for node in nodes.values()
    ]
    return LocalProjection(
        origin_easting_m=min(point[0] for point in projected),
        origin_northing_m=min(point[1] for point in projected),
    )
