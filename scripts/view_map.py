#!/usr/bin/env python3
"""Интерактивный выбор старта А и финиша Б на карте зданий."""

from __future__ import annotations

import argparse
import json
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WINDOW_SIZE = (1280, 800)
BACKGROUND = (21, 25, 31)
BUILDING = (102, 112, 122)
BUILDING_OUTLINE = (148, 158, 169)
BUILDING_BOUNDARY = (20, 75, 230)
START = (62, 196, 121)
GOAL = (229, 83, 80)
HOVER = (255, 208, 74)
TEXT = (235, 239, 244)

# Калибровка map.png относительно локальных координат MapData.
# Она получена сопоставлением контуров зданий с их растровым отображением.
# Полная матрица local UTM -> пиксели исходного PNG, включая поворот.
# Привязка относится к data/map.png и текущему началу UTM в MapData.
MAP_IMAGE_AFFINE = np.array([
    [0.7106805682182311, -0.028956596180796622, -69.08141938596964],
    [-0.029012513160705564, -0.7118778586387634, 1615.6931612491608],
    [0.0, 0.0, 1.0],
])


@dataclass
class Camera:
    center_x_m: float
    center_y_m: float
    zoom_px_per_m: float

    def matrix(self, size):
        z = self.zoom_px_per_m
        return np.array([[z, 0, size[0]/2-z*self.center_x_m],
                         [0, -z, size[1]/2+z*self.center_y_m], [0, 0, 1]])

    def world_to_screen(self, point: tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
        return (
            round((point[0] - self.center_x_m) * self.zoom_px_per_m + size[0] / 2),
            round(size[1] / 2 - (point[1] - self.center_y_m) * self.zoom_px_per_m),
        )

    def screen_to_world(self, point: tuple[int, int], size: tuple[int, int]) -> tuple[float, float]:
        return (
            self.center_x_m + (point[0] - size[0] / 2) / self.zoom_px_per_m,
            self.center_y_m - (point[1] - size[1] / 2) / self.zoom_px_per_m,
        )

    def zoom_at(self, screen_point: tuple[int, int], factor: float, size: tuple[int, int]) -> None:
        world_point = self.screen_to_world(screen_point, size)
        self.zoom_px_per_m = min(20.0, max(0.05, self.zoom_px_per_m * factor))
        self.center_x_m = world_point[0] - (screen_point[0] - size[0] / 2) / self.zoom_px_per_m
        self.center_y_m = world_point[1] + (screen_point[1] - size[1] / 2) / self.zoom_px_per_m

    def pan(self, delta_px: tuple[int, int]) -> None:
        self.center_x_m -= delta_px[0] / self.zoom_px_per_m
        self.center_y_m += delta_px[1] / self.zoom_px_per_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-data", type=Path, default=PROJECT_ROOT / "data/prepared/map_data.json"
    )
    parser.add_argument("--map-image", type=Path, default=PROJECT_ROOT / "data/map.png")
    parser.add_argument(
        "--selection-output", type=Path, default=PROJECT_ROOT / "data/selected_points.json"
    )
    return parser.parse_args()


class RasterCamera(Camera):
    """Общая камера в ориентации PNG; клики возвращаются в локальные метры."""

    def matrix(self, size):
        return self._matrix(size, self.center_x_m, self.center_y_m, self.zoom_px_per_m)

    @staticmethod
    @lru_cache(maxsize=16)
    def _matrix(size, center_x, center_y, zoom):
        center = MAP_IMAGE_AFFINE @ [center_x, center_y, 1]
        scale = zoom / 0.711
        viewport = np.array([[scale, 0, size[0]/2 - scale*center[0]],
                             [0, scale, size[1]/2 - scale*center[1]], [0, 0, 1]])
        return viewport @ MAP_IMAGE_AFFINE

    def world_to_screen(self, point, size):
        p = self.matrix(size) @ [*point, 1]
        return round(p[0]), round(p[1])

    def screen_to_world(self, point, size):
        p = np.linalg.inv(self.matrix(size)) @ [*point, 1]
        return float(p[0]), float(p[1])

    def zoom_at(self, screen_point, factor, size):
        before = self.screen_to_world(screen_point, size)
        self.zoom_px_per_m = min(20., max(.05, self.zoom_px_per_m * factor))
        after = self.screen_to_world(screen_point, size)
        self.center_x_m += before[0] - after[0]
        self.center_y_m += before[1] - after[1]

    def pan(self, delta_px):
        delta = np.linalg.solve(self.matrix((0, 0))[:2, :2], delta_px)
        self.center_x_m -= delta[0]
        self.center_y_m -= delta[1]

    def image_rect(self, pygame, image_size, window_size):
        transform = self.matrix(window_size) @ np.linalg.inv(MAP_IMAGE_AFFINE)
        a = transform @ [0, 0, 1]
        b = transform @ [*image_size, 1]
        return pygame.Rect(round(a[0]), round(a[1]), round(b[0])-round(a[0]), round(b[1])-round(a[1]))


def load_map_data(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload.get("buildings"), list):
        raise ValueError(f"{path} не содержит список buildings")
    return payload


def building_bounds(buildings: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    points = [point for building in buildings
              for ring in building.get('outers', [building['polygon']]) for point in ring]
    if not points:
        raise ValueError("В MapData нет контуров зданий для отображения")
    min_x, max_x = min(point[0] for point in points), max(point[0] for point in points)
    min_y, max_y = min(point[1] for point in points), max(point[1] for point in points)
    return min_x, min_y, max_x, max_y


def fit_camera(buildings: list[dict[str, Any]], size: tuple[int, int]) -> Camera:
    min_x, min_y, max_x, max_y = building_bounds(buildings)
    width_m, height_m = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    zoom = min((size[0] - 80) / width_m, (size[1] - 120) / height_m)
    return Camera((min_x + max_x) / 2, (min_y + max_y) / 2, zoom)


def map_image_bounds(image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Восстанавливает границы PNG в локальных метрах, не меняя само изображение."""
    w, h = image_size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
    world = corners @ np.linalg.inv(MAP_IMAGE_AFFINE).T
    return (*world[:, :2].min(axis=0), *world[:, :2].max(axis=0))


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Проверяет принадлежность точки замкнутому контуру здания."""
    x, y = point
    inside = False
    for start, end in zip(polygon, polygon[1:]):
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= intersection_x:
                inside = not inside
    return inside


def building_at(point: tuple[float, float], buildings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Возвращает здание под курсором или точкой клика."""
    return next((b for b in buildings
                 if bounds_contain(building_geometry(b)[1], point)
                 if any(point_in_polygon(point, p) for p in b.get('outers', [b['polygon']]))
                 and not any(point_in_polygon(point, p) for p in b.get('holes', []))), None)


def building_geometry(building):
    """Массивы и bounding box вычисляются один раз при первом обращении."""
    if '_geometry' not in building:
        rings = [np.asarray(r, dtype=np.float64) for r in
                 building.get('outers', [building['polygon']]) + building.get('holes', [])]
        points = np.concatenate(rings)
        lo, hi = points.min(axis=0), points.max(axis=0)
        building['_geometry'] = rings, (lo[0], lo[1], hi[0], hi[1])
    return building['_geometry']


def bounds_contain(bounds, point):
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def visible_bounds(camera, surface):
    rect = surface.get_clip()
    corners = np.array([[rect.left, rect.top, 1], [rect.right, rect.top, 1],
                        [rect.right, rect.bottom, 1], [rect.left, rect.bottom, 1]])
    world = corners @ np.linalg.inv(camera.matrix(surface.get_size())).T
    return (*world[:, :2].min(axis=0), *world[:, :2].max(axis=0))


def save_selection(
    path: Path,
    map_data: dict[str, Any],
    start: tuple[float, float],
    goal: tuple[float, float],
    start_building: dict[str, Any] | None,
    goal_building: dict[str, Any] | None,
) -> None:
    """Сохраняет клики в локальной системе MapData для последующей привязки к дорогам."""
    payload = {
        "map_data_version": map_data.get("version"),
        "coordinate_system": "local_xy_m",
        "start": {"x_m": start[0], "y_m": start[1]},
        "goal": {"x_m": goal[0], "y_m": goal[1]},
        "start_building_osm_id": start_building["osm_id"] if start_building else None,
        "goal_building_osm_id": goal_building["osm_id"] if goal_building else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_marker(pygame: Any, surface: Any, camera: Camera, point: tuple[float, float], color: tuple[int, int, int], label: str) -> None:
    position = camera.world_to_screen(point, surface.get_size())
    pygame.draw.circle(surface, color, position, 10)
    pygame.draw.circle(surface, BACKGROUND, position, 10, 2)
    font = pygame.font.Font(None, 26)
    rendered = font.render(label, True, color)
    surface.blit(rendered, (position[0] + 13, position[1] - 12))


def draw_building_outline(
    pygame: Any,
    surface: Any,
    camera: Camera,
    building: dict[str, Any] | None,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if building is None:
        return
    matrix = camera.matrix(surface.get_size())
    for ring in building_geometry(building)[0]:
        polygon = np.rint(ring @ matrix[:2, :2].T + matrix[:2, 2]).astype(np.int32)
        if len(polygon) >= 3:
            pygame.draw.lines(surface, color, True, polygon, width)


def draw_building_boundaries(
    pygame: Any, surface: Any, camera: Camera, buildings: list[dict[str, Any]]
) -> None:
    """Рисует все контуры зданий поверх raster-подложки."""
    key = (id(buildings), surface.get_size(), tuple(surface.get_clip()),
           camera.center_x_m, camera.center_y_m, camera.zoom_px_per_m)
    if getattr(camera, '_contour_key', None) != key:
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        layer.set_clip(surface.get_clip())
        x0, y0, x1, y1 = visible_bounds(camera, surface)
        for building in buildings:
            _, (bx0, by0, bx1, by1) = building_geometry(building)
            if bx1 < x0 or by1 < y0 or bx0 > x1 or by0 > y1:
                continue
            draw_building_outline(pygame, layer, camera, building, BUILDING_BOUNDARY, 1)
        camera._contour_key, camera._contour_layer = key, layer
    surface.blit(camera._contour_layer, (0, 0))


def draw_map_image(
    pygame: Any,
    surface: Any,
    image: Any,
    camera: Camera,
    bounds: tuple[float, float, float, float],
    cached_size: tuple | None,
    cached_image: Any | None,
) -> tuple[tuple, Any]:
    """Преобразует только PNG в систему камеры; геометрия OSM неизменна.

    Рендер ограничен размером окна даже при сильном приближении.
    """
    width, height = surface.get_size()
    z = camera.zoom_px_per_m
    key = (width, height, camera.center_x_m, camera.center_y_m, z)
    if cached_image is None or cached_size != key:
        world_to_screen = np.array([
            [z, 0, width / 2 - z * camera.center_x_m],
            [0, -z, height / 2 + z * camera.center_y_m],
            [0, 0, 1],
        ])
        if isinstance(camera, RasterCamera):
            world_to_screen = camera.matrix(surface.get_size())
        transform = world_to_screen @ np.linalg.inv(MAP_IMAGE_AFFINE)
        if getattr(camera, '_source_image', None) is not image:
            camera._source_image = image
            camera._source_pixels = np.ascontiguousarray(pygame.surfarray.array3d(image).transpose(1, 0, 2))
        pixels = camera._source_pixels
        frame = cv2.warpAffine(pixels, transform[:2], (width, height),
                               flags=cv2.INTER_LINEAR, borderValue=BACKGROUND)
        cached_image = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    surface.blit(cached_image, (0, 0))
    return key, cached_image


def main() -> None:
    args = parse_args()
    map_data = load_map_data(args.map_data)
    buildings = map_data["buildings"]

    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Установите зависимости: python3 -m pip install -r requirements.txt") from error

    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Neva Drive - выбор пункта А и Б")
    clock = pygame.time.Clock()
    camera = fit_camera(buildings, window.get_size())
    map_image = pygame.image.load(args.map_image).convert() if args.map_image.exists() else None
    if map_image is not None:
        iw, ih = map_image.get_size()
        center = np.linalg.inv(MAP_IMAGE_AFFINE) @ [iw/2, ih/2, 1]
        camera = RasterCamera(center[0], center[1],
                              .711 * min((window.get_width()-80)/iw, (window.get_height()-160)/ih))
    map_bounds = map_image_bounds(map_image.get_size()) if map_image is not None else building_bounds(buildings)
    scaled_image_size: tuple | None = None
    scaled_image: Any | None = None
    start: tuple[float, float] | None = None
    goal: tuple[float, float] | None = None
    start_building: dict[str, Any] | None = None
    goal_building: dict[str, Any] | None = None
    dragging = False
    previous_mouse = (0, 0)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if map_image is not None and not camera.image_rect(pygame, map_image.get_size(), window.get_size()).collidepoint(event.pos):
                        continue
                    point = camera.screen_to_world(event.pos, window.get_size())
                    if start is None or goal is not None:
                        start, start_building = point, building_at(point, buildings)
                        goal, goal_building = None, None
                    else:
                        goal = point
                        goal_building = building_at(point, buildings)
                        save_selection(
                            args.selection_output,
                            map_data,
                            start,
                            goal,
                            start_building,
                            goal_building,
                        )
                elif event.button in {2, 3}:
                    dragging = True
                    previous_mouse = event.pos
                elif event.button == 4:
                    camera.zoom_at(event.pos, 1.2, window.get_size())
                elif event.button == 5:
                    camera.zoom_at(event.pos, 1 / 1.2, window.get_size())
            elif event.type == pygame.MOUSEBUTTONUP and event.button in {2, 3}:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                camera.pan((event.pos[0] - previous_mouse[0], event.pos[1] - previous_mouse[1]))
                previous_mouse = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom_at(pygame.mouse.get_pos(), 1.2 if event.y > 0 else 1 / 1.2, window.get_size())

        window.fill(BACKGROUND)
        if map_image is not None:
            scaled_image_size, scaled_image = draw_map_image(
                pygame,
                window,
                map_image,
                camera,
                map_bounds,
                scaled_image_size,
                scaled_image,
            )
        else:
            for building in buildings:
                polygon = [camera.world_to_screen(tuple(point), window.get_size()) for point in building["polygon"]]
                if len(polygon) >= 3:
                    pygame.draw.polygon(window, BUILDING, polygon)
                    pygame.draw.lines(window, BUILDING_OUTLINE, True, polygon, 1)

        if map_image is not None:
            window.set_clip(camera.image_rect(pygame, map_image.get_size(), window.get_size()))
            draw_building_boundaries(pygame, window, camera, buildings)

        hovered_building = building_at(camera.screen_to_world(pygame.mouse.get_pos(), window.get_size()), buildings)
        draw_building_outline(pygame, window, camera, hovered_building, HOVER, 2)
        draw_building_outline(pygame, window, camera, start_building, START, 4)
        draw_building_outline(pygame, window, camera, goal_building, GOAL, 4)

        if start is not None:
            draw_marker(pygame, window, camera, start, START, "A")
        if goal is not None:
            draw_marker(pygame, window, camera, goal, GOAL, "Б")

        window.set_clip(None)

        font = pygame.font.Font(None, 25)
        if start is None:
            message = "Левая кнопка: выберите пункт А"
        elif goal is None:
            message = "Левая кнопка: выберите пункт Б"
        else:
            message = f"А и Б сохранены в {args.selection_output.name}. Новый левый клик начнёт выбор заново"
        help_text = "Жёлтый контур - под курсором; зелёный/красный - здания А/Б"
        window.blit(font.render(message, True, TEXT), (18, 16))
        window.blit(font.render(help_text, True, TEXT), (18, 42))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
