"""Shapes drawn on the camera image and their rasterisation into DMD masks."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from skimage.draw import polygon as _draw_polygon
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


@dataclass
class Circle:
    cx: float
    cy: float
    r: float
    name: str = ""
    enabled: bool = True
    kind: str = field(default="circle", init=False)

    def to_polygon(self, n: int = 64) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([self.cx + self.r * np.cos(t), self.cy + self.r * np.sin(t)])

    def to_dict(self) -> dict:
        return {"kind": "circle", "name": self.name, "enabled": self.enabled,
                "cx": float(self.cx), "cy": float(self.cy), "r": float(self.r)}


@dataclass
class Polygon:
    points: np.ndarray            # (N, 2) camera coords (x, y)
    name: str = ""
    enabled: bool = True
    kind: str = field(default="polygon", init=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 2)

    def to_polygon(self, n: int = 64) -> np.ndarray:
        return self.points

    def to_dict(self) -> dict:
        return {"kind": "polygon", "name": self.name, "enabled": self.enabled,
                "points": self.points.tolist()}


Shape = Circle | Polygon


def shape_from_dict(d: dict) -> Shape:
    if d["kind"] == "circle":
        return Circle(d["cx"], d["cy"], d["r"], name=d.get("name", ""), enabled=d.get("enabled", True))
    if d["kind"] == "polygon":
        return Polygon(np.asarray(d["points"], float), name=d.get("name", ""), enabled=d.get("enabled", True))
    raise ValueError(f"unknown shape kind {d['kind']!r}")


def simplify_polyline(points: Sequence[Sequence[float]], tolerance: float = 1.0) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an open polyline (keeps endpoints)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        seg = p1 - p0
        seg_len = np.hypot(*seg)
        mid = pts[i0 + 1:i1]
        if seg_len < 1e-12:
            d = np.linalg.norm(mid - p0, axis=1)
        else:
            d = np.abs(np.cross(seg, mid - p0)) / seg_len
        j = int(np.argmax(d))
        if d[j] > tolerance:
            idx = i0 + 1 + j
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return pts[keep]


def transform_points(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (homo @ matrix.T)[:, :2]


def rasterize_polygon(dmd_pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill a polygon given in DMD (x, y) coords into a bool mask of ``shape`` (rows, cols)."""
    mask = np.zeros(shape, dtype=bool)
    if len(dmd_pts) < 3:
        return mask
    rr, cc = _draw_polygon(dmd_pts[:, 1], dmd_pts[:, 0], shape=shape)
    mask[rr, cc] = True
    return mask


def rasterize_shape(shape_obj: Shape, matrix: np.ndarray, dmd_shape: tuple[int, int],
                    margin_px: float = 0.0) -> np.ndarray:
    """Rasterise one shape (camera coords) to a DMD mask through the affine ``matrix``."""
    poly_cam = shape_obj.to_polygon()
    poly_dmd = transform_points(matrix, poly_cam)
    mask = rasterize_polygon(poly_dmd, dmd_shape)
    if margin_px > 0 and mask.any():
        mask = binary_dilation(mask, structure=disk(int(round(margin_px))))
    return mask


def rasterize_shapes(shapes: Iterable[Shape], matrix: np.ndarray, dmd_shape: tuple[int, int],
                     margin_px: float = 0.0) -> np.ndarray:
    """Union of all *enabled* shapes."""
    out = np.zeros(dmd_shape, dtype=bool)
    for s in shapes:
        if s.enabled:
            out |= rasterize_shape(s, matrix, dmd_shape, margin_px)
    return out


def shape_in_field(shape_obj: Shape, matrix: np.ndarray, dmd_shape: tuple[int, int]) -> bool:
    """True if all polygon vertices fall inside the DMD frame."""
    pts = transform_points(matrix, shape_obj.to_polygon())
    h, w = dmd_shape
    return bool(np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)))
