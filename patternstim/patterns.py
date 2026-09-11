"""Pattern sets: shapes drawn on a camera image -> DMD masks, saved for merging.

A pattern set is saved as two files with the same stem:
  <name>.patterns.npz   masks (uint8, N x H x W) + frame names  -> used by merge (source of truth)
  <name>.patterns.json  shapes in camera coords, calibration, config, provenance -> re-editable
plus a preview PNG per frame in <name>_previews/.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .config import Config
from .geometry import Shape, rasterize_shape, rasterize_shapes, shape_from_dict

NPZ_SUFFIX = ".patterns.npz"
JSON_SUFFIX = ".patterns.json"


def _safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name or "pattern"


def pattern_stem(path) -> Path:
    """Strip .patterns.npz / .patterns.json / .npz / .json from a path."""
    p = Path(path)
    s = str(p)
    for suf in (NPZ_SUFFIX, JSON_SUFFIX, ".npz", ".json"):
        if s.endswith(suf):
            return Path(s[: -len(suf)])
    return p


@dataclass
class PatternSet:
    name: str = "pattern"
    shapes: list = field(default_factory=list)          # list[Shape], camera coords
    calibration: Calibration | None = None
    config: Config = field(default_factory=Config)
    image_path: str = ""
    margin_um: float = 0.0
    export_combined: bool = True
    export_per_shape: bool = False

    # -- shape helpers ----------------------------------------------------
    def next_name(self) -> str:
        used = {s.name for s in self.shapes}
        i = len(self.shapes) + 1
        while f"S{i}" in used:
            i += 1
        return f"S{i}"

    def add(self, shape: Shape) -> Shape:
        if not shape.name:
            shape.name = self.next_name()
        self.shapes.append(shape)
        return shape

    @property
    def margin_px(self) -> float:
        return self.margin_um / self.config.pixel_size_um

    def _matrix(self) -> np.ndarray:
        if self.calibration is None:
            raise ValueError("no calibration loaded")
        return self.calibration.matrix

    def combined_mask(self) -> np.ndarray:
        return rasterize_shapes(self.shapes, self._matrix(), self.config.dmd_shape, self.margin_px)

    def shape_mask(self, shape: Shape) -> np.ndarray:
        return rasterize_shape(shape, self._matrix(), self.config.dmd_shape, self.margin_px)

    def build_frames(self, combined: bool | None = None, per_shape: bool | None = None) -> list[tuple[str, np.ndarray]]:
        """Ordered (frame_name, bool mask) list. Negatives are made at merge time."""
        combined = self.export_combined if combined is None else combined
        per_shape = self.export_per_shape if per_shape is None else per_shape
        base = _safe(self.name)
        frames: list[tuple[str, np.ndarray]] = []
        enabled = [s for s in self.shapes if s.enabled]
        if combined:
            frames.append((f"{base}_all", self.combined_mask()))
        if per_shape:
            for s in enabled:
                frames.append((f"{base}_{_safe(s.name)}", self.shape_mask(s)))
        return frames

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "image_path": str(self.image_path),
            "margin_um": self.margin_um,
            "export_combined": self.export_combined,
            "export_per_shape": self.export_per_shape,
            "config": self.config.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "calibration_path": self.calibration.path if self.calibration else "",
            "shapes": [s.to_dict() for s in self.shapes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatternSet":
        cal = Calibration.from_dict(d["calibration"], path=d.get("calibration_path", "")) if d.get("calibration") else None
        return cls(name=d.get("name", "pattern"),
                   shapes=[shape_from_dict(s) for s in d.get("shapes", [])],
                   calibration=cal,
                   config=Config.from_dict(d.get("config", {})),
                   image_path=d.get("image_path", ""),
                   margin_um=float(d.get("margin_um", 0.0)),
                   export_combined=bool(d.get("export_combined", True)),
                   export_per_shape=bool(d.get("export_per_shape", False)))

    def save_json(self, path) -> Path:
        """Save only the editable description (used for autosave)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def export(self, directory, combined: bool | None = None, per_shape: bool | None = None,
               previews: bool = True) -> dict:
        """Write <name>.patterns.npz + .json (+ previews). Returns paths and frame names."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / _safe(self.name)
        frames = self.build_frames(combined, per_shape)
        if not frames:
            raise ValueError("nothing to export: no frames selected or no enabled shapes")
        names = [n for n, _ in frames]
        masks = np.stack([m.astype(np.uint8) for _, m in frames], axis=0)

        npz_path = Path(str(stem) + NPZ_SUFFIX)
        np.savez_compressed(npz_path, masks=masks, names=np.array(names),
                            dmd_shape=np.array(self.config.dmd_shape),
                            pixel_size_um=np.array(self.config.pixel_size_um),
                            rig_id=np.array(self.config.rig_id))
        d = self.to_dict()
        d["frames"] = [{"name": n, "n_pixels": int(m.sum())} for n, m in frames]
        d["npz"] = npz_path.name
        json_path = Path(str(stem) + JSON_SUFFIX)
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2)

        preview_dir = None
        if previews:
            import imageio.v2 as imageio
            preview_dir = Path(str(stem) + "_previews")
            preview_dir.mkdir(exist_ok=True)
            for n, m in frames:
                imageio.imwrite(str(preview_dir / f"{n}_pos.png"), (m * 255).astype(np.uint8))
                imageio.imwrite(str(preview_dir / f"{n}_neg.png"), ((~m) * 255).astype(np.uint8))
        return {"npz": str(npz_path), "json": str(json_path),
                "previews": str(preview_dir) if preview_dir else "", "frames": names}

    @classmethod
    def load(cls, path) -> "PatternSet":
        """Load the editable description from a .patterns.json (or its npz sibling)."""
        json_path = Path(str(pattern_stem(path)) + JSON_SUFFIX)
        with open(json_path) as f:
            return cls.from_dict(json.load(f))


def load_masks(path) -> tuple[list[str], np.ndarray, dict]:
    """Read a .patterns.npz -> (names, masks uint8 N x H x W, meta)."""
    p = Path(path)
    if not str(p).endswith(".npz"):
        p = Path(str(pattern_stem(p)) + NPZ_SUFFIX)
    with np.load(p, allow_pickle=False) as z:
        masks = z["masks"].astype(np.uint8)
        names = [str(n) for n in z["names"]]
        meta = {"dmd_shape": tuple(int(v) for v in z["dmd_shape"]),
                "pixel_size_um": float(z["pixel_size_um"]),
                "rig_id": int(z["rig_id"])}
    return names, masks, meta
