"""Rig / DMD configuration shared by all modules."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".patternstim" / "config.json"


@dataclass
class Config:
    rig_id: int = 3
    dmd_height: int = 672          # rows of the frame array
    dmd_width: int = 672           # columns of the frame array
    pixel_size_um: float = 2.5     # size of one DMD frame pixel on the sample
    last_calibration: str = ""     # path of the last saved calibration JSON
    calibration_max_age_days: int = 30   # warn in the GUI when the calibration is older
    last_dir: str = ""             # last directory used in file dialogs
    path: str = field(default="", compare=False)   # where this config is saved (not serialised)

    @property
    def dmd_shape(self) -> tuple[int, int]:
        return (self.dmd_height, self.dmd_width)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "path"}
        return cls(**known)

    def save(self, path: os.PathLike | str | None = None) -> Path:
        """Save to ``path``, else to where it was loaded from, else the default location."""
        path = Path(path or self.path or DEFAULT_CONFIG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self.path = str(path)
        return path

    @classmethod
    def load(cls, path: os.PathLike | str = DEFAULT_CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls(path=str(path))
        with open(path) as f:
            cfg = cls.from_dict(json.load(f))
        cfg.path = str(path)
        return cfg
