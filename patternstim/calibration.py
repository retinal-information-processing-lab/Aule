"""Camera image -> DMD frame registration.

The DMD projects one *calibration target* (cross at the centre, ring, bar towards +x, dot in
the upper-right quadrant). The user images it with the microscope camera and aligns a template
of the target on that image (translate / rotate / scale / mirror). That alignment *is* the
calibration: a similarity transform camera -> DMD, saved as JSON with its date for reuse.

* cross  -> centring of the two planes
* ring   -> scaling
* bar    -> rotation (points towards +x of the DMD frame)
* dot    -> mirror check (upper-right quadrant: +x, -y of the logical frame)

Coordinate conventions
----------------------
* camera coords: (x, y) in camera image pixels, x = column, y = row.
* DMD coords: (x, y) in DMD frame pixels, x = column, y = row of the *logical* frame array
  (the array passed to ``BinFile.append`` before any rig flip/polarity).
Both are homogeneous 3-vectors (x, y, 1); ``matrix @ [x, y, 1]`` maps camera -> DMD.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .binfile import BinFile
from .config import Config

# ----------------------------------------------------------------------------- target geometry
# All sizes are fractions of L = min(width, height) of the DMD frame.
CROSS_ARM = 0.22        # half length of each cross arm
RING_RADIUS = 0.30
BAR_X = (0.34, 0.44)    # bar spans centre + BAR_X * L along +x
BAR_HALF_HEIGHT = 0.03
DOT_OFFSET = 0.16       # dot at centre + (+DOT_OFFSET, -DOT_OFFSET) * L
DOT_RADIUS = 0.03
LINE_FRACTION = 1 / 150  # thickness of cross / ring lines (min 3 px)

TARGET_FRAME_INDEX = 0  # frame to display in the DMD software


@dataclass(frozen=True)
class TargetGeometry:
    """Calibration target in DMD pixel coordinates (x = col, y = row)."""
    cx: float
    cy: float
    arm: float
    ring_r: float
    bar: tuple[float, float, float, float]   # x0, y0, x1, y1
    dot_c: tuple[float, float]
    dot_r: float
    thickness: float


def target_geometry(config: Config) -> TargetGeometry:
    h, w = config.dmd_shape
    L = min(h, w)
    cx, cy = w / 2, h / 2
    return TargetGeometry(
        cx=cx, cy=cy,
        arm=CROSS_ARM * L,
        ring_r=RING_RADIUS * L,
        bar=(cx + BAR_X[0] * L, cy - BAR_HALF_HEIGHT * L, cx + BAR_X[1] * L, cy + BAR_HALF_HEIGHT * L),
        dot_c=(cx + DOT_OFFSET * L, cy - DOT_OFFSET * L),
        dot_r=DOT_RADIUS * L,
        thickness=max(3.0, round(LINE_FRACTION * L)),
    )


def target_mask(config: Config) -> np.ndarray:
    """Boolean (rows, cols) mask of the calibration target, 1 = light on."""
    g = target_geometry(config)
    h, w = config.dmd_shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx + 0.5 - g.cx
    yy = yy + 0.5 - g.cy
    t2 = g.thickness / 2
    cross = ((np.abs(xx) <= t2) & (np.abs(yy) <= g.arm)) | ((np.abs(yy) <= t2) & (np.abs(xx) <= g.arm))
    r = np.hypot(xx, yy)
    ring = np.abs(r - g.ring_r) <= t2
    x0, y0, x1, y1 = g.bar
    bar = (xx + g.cx >= x0) & (xx + g.cx <= x1) & (yy + g.cy >= y0) & (yy + g.cy <= y1)
    dot = np.hypot(xx + g.cx - g.dot_c[0], yy + g.cy - g.dot_c[1]) <= g.dot_r
    return cross | ring | bar | dot


def calibration_frames(config: Config) -> list[tuple[str, np.ndarray]]:
    """Frames of the calibration bin: the target, then black."""
    return [("target", target_mask(config).astype(float)),
            ("black", np.zeros(config.dmd_shape))]


def make_calibration_bin(out_path, config: Config) -> dict:
    """Write the calibration bin and an annotated reference PNG next to it.

    Returns {"bin": path, "png": path, "frames": [names...], "target_frame": index}.
    """
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = calibration_frames(config)
    h, w = config.dmd_shape
    with BinFile(str(out_path), w, h, rig_id=config.rig_id, nb_images=len(frames), mode="w") as bf:
        for _, frame in frames:
            bf.append(frame)
    png_path = out_path.with_suffix(".png")
    imageio.imwrite(str(png_path), reference_image(config))
    return {"bin": str(out_path), "png": str(png_path),
            "frames": [n for n, _ in frames], "target_frame": TARGET_FRAME_INDEX}


def _font(size: int):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def reference_image(config: Config, annotate: bool = True) -> np.ndarray:
    """RGB uint8 image of the target (as projected) with labels explaining each element."""
    from PIL import Image, ImageDraw

    mask = target_mask(config)
    rgb = np.zeros(mask.shape + (3,), np.uint8)
    rgb[mask] = 255
    img = Image.fromarray(rgb)
    if annotate:
        g = target_geometry(config)
        draw = ImageDraw.Draw(img)
        font = _font(max(10, int(min(config.dmd_shape) / 30)))
        col = (255, 170, 0)
        draw.text((g.cx + 8, g.cy + 8), "centre", fill=col, font=font)
        draw.text((g.cx - g.ring_r * 0.7, g.cy + g.ring_r * 0.75), "ring = scale", fill=col, font=font)
        draw.text((g.bar[0], g.bar[3] + 4), "bar = +x", fill=col, font=font)
        draw.text((g.dot_c[0] + g.dot_r + 4, g.dot_c[1] - g.dot_r), "dot = upper right", fill=col, font=font)
        h, w = config.dmd_shape
        draw.rectangle([0, 0, w - 1, h - 1], outline=(120, 120, 120), width=2)
    return np.asarray(img)


# ----------------------------------------------------------------------------- similarity fit
def _rot(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _trans(x: float, y: float) -> np.ndarray:
    T = np.eye(3)
    T[0, 2], T[1, 2] = x, y
    return T


def dmd_to_camera_matrix(center_camera, scale, angle_deg, mirror, dmd_shape) -> np.ndarray:
    """DMD -> camera similarity.

    ``center_camera``: camera position (x, y) of the DMD frame centre.
    ``scale``: camera pixels per DMD pixel. ``angle_deg``: rotation (image coordinates, y down).
    ``mirror``: flip the DMD x axis about the frame centre before rotating.
    """
    h, w = dmd_shape
    cx, cy = center_camera
    Mir = np.diag([-1.0 if mirror else 1.0, 1.0, 1.0])
    S = np.diag([float(scale), float(scale), 1.0])
    return _trans(cx, cy) @ _rot(angle_deg) @ S @ Mir @ _trans(-w / 2, -h / 2)


def similarity_matrix(center_camera, scale, angle_deg, mirror, dmd_shape) -> np.ndarray:
    """Camera -> DMD matrix for the given template placement (inverse of dmd_to_camera_matrix)."""
    return np.linalg.inv(dmd_to_camera_matrix(center_camera, scale, angle_deg, mirror, dmd_shape))


def decompose_similarity(matrix, dmd_shape) -> dict:
    """Best similarity parameters for a camera -> DMD matrix (exact if it is a similarity).

    Returns {"center_camera": [x, y], "scale": camera px per DMD px, "angle_deg", "mirror"}.
    """
    h, w = dmd_shape
    A = np.linalg.inv(np.asarray(matrix, float))
    L = A[:2, :2]
    det = np.linalg.det(L)
    mirror = bool(det < 0)
    scale = float(np.sqrt(abs(det)))
    R = L @ np.diag([-1.0 if mirror else 1.0, 1.0]) / scale
    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    center = (A @ np.array([w / 2, h / 2, 1.0]))[:2]
    return {"center_camera": center.tolist(), "scale": scale, "angle_deg": angle, "mirror": mirror}


def apply_affine(matrix: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = homo @ matrix.T
    return out[:, :2]


def fit_affine(camera_pts, dmd_pts) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine mapping camera (x, y) -> DMD (x, y) from point pairs.

    Returns (matrix 3x3, residuals (N,) in DMD pixels).
    """
    cam = np.asarray(camera_pts, dtype=float)
    dmd = np.asarray(dmd_pts, dtype=float)
    if cam.shape != dmd.shape or cam.ndim != 2 or cam.shape[1] != 2:
        raise ValueError("camera_pts and dmd_pts must both be (N, 2)")
    n = len(cam)
    if n < 3:
        raise ValueError("at least 3 point pairs are needed for an affine fit")
    A = np.hstack([cam, np.ones((n, 1))])           # (N, 3)
    sol, *_ = np.linalg.lstsq(A, dmd, rcond=None)   # (3, 2)
    M = np.eye(3)
    M[:2, :] = sol.T
    if abs(np.linalg.det(M[:2, :2])) < 1e-12:
        raise ValueError("degenerate point configuration (points are collinear?)")
    pred = A @ sol
    residuals = np.linalg.norm(pred - dmd, axis=1)
    return M, residuals


# ----------------------------------------------------------------------------- image-based refinement
def _normalize(image: np.ndarray) -> np.ndarray:
    """0 = background (median), 1 = bright projected light; robust to isolated hot pixels."""
    from scipy.ndimage import median_filter

    img = np.asarray(image, np.float32)
    lo = float(np.median(img))
    hi = float(np.percentile(median_filter(img, size=3), 99.9))
    if hi <= lo:
        hi = lo + 1
    return (img - lo) / (hi - lo)


def target_sample_points(config: Config, n: int = 4000, seed: int = 0) -> np.ndarray:
    """(n, 2) DMD coords (x, y) of pixels inside the target, for scoring a placement."""
    rows, cols = np.nonzero(target_mask(config))
    pts = np.column_stack([cols + 0.5, rows + 0.5]).astype(float)
    if len(pts) > n:
        idx = np.random.default_rng(seed).choice(len(pts), n, replace=False)
        pts = pts[idx]
    return pts


def template_signal(image: np.ndarray, params: dict, config: Config, _norm=None, _pts=None) -> float:
    """Mean normalised intensity under the template (~1 = on the projected light, ~0 = background)."""
    from scipy.ndimage import map_coordinates

    norm = _normalize(image) if _norm is None else _norm
    pts = target_sample_points(config) if _pts is None else _pts
    A = dmd_to_camera_matrix(params["center_camera"], params["scale"], params["angle_deg"],
                             params["mirror"], config.dmd_shape)
    cam = apply_affine(A, pts)
    vals = map_coordinates(norm, [cam[:, 1], cam[:, 0]], order=1, mode="constant", cval=0.0)
    return float(vals.mean())


def refine_fit(image: np.ndarray, params: dict, config: Config) -> dict:
    """Locally optimise centre / scale / angle so the template sits on the bright target.

    Mirror is kept as given. Coarse-to-fine: the image is blurred first so that the basin of
    attraction is a few line widths wide, then refined on a lightly blurred image.
    Returns a new params dict with an added "signal" entry.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import minimize

    norm = _normalize(image)
    pts = target_sample_points(config)
    thickness_cam = target_geometry(config).thickness * params["scale"]
    p = dict(params)

    for sigma in (2.0 * thickness_cam, 0.5 * thickness_cam):
        blurred = gaussian_filter(norm, sigma) if sigma > 0.3 else norm
        blurred = blurred / max(float(blurred.max()), 1e-6)

        def cost(v):
            q = {"center_camera": [v[0], v[1]], "scale": v[2], "angle_deg": v[3], "mirror": p["mirror"]}
            return -template_signal(blurred, q, config, _norm=blurred, _pts=pts)

        x0 = np.array([p["center_camera"][0], p["center_camera"][1], p["scale"], p["angle_deg"]], float)
        step = np.array([max(2.0, sigma), max(2.0, sigma), 0.03 * p["scale"], 1.5])
        simplex = np.vstack([x0] + [x0 + np.eye(4)[i] * step[i] for i in range(4)])
        res = minimize(cost, x0, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 0.05, "fatol": 1e-5, "maxiter": 600})
        p = {"center_camera": [float(res.x[0]), float(res.x[1])], "scale": float(res.x[2]),
             "angle_deg": float(res.x[3]), "mirror": p["mirror"]}
    p["signal"] = template_signal(image, p, config, _norm=norm, _pts=pts)
    return p


# ----------------------------------------------------------------------------- Calibration
@dataclass
class Calibration:
    matrix: np.ndarray                      # camera -> DMD, 3x3
    method: str = "template"                # "template" | "points" | "identity"
    fit: dict = field(default_factory=dict)  # template placement (center_camera, scale, angle_deg, mirror, signal)
    rms_residual_px: float = 0.0            # in DMD pixels (points method only)
    camera_points: list = field(default_factory=list)
    dmd_points: list = field(default_factory=list)
    image_shape: tuple = ()                 # (rows, cols) of the camera image used
    image_path: str = ""
    created: str = ""                       # ISO date-time of the calibration
    pixel_size_um: float = 2.5
    dmd_shape: tuple = (672, 672)
    path: str = ""

    # -- geometry helpers --------------------------------------------------
    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    @property
    def scale_dmd_per_camera(self) -> float:
        """DMD pixels per camera pixel (geometric mean of the linear part)."""
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:2, :2]))))

    @property
    def rms_residual_um(self) -> float:
        return self.rms_residual_px * self.pixel_size_um

    def um_to_camera_px(self, um: float) -> float:
        return um / self.pixel_size_um / self.scale_dmd_per_camera

    def camera_px_to_um(self, px: float) -> float:
        return px * self.scale_dmd_per_camera * self.pixel_size_um

    def camera_to_dmd(self, pts) -> np.ndarray:
        return apply_affine(self.matrix, pts)

    def dmd_to_camera(self, pts) -> np.ndarray:
        return apply_affine(self.inverse, pts)

    def dmd_field_in_camera(self) -> np.ndarray:
        """Corners of the DMD frame mapped into camera coords (4, 2), closed order."""
        h, w = self.dmd_shape
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        return self.dmd_to_camera(corners)

    def similarity_params(self) -> dict:
        return decompose_similarity(self.matrix, self.dmd_shape)

    # -- date helpers -------------------------------------------------------
    @property
    def created_datetime(self) -> _dt.datetime | None:
        try:
            return _dt.datetime.fromisoformat(self.created)
        except (TypeError, ValueError):
            return None

    @property
    def age_days(self) -> float | None:
        d = self.created_datetime
        return None if d is None else (_dt.datetime.now() - d).total_seconds() / 86400

    def created_text(self) -> str:
        d = self.created_datetime
        if d is None:
            return self.created or "unknown date"
        age = self.age_days
        if age < 1:
            when = "today"
        elif age < 2:
            when = "yesterday"
        else:
            when = f"{int(age)} days ago"
        return f"{d:%Y-%m-%d %H:%M} ({when})"

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_template_fit(cls, params: dict, config: Config, image_shape=(), image_path="") -> "Calibration":
        """Calibration from a template placement {"center_camera", "scale", "angle_deg", "mirror"[, "signal"]}."""
        M = similarity_matrix(params["center_camera"], params["scale"], params["angle_deg"],
                              params["mirror"], config.dmd_shape)
        fit = {"center_camera": [float(v) for v in params["center_camera"]],
               "scale": float(params["scale"]), "angle_deg": float(params["angle_deg"]),
               "mirror": bool(params["mirror"])}
        if "signal" in params and params["signal"] is not None:
            fit["signal"] = float(params["signal"])
        return cls(matrix=M, method="template", fit=fit,
                   image_shape=tuple(int(v) for v in image_shape), image_path=str(image_path),
                   created=_dt.datetime.now().isoformat(timespec="seconds"),
                   pixel_size_um=config.pixel_size_um, dmd_shape=tuple(config.dmd_shape))

    @classmethod
    def from_points(cls, camera_pts, dmd_pts, config: Config, image_shape=(), image_path="") -> "Calibration":
        """Affine calibration from matched point pairs (general affine, not used by the GUI)."""
        M, res = fit_affine(camera_pts, dmd_pts)
        return cls(matrix=M, method="points",
                   rms_residual_px=float(np.sqrt(np.mean(res ** 2))),
                   camera_points=np.asarray(camera_pts, float).tolist(),
                   dmd_points=np.asarray(dmd_pts, float).tolist(),
                   image_shape=tuple(int(v) for v in image_shape),
                   image_path=str(image_path),
                   created=_dt.datetime.now().isoformat(timespec="seconds"),
                   pixel_size_um=config.pixel_size_um,
                   dmd_shape=tuple(config.dmd_shape))

    @classmethod
    def identity(cls, config: Config) -> "Calibration":
        """Camera image already in DMD space (1 camera px = 1 DMD px)."""
        return cls(matrix=np.eye(3), method="identity", pixel_size_um=config.pixel_size_um,
                   dmd_shape=tuple(config.dmd_shape), created="identity")

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "matrix": np.asarray(self.matrix).tolist(),
            "method": self.method,
            "fit": self.fit,
            "rms_residual_px": self.rms_residual_px,
            "rms_residual_um": self.rms_residual_um,
            "camera_points": self.camera_points,
            "dmd_points": self.dmd_points,
            "image_shape": list(self.image_shape),
            "image_path": self.image_path,
            "created": self.created,
            "pixel_size_um": self.pixel_size_um,
            "dmd_shape": list(self.dmd_shape),
        }

    @classmethod
    def from_dict(cls, d: dict, path: str = "") -> "Calibration":
        return cls(matrix=np.asarray(d["matrix"], dtype=float),
                   method=d.get("method", "points" if d.get("camera_points") else "template"),
                   fit=dict(d.get("fit", {})),
                   rms_residual_px=float(d.get("rms_residual_px", 0.0)),
                   camera_points=d.get("camera_points", []),
                   dmd_points=d.get("dmd_points", []),
                   image_shape=tuple(d.get("image_shape", ())),
                   image_path=d.get("image_path", ""),
                   created=d.get("created", ""),
                   pixel_size_um=float(d.get("pixel_size_um", 2.5)),
                   dmd_shape=tuple(d.get("dmd_shape", (672, 672))),
                   path=path)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self.path = str(path)
        return path

    @classmethod
    def load(cls, path) -> "Calibration":
        with open(path) as f:
            return cls.from_dict(json.load(f), path=str(path))
