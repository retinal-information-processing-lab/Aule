import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aule.config import Config  # noqa: E402
from aule.calibration import Calibration  # noqa: E402

# Lab stimulus bins (not in the repository). When absent, equivalent synthetic bins are generated:
# 15 frames of 672x672 written for rig 3, frame 0 black, 1 white, 2 grey (128), then gratings.
_LAB_BIN = ROOT / "ReversingGrating" / "Bin_Files" / "ReversingGrating_OssStim_MEA3_12SpatialPatterns.bin"
_LAB_BIN_2 = ROOT / "PulsingGratingOSS" / "Bin_Files" / "PulsingGrating_OssStim_MEA3_12SpatialPatterns.bin"


def _synthetic_lab_bin(path: Path, seed: int) -> Path:
    from aule.binfile import BinFile, apply_write_transform
    n, h, w = 15, 672, 672
    rng = np.random.default_rng(seed)
    with BinFile(str(path), w, h, rig_id=3, nb_images=n, mode="w") as bf:
        bf.append(apply_write_transform(np.zeros((h, w)), 3).tobytes())        # black
        bf.append(apply_write_transform(np.ones((h, w)), 3).tobytes())         # white
        bf.append(np.full((h, w), 255 - 128, np.uint8).tobytes())              # grey 128 after polarity undo
        for i in range(n - 3):
            period = int(rng.integers(8, 96))
            grating = ((np.arange(w) // period) % 2).astype(float)[None, :].repeat(h, axis=0)
            bf.append(grating if i % 2 == 0 else grating.T)
    return path


def _lab_bin_or_synthetic(lab_path: Path, name: str, seed: int) -> Path:
    if lab_path.exists() and not os.environ.get("AULE_TEST_SYNTHETIC_BIN"):
        return lab_path
    cache = Path(tempfile.gettempdir()) / "aule_tests"
    cache.mkdir(exist_ok=True)
    path = cache / name
    if not path.exists() or path.stat().st_size != 15 * 672 * 672 + 8:
        _synthetic_lab_bin(path, seed)
    return path


EXISTING_BIN = _lab_bin_or_synthetic(_LAB_BIN, "synthetic_reversing.bin", 1)
EXISTING_BIN_2 = _lab_bin_or_synthetic(_LAB_BIN_2, "synthetic_pulsing.bin", 2)


@pytest.fixture
def config():
    return Config(rig_id=3, dmd_height=672, dmd_width=672, pixel_size_um=2.5)


@pytest.fixture
def small_config():
    return Config(rig_id=3, dmd_height=64, dmd_width=80, pixel_size_um=2.5)


@pytest.fixture
def identity_cal(config):
    return Calibration.identity(config)


@pytest.fixture
def synthetic_matrix():
    """Camera -> DMD: flip x, rotate 10 deg, scale 0.4, translate."""
    th = np.deg2rad(10)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    F = np.diag([-1.0, 1.0])
    A = 0.4 * R @ F
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = [500.0, -20.0]
    return M


@pytest.fixture
def template_matrix():
    """Camera -> DMD similarity with the whole target visible in a 1000x1200 camera image."""
    from aule.calibration import similarity_matrix
    return similarity_matrix((610.0, 480.0), 1.15, 10.0, True, (672, 672))


def synthetic_camera_image(matrix_cam_to_dmd, shape=(480, 640), noise=0.0, seed=0, config=None):
    """Camera image of the projected calibration target, seen through the given camera -> DMD matrix.

    Returns (image float32 (rows, cols), template params of the truth).
    """
    from scipy.ndimage import map_coordinates, gaussian_filter
    from aule.calibration import target_mask, decompose_similarity

    config = config or Config()
    mask = target_mask(config).astype(np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5, np.ones(xx.size)])
    dmd = (pts @ matrix_cam_to_dmd.T)[:, :2] - 0.5
    img = map_coordinates(mask, [dmd[:, 1], dmd[:, 0]], order=1, mode="constant", cval=0.0)
    img = gaussian_filter(img.reshape(shape), 0.7) * 3000 + 200
    if noise:
        img += np.random.default_rng(seed).normal(0, noise * 3000, shape)
    return np.clip(img, 0, 65535).astype(np.float32), decompose_similarity(matrix_cam_to_dmd, config.dmd_shape)
