"""Loading microscope images (TIFF 8/16-bit, PNG, JPG) as float 2D arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path) -> tuple[list[np.ndarray], list[str]]:
    """Return (channels, channel_names). Each channel is a 2D float32 array (rows, cols).

    * 2D image -> one channel.
    * RGB/RGBA (last dim 3 or 4) -> one channel per colour + a luminance channel.
    * stack with a small first dim (<= 6) -> treated as channels.
    * larger stacks -> max projection along the first axis.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
    else:
        import imageio.v2 as imageio
        arr = imageio.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return [arr.astype(np.float32)], ["image"]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            chans = [rgb[..., 0], rgb[..., 1], rgb[..., 2], rgb.mean(axis=-1)]
            return chans, ["red", "green", "blue", "luminance"]
        if arr.shape[0] <= 6:
            return [arr[i].astype(np.float32) for i in range(arr.shape[0])], \
                   [f"channel {i}" for i in range(arr.shape[0])]
        return [arr.max(axis=0).astype(np.float32)], [f"max projection ({arr.shape[0]} planes)"]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        return load_image_array(arr.max(axis=0))
    raise ValueError(f"unsupported image shape {arr.shape}")


def load_image_array(arr) -> tuple[list[np.ndarray], list[str]]:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        return [rgb[..., 0], rgb[..., 1], rgb[..., 2], rgb.mean(axis=-1)], ["red", "green", "blue", "luminance"]
    return [arr.astype(np.float32)], ["image"]


def warp_to_dmd(image: np.ndarray, matrix: np.ndarray, dmd_shape: tuple[int, int]) -> np.ndarray:
    """Resample a camera image into DMD frame space using the camera->DMD affine ``matrix``."""
    from skimage.transform import AffineTransform, warp
    inv = np.linalg.inv(matrix)                 # DMD -> camera (x, y)
    tf = AffineTransform(matrix=inv)
    out = warp(image.astype(np.float64), tf, output_shape=dmd_shape, order=1,
               preserve_range=True, cval=0.0)
    return out.astype(np.float32)


def to_uint8(image: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    """Contrast-stretch to uint8 using [low, high] (defaults: 1st / 99.5th percentile)."""
    img = np.asarray(image, dtype=np.float32)
    if low is None or high is None:
        lo, hi = np.percentile(img, [1, 99.5])
        low = lo if low is None else low
        high = hi if high is None else high
    if high <= low:
        high = low + 1
    return np.clip((img - low) / (high - low) * 255, 0, 255).astype(np.uint8)
