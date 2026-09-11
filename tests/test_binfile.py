import numpy as np
import pytest

from patternstim.binfile import BinFile, read_header, read_raw_frame_bytes, apply_write_transform
from tests.conftest import EXISTING_BIN


def test_existing_bin_header():
    h = read_header(EXISTING_BIN)
    assert h == {"xsize": 672, "ysize": 672, "nb_images": 15, "nb_bits": 8}


def test_existing_bin_first_frames_match_scripts():
    with BinFile(str(EXISTING_BIN), rig_id=3, mode="r") as bf:
        black = bf.read_frame_uint8(0)
        white = bf.read_frame_uint8(1)
        grey = bf.read_frame_uint8(2)
    assert black.shape == (672, 672)
    assert black.max() == 0
    assert white.min() == 255
    assert grey.min() == grey.max() == 128


def test_raw_bytes_equal_read_frame_as_bytes():
    raw = read_raw_frame_bytes(EXISTING_BIN, 3)
    with BinFile(str(EXISTING_BIN), rig_id=3, mode="r") as bf:
        assert bf.read_frame_as_bytes(3) == raw
    assert len(raw) == 672 * 672


def test_write_read_roundtrip(tmp_path):
    h, w = 40, 50
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 2, size=(h, w)).astype(float) for _ in range(3)]
    path = tmp_path / "t.bin"
    with BinFile(str(path), w, h, rig_id=3, nb_images=3, mode="w") as bf:
        for f in frames:
            bf.append(f)
    assert read_header(path) == {"xsize": w, "ysize": h, "nb_images": 3, "nb_bits": 8}
    with BinFile(str(path), rig_id=3, mode="r") as bf:
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(bf.read_frame_uint8(i), (f * 255).astype(np.uint8))
        # stored bytes are inverted + flipped for rig 3
        stored = np.frombuffer(bf.read_frame_as_bytes(0), dtype=np.uint8).reshape(h, w)
        np.testing.assert_array_equal(stored, apply_write_transform(frames[0], 3))
        np.testing.assert_array_equal(stored, np.fliplr(255 - (frames[0] * 255)).astype(np.uint8))


def test_read_frame_legacy_api_works_on_numpy_124():
    with BinFile(str(EXISTING_BIN), rig_id=3, mode="r") as bf:
        f = bf.read_frame(1)
    assert f.dtype == float
    assert f.min() == f.max() == 1.0  # white frame, rig-3 polarity undone


def test_append_bytes_verbatim(tmp_path):
    h, w = 4, 6
    raw = bytes(range(h * w))
    path = tmp_path / "b.bin"
    with BinFile(str(path), w, h, rig_id=3, nb_images=1, mode="w") as bf:
        bf.append(raw)
    assert read_raw_frame_bytes(path, 0) == raw


def test_append_wrong_shape_raises(tmp_path):
    with BinFile(str(tmp_path / "c.bin"), 6, 4, rig_id=3, nb_images=1, mode="w") as bf:
        with pytest.raises(AssertionError):
            bf.append(np.zeros((6, 4)))
