import json
import shutil
import numpy as np
import pytest

from aule.binfile import BinFile, read_header, read_raw_frame_bytes
from aule.calibration import Calibration
from aule.cli import main
from aule.geometry import Circle
from aule.merge import merge_bin, plan_merge
from aule.patterns import PatternSet, load_masks
from tests.conftest import EXISTING_BIN, EXISTING_BIN_2


@pytest.fixture
def pattern_npz(config, tmp_path):
    ps = PatternSet(name="cells", calibration=Calibration.identity(config), config=config)
    ps.add(Circle(100, 200, 15))
    ps.add(Circle(400, 350, 20))
    res = ps.export(tmp_path / "pat", combined=True, per_shape=True, previews=False)
    return res["npz"]


def test_merge_into_existing_bin(config, pattern_npz, tmp_path):
    src = tmp_path / "orig.bin"
    shutil.copy(EXISTING_BIN, src)
    out = tmp_path / "merged.bin"
    m = merge_bin(src, [pattern_npz], out)

    assert read_header(out) == {"xsize": 672, "ysize": 672, "nb_images": 15 + 6, "nb_bits": 8}
    assert m["n_original_frames"] == 15 and m["n_total_frames"] == 21
    assert list(m["frames"]) == ["cells_all", "cells_S1", "cells_S2"]
    assert m["frames"]["cells_all"] == {"pos_idx": 15, "neg_idx": 16,
                                        "n_pixels": m["frames"]["cells_all"]["n_pixels"],
                                        "pattern_file": str(pattern_npz)}
    assert m["frames"]["cells_S2"]["neg_idx"] == 20

    # original untouched and copied byte for byte
    assert src.read_bytes() == EXISTING_BIN.read_bytes()
    for i in range(15):
        assert read_raw_frame_bytes(out, i) == read_raw_frame_bytes(src, i)

    # appended frames: pos == mask, neg == ~mask, after undoing the rig transform
    names, masks, _ = load_masks(pattern_npz)
    with BinFile(str(out), rig_id=3, mode="r") as bf:
        for k, (name, mask) in enumerate(zip(names, masks)):
            pos = bf.read_frame_uint8(15 + 2 * k)
            neg = bf.read_frame_uint8(16 + 2 * k)
            np.testing.assert_array_equal(pos, mask * 255)
            np.testing.assert_array_equal(neg, (1 - mask) * 255)
            np.testing.assert_array_equal(pos.astype(int) + neg, 255)
    with open(m["indices_json"]) as f:
        j = json.load(f)
    assert j["frames"]["cells_S1"]["pos_idx"] == 17


def test_merge_same_patterns_into_two_bins_identical_bytes(pattern_npz, tmp_path):
    out1 = tmp_path / "m1.bin"
    out2 = tmp_path / "m2.bin"
    merge_bin(EXISTING_BIN, [pattern_npz], out1)
    merge_bin(EXISTING_BIN_2, [pattern_npz], out2)
    for i in range(15, 21):
        assert read_raw_frame_bytes(out1, i) == read_raw_frame_bytes(out2, i)


def test_merge_refuses_overwrite_and_shape_mismatch(config, small_config, pattern_npz, tmp_path):
    with pytest.raises(ValueError):
        merge_bin(EXISTING_BIN, [pattern_npz], EXISTING_BIN)
    ps = PatternSet(name="tiny", calibration=Calibration.identity(small_config), config=small_config)
    ps.add(Circle(10, 10, 3))
    tiny = ps.export(tmp_path / "tiny", previews=False)["npz"]
    with pytest.raises(ValueError, match="does not match"):
        plan_merge(EXISTING_BIN, [tiny])
    assert not (tmp_path / "never.bin").exists()


def test_cli_merge_and_info(pattern_npz, tmp_path, capsys):
    out = tmp_path / "cli.bin"
    assert main(["merge", "--bin", str(EXISTING_BIN), "--patterns", pattern_npz, "--out", str(out)]) == 0
    assert read_header(out)["nb_images"] == 21
    assert main(["info", str(out)]) == 0
    assert "21 frames" in capsys.readouterr().out


def test_cli_calib_bin(tmp_path):
    out = tmp_path / "calib.bin"
    assert main(["calib-bin", "--out", str(out), "--config", str(tmp_path / "nocfg.json")]) == 0
    assert read_header(out)["nb_images"] == 2
