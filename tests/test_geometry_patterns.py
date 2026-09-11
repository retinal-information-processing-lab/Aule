import json
import numpy as np
import pytest

from aule.calibration import Calibration
from aule.geometry import (Circle, Polygon, rasterize_shape, rasterize_shapes,
                                  simplify_polyline, shape_in_field, shape_from_dict)
from aule.patterns import PatternSet, load_masks, pattern_stem


def test_circle_rasterizes_to_expected_area(small_config):
    cal = Calibration.identity(small_config)
    c = Circle(40, 30, 10)
    m = rasterize_shape(c, cal.matrix, small_config.dmd_shape)
    assert m.shape == (64, 80)
    assert m[30, 40]
    assert abs(m.sum() - np.pi * 100) < 15


def test_margin_dilates(small_config):
    cal = Calibration.identity(small_config)
    c = Circle(40, 30, 5)
    m0 = rasterize_shape(c, cal.matrix, small_config.dmd_shape)
    m1 = rasterize_shape(c, cal.matrix, small_config.dmd_shape, margin_px=3)
    assert m1.sum() > m0.sum()
    assert np.all(m1[m0])


def test_polygon_and_transform(small_config, synthetic_matrix):
    cal = Calibration.identity(small_config)
    sq = Polygon([[10, 10], [30, 10], [30, 20], [10, 20]])
    m = rasterize_shape(sq, cal.matrix, small_config.dmd_shape)
    assert m.sum() == pytest.approx(20 * 10, abs=40)
    # through an affine the square lands where the matrix says
    M = np.eye(3); M[:2, :2] = [[0.5, 0], [0, 0.5]]; M[:2, 2] = [20, 20]
    m2 = rasterize_shape(sq, M, small_config.dmd_shape)
    rows, cols = np.nonzero(m2)
    assert 25 <= cols.min() and cols.max() <= 35
    assert 25 <= rows.min() and rows.max() <= 30


def test_out_of_field_clipped_and_reported(small_config):
    cal = Calibration.identity(small_config)
    c = Circle(78, 10, 5)
    m = rasterize_shape(c, cal.matrix, small_config.dmd_shape)
    assert m.sum() > 0
    assert not shape_in_field(c, cal.matrix, small_config.dmd_shape)
    assert shape_in_field(Circle(40, 30, 5), cal.matrix, small_config.dmd_shape)


def test_disabled_shapes_skipped(small_config):
    cal = Calibration.identity(small_config)
    a = Circle(20, 20, 5)
    b = Circle(60, 40, 5, enabled=False)
    m = rasterize_shapes([a, b], cal.matrix, small_config.dmd_shape)
    assert m[20, 20] and not m[40, 60]


def test_simplify_polyline():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    pts = np.column_stack([50 + 20 * np.cos(t), 50 + 20 * np.sin(t)])
    s = simplify_polyline(pts, tolerance=0.5)
    assert 8 < len(s) < 100
    assert np.allclose(s[0], pts[0]) and np.allclose(s[-1], pts[-1])
    assert len(simplify_polyline([[0, 0], [1, 0]])) == 2


def test_shape_dict_roundtrip():
    c = Circle(1, 2, 3, name="a", enabled=False)
    p = Polygon([[0, 0], [1, 0], [0, 1]], name="b")
    for s in (c, p):
        s2 = shape_from_dict(json.loads(json.dumps(s.to_dict())))
        assert s2.to_dict() == s.to_dict()


def test_patternset_frames_and_export(small_config, tmp_path):
    ps = PatternSet(name="cells 1", calibration=Calibration.identity(small_config), config=small_config,
                    image_path="x.tif")
    ps.add(Circle(20, 20, 5))
    ps.add(Circle(60, 40, 6))
    ps.add(Polygon([[5, 50], [15, 50], [10, 60]]))
    assert [s.name for s in ps.shapes] == ["S1", "S2", "S3"]

    assert [n for n, _ in ps.build_frames(combined=True, per_shape=False)] == ["cells_1_all"]
    frames = ps.build_frames(combined=True, per_shape=True)
    assert [n for n, _ in frames] == ["cells_1_all", "cells_1_S1", "cells_1_S2", "cells_1_S3"]
    np.testing.assert_array_equal(frames[0][1], frames[1][1] | frames[2][1] | frames[3][1])

    with pytest.raises(ValueError):
        ps.export(tmp_path, combined=False, per_shape=False)

    res = ps.export(tmp_path, combined=True, per_shape=True)
    names, masks, meta = load_masks(res["npz"])
    assert names == [n for n, _ in frames]
    assert masks.shape == (4, 64, 80) and masks.dtype == np.uint8
    assert meta == {"dmd_shape": (64, 80), "pixel_size_um": 2.5, "rig_id": 3}
    assert (tmp_path / "cells_1_previews" / "cells_1_all_pos.png").exists()
    assert (tmp_path / "cells_1_previews" / "cells_1_all_neg.png").exists()

    # reload for editing gives the same masks
    ps2 = PatternSet.load(res["json"])
    assert ps2.name == "cells 1" and len(ps2.shapes) == 3 and ps2.image_path == "x.tif"
    np.testing.assert_array_equal(ps2.combined_mask(), ps.combined_mask())
    assert pattern_stem(res["npz"]) == pattern_stem(res["json"])


def test_export_twice_is_bit_identical(small_config, tmp_path):
    ps = PatternSet(name="p", calibration=Calibration.identity(small_config), config=small_config)
    ps.add(Circle(30, 30, 7))
    a = ps.export(tmp_path / "a", previews=False)
    b = ps.export(tmp_path / "b", previews=False)
    np.testing.assert_array_equal(load_masks(a["npz"])[1], load_masks(b["npz"])[1])
