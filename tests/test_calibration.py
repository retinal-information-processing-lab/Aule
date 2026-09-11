import json
import numpy as np
import pytest

from patternstim.binfile import BinFile, read_header
from patternstim.calibration import (Calibration, fit_affine, apply_affine, make_calibration_bin,
                                     target_mask, target_geometry, similarity_matrix,
                                     dmd_to_camera_matrix, decompose_similarity, refine_fit,
                                     template_signal, reference_image)
from tests.conftest import synthetic_camera_image


def test_target_mask_elements(config):
    m = target_mask(config)
    g = target_geometry(config)
    assert m.shape == (672, 672) and m.dtype == bool
    assert m[int(g.cy), int(g.cx)]                                   # cross centre
    assert m[int(g.cy), int(g.cx + g.ring_r)]                        # ring on +x
    assert m[int(g.cy), int((g.bar[0] + g.bar[2]) / 2)]              # bar
    assert m[int(g.dot_c[1]), int(g.dot_c[0])]                       # dot upper right
    assert not m[int(g.dot_c[0]), int(g.dot_c[1])]                   # nothing at the mirrored place
    assert not m[int(g.cy), int(g.cx - (g.bar[0] + g.bar[2]) / 2 + g.cx)]  # no bar towards -x
    assert 0.01 < m.mean() < 0.05


def test_similarity_roundtrip(config, synthetic_matrix):
    p = decompose_similarity(synthetic_matrix, config.dmd_shape)
    assert p["mirror"] is True and p["scale"] == pytest.approx(2.5) and p["angle_deg"] == pytest.approx(10)
    M = similarity_matrix(p["center_camera"], p["scale"], p["angle_deg"], p["mirror"], config.dmd_shape)
    np.testing.assert_allclose(M, synthetic_matrix, atol=1e-9)
    # DMD centre maps to center_camera, +x of the DMD goes along the (mirrored, rotated) x axis
    A = dmd_to_camera_matrix(p["center_camera"], p["scale"], p["angle_deg"], p["mirror"], config.dmd_shape)
    np.testing.assert_allclose(apply_affine(A, [(336, 336)])[0], p["center_camera"])
    dx = apply_affine(A, [(346, 336)])[0] - p["center_camera"]
    assert np.hypot(*dx) == pytest.approx(25)
    p2 = decompose_similarity(similarity_matrix((100, 50), 0.8, -30, False, (64, 80)), (64, 80))
    assert p2["mirror"] is False and p2["angle_deg"] == pytest.approx(-30) and p2["scale"] == pytest.approx(0.8)
    np.testing.assert_allclose(p2["center_camera"], [100, 50])


def test_calibration_from_template_fit_and_json(config, synthetic_matrix, tmp_path):
    p = decompose_similarity(synthetic_matrix, config.dmd_shape)
    p["signal"] = 0.9
    cal = Calibration.from_template_fit(p, config, image_shape=(600, 800), image_path="img.tif")
    np.testing.assert_allclose(cal.matrix, synthetic_matrix, atol=1e-9)
    assert cal.method == "template" and cal.fit["mirror"] is True and cal.fit["signal"] == 0.9
    assert cal.created_datetime is not None and cal.age_days < 0.01
    assert "today" in cal.created_text()
    path = cal.save(tmp_path / "cal.json")
    cal2 = Calibration.load(path)
    np.testing.assert_allclose(cal2.matrix, cal.matrix)
    assert cal2.method == "template" and cal2.fit == cal.fit and cal2.path == str(path)
    assert cal2.dmd_shape == (672, 672) and cal2.image_shape == (600, 800)
    assert cal2.scale_dmd_per_camera == pytest.approx(0.4)
    assert cal2.um_to_camera_px(10) == pytest.approx(10 / 2.5 / 0.4)
    assert cal2.dmd_field_in_camera().shape == (4, 2)
    assert cal2.similarity_params()["angle_deg"] == pytest.approx(10)


def test_old_calibration_files_still_load(config, tmp_path):
    d = {"matrix": np.eye(3).tolist(), "camera_points": [[0, 0]], "dmd_points": [[0, 0]],
         "created": "2026-09-01T10:00:00", "pixel_size_um": 2.5, "dmd_shape": [672, 672]}
    (tmp_path / "old.json").write_text(json.dumps(d))
    cal = Calibration.load(tmp_path / "old.json")
    assert cal.method == "points" and cal.age_days > 1 and "days ago" in cal.created_text()
    ident = Calibration.identity(config)
    assert ident.method == "identity" and ident.created_datetime is None


def test_fit_affine_from_points(config, synthetic_matrix):
    dmd = np.array([(100, 100), (500, 120), (150, 600), (336, 336)], float)
    cam = apply_affine(np.linalg.inv(synthetic_matrix), dmd)
    M, res = fit_affine(cam, dmd)
    np.testing.assert_allclose(M, synthetic_matrix, atol=1e-9)
    assert res.max() < 1e-9
    with pytest.raises(ValueError):
        fit_affine([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])


def test_refine_recovers_perturbed_placement(config, template_matrix):
    img, _ = synthetic_camera_image(template_matrix, shape=(1000, 1200), noise=0.05, seed=3)
    truth = decompose_similarity(template_matrix, config.dmd_shape)
    assert template_signal(img, truth, config) > 0.85
    start = dict(truth)
    start["center_camera"] = [truth["center_camera"][0] + 12, truth["center_camera"][1] - 9]
    start["scale"] = truth["scale"] * 1.04
    start["angle_deg"] = truth["angle_deg"] + 2.0
    s0 = template_signal(img, start, config)
    p = refine_fit(img, start, config)
    assert p["signal"] > s0 and p["signal"] > 0.8
    assert p["mirror"] is True
    assert abs(p["angle_deg"] - truth["angle_deg"]) < 0.15
    assert abs(p["scale"] / truth["scale"] - 1) < 0.005
    assert np.hypot(*(np.subtract(p["center_camera"], truth["center_camera"]))) < 1.0
    # corners of the DMD field land within a pixel of the truth
    M = similarity_matrix(p["center_camera"], p["scale"], p["angle_deg"], p["mirror"], config.dmd_shape)
    corners = [(0, 0), (672, 0), (672, 672), (0, 672)]
    err = np.linalg.norm(apply_affine(np.linalg.inv(M), corners) -
                         apply_affine(np.linalg.inv(template_matrix), corners), axis=1)
    assert err.max() < 1.5


def test_make_calibration_bin(config, tmp_path):
    res = make_calibration_bin(tmp_path / "calib.bin", config)
    assert read_header(res["bin"]) == {"xsize": 672, "ysize": 672, "nb_images": 2, "nb_bits": 8}
    assert res["frames"] == ["target", "black"] and res["target_frame"] == 0
    assert (tmp_path / "calib.png").exists()
    with BinFile(res["bin"], rig_id=3, mode="r") as bf:
        np.testing.assert_array_equal(bf.read_frame_uint8(0), target_mask(config) * 255)
        assert bf.read_frame_uint8(1).max() == 0
    assert reference_image(config).shape == (672, 672, 3)


def test_config_roundtrip_keeps_path(tmp_path):
    from patternstim.config import Config
    p = tmp_path / "cfg.json"
    cfg = Config.load(p)              # missing file -> defaults, remembers path
    assert cfg.path == str(p) and cfg.rig_id == 3 and cfg.calibration_max_age_days == 30
    cfg.pixel_size_um = 3.0
    cfg.save()                        # no argument -> same path
    cfg2 = Config.load(p)
    assert cfg2.pixel_size_um == 3.0 and "path" not in json.load(open(p))
