import os
import json
import shutil

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
from PyQt5 import QtWidgets, QtCore  # noqa: E402

from patternstim.binfile import BinFile, read_header  # noqa: E402
from patternstim.calibration import apply_affine  # noqa: E402
from patternstim.config import Config  # noqa: E402
from patternstim.geometry import Circle, Polygon  # noqa: E402
from patternstim.gui.app import MainWindow  # noqa: E402
from patternstim.patterns import load_masks  # noqa: E402
from tests.conftest import EXISTING_BIN, synthetic_camera_image  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    # keep autosave / config out of the user's home
    monkeypatch.setattr("patternstim.gui.select_tab.AUTOSAVE_PATH", tmp_path / "autosave.json")
    cfg = Config()
    w = MainWindow(cfg, config_path=tmp_path / "cfg.json")
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    w.show()
    yield w
    w.close()




def test_full_workflow(window, tmp_path, template_matrix):
    img, truth = synthetic_camera_image(template_matrix, shape=(1000, 1200))
    import tifffile
    tif = tmp_path / "target.tif"
    tifffile.imwrite(str(tif), img.astype(np.uint16))

    # --- calibration tab: load image, place the template roughly, refine, save
    sel = window.select_tab
    assert window.tabs.currentIndex() == 0 and "No calibration" in sel.lbl_cal.text()
    cal_tab = window.calibrate_tab
    cal_tab.load_image(str(tif))
    assert cal_tab.roi.isVisible() and cal_tab.spin_cx.value() == pytest.approx(600)   # default: image centre
    cal_tab.spin_cx.setValue(truth["center_camera"][0] + 8)
    cal_tab.spin_cy.setValue(truth["center_camera"][1] - 6)
    cal_tab.spin_scale.setValue(truth["scale"] * 1.03)
    cal_tab.spin_angle.setValue(truth["angle_deg"] + 1.5)
    cal_tab.chk_mirror.setChecked(True)
    roi_p = cal_tab.roi.params()                    # ROI follows the boxes
    assert roi_p["center_camera"][0] == pytest.approx(truth["center_camera"][0] + 8, abs=0.1)
    assert roi_p["mirror"] is True
    cal_tab.refine()
    p = cal_tab.params()
    assert abs(p["angle_deg"] - truth["angle_deg"]) < 0.15
    assert np.hypot(*(np.subtract(p["center_camera"], truth["center_camera"]))) < 1.0
    # moving the ROI updates the boxes
    cal_tab.roi.setPos([cal_tab.roi.pos()[0] + 3, cal_tab.roi.pos()[1]], finish=True)
    assert cal_tab.spin_cx.value() == pytest.approx(p["center_camera"][0] + 3, abs=0.1)
    cal_tab.roi.setPos([cal_tab.roi.pos()[0] - 3, cal_tab.roi.pos()[1]], finish=True)

    cal_path = tmp_path / "cal.json"
    cal_tab.save_calibration(cal_tab.current_fit(), cal_path)
    assert window.config.last_calibration == str(cal_path)
    assert sel.calibration is not None and sel.calibration.path == str(cal_path)
    assert "Last calibration" in sel.lbl_cal.text() and "today" in sel.lbl_cal.text()
    corners = [(0, 0), (672, 0), (672, 672), (0, 672)]
    err = np.linalg.norm(sel.calibration.dmd_to_camera(corners) -
                         apply_affine(np.linalg.inv(template_matrix), corners), axis=1)
    assert err.max() < 1.5
    # link in the label opens the calibration tab
    sel.lbl_cal.linkActivated.emit("#calibrate")
    assert window.tabs.currentWidget() is cal_tab
    window.tabs.setCurrentIndex(0)
    # reload from file
    cal_tab.load_calibration(str(cal_path))
    assert cal_tab.calibration.fit["mirror"] is True

    # --- select: load a microscope image, drop circles, draw a freehand outline
    cells = tmp_path / "cells.tif"
    tifffile.imwrite(str(cells), np.full((480, 640), 100, np.uint16))
    sel.load_image(str(cells))
    sel.set_mode("circle")
    sel.spin_radius.setValue(10.0)                 # um
    sel.canvas.vb.clicked.emit(300.0, 240.0)
    sel.canvas.vb.clicked.emit(350.0, 200.0)
    assert len(sel.pattern.shapes) == 2
    c0 = sel.pattern.shapes[0]
    assert isinstance(c0, Circle)
    assert c0.r == pytest.approx(sel.calibration.um_to_camera_px(10.0))
    sel.set_mode("freehand")
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    stroke = [(250 + 15 * np.cos(a), 300 + 15 * np.sin(a)) for a in t]
    sel.canvas.vb.strokeFinished.emit(stroke)
    assert len(sel.pattern.shapes) == 3 and isinstance(sel.pattern.shapes[2], Polygon)
    assert [s.name for s in sel.pattern.shapes] == ["S1", "S2", "S3"]
    assert sel.list.count() == 3

    # undo removes the freehand, redo it by re-emitting
    sel.undo()
    assert len(sel.pattern.shapes) == 2
    sel.canvas.vb.strokeFinished.emit(stroke)
    assert len(sel.pattern.shapes) == 3

    # rename + disable through the list
    li = sel.list.item(1)
    li.setText("cellA")
    li.setCheckState(QtCore.Qt.Unchecked)
    assert sel.pattern.shapes[1].name == "cellA" and not sel.pattern.shapes[1].enabled

    # move a circle ROI and check the shape follows
    item0 = sel.items[0]
    item0.roi.setPos([item0.roi.pos()[0] + 5, item0.roi.pos()[1]], finish=True)
    assert sel.pattern.shapes[0].cx == pytest.approx(305.0)

    # preview reflects the combined mask, autosave written
    mask = sel.pattern.combined_mask()
    assert mask.sum() > 0
    assert (tmp_path / "autosave.json").exists()

    # --- export (dialog replaced by a fixed directory)
    sel.edit_name.setText("test set")
    sel.chk_combined.setChecked(True)
    sel.chk_per_shape.setChecked(True)
    out_dir = tmp_path / "export"
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(out_dir))
    sel.export_dialog()
    npz = out_dir / "test_set.patterns.npz"
    names, masks, meta = load_masks(npz)
    assert names == ["test_set_all", "test_set_S1", "test_set_S3"]   # cellA disabled
    np.testing.assert_array_equal(masks[0], masks[1] | masks[2])

    # --- reload the pattern set into a fresh select tab: same masks
    sel.clear_all()
    assert len(sel.pattern.shapes) == 0
    sel.load_pattern(str(out_dir / "test_set.patterns.json"))
    assert [s.name for s in sel.pattern.shapes] == ["S1", "cellA", "S3"]
    np.testing.assert_array_equal(sel.pattern.combined_mask(), masks[0].astype(bool))

    # --- merge tab
    mt = window.merge_tab
    src = tmp_path / "orig.bin"
    shutil.copy(EXISTING_BIN, src)
    mt.edit_bin.setText(str(src))
    mt.add_patterns([str(npz)])
    assert mt.table.rowCount() == 3
    assert mt.table.item(0, 1).text() == "15" and mt.table.item(2, 2).text() == "20"
    assert mt.edit_out.text().endswith("orig_test_set.bin")
    mt.merge()
    out = tmp_path / "orig_test_set.bin"
    assert read_header(out)["nb_images"] == 21
    with BinFile(str(out), rig_id=3, mode="r") as bf:
        np.testing.assert_array_equal(bf.read_frame_uint8(15), masks[0] * 255)
        np.testing.assert_array_equal(bf.read_frame_uint8(16), (1 - masks[0]) * 255)
    with open(str(out) + ".indices.json") as f:
        assert json.load(f)["frames"]["test_set_S3"] == {
            "pos_idx": 19, "neg_idx": 20, "n_pixels": int(masks[2].sum()), "pattern_file": str(npz)}


def test_identity_calibration_and_settings(window, tmp_path):
    window.calibrate_tab.use_identity()
    sel = window.select_tab
    assert sel.calibration is not None and sel.calibration.scale_dmd_per_camera == 1.0
    assert "identity" in sel.lbl_cal.text()
    sel.load_image.__func__  # exists
    img = np.zeros((672, 672), np.float32)
    sel.channels, sel.channel_names = [img], ["image"]
    sel.canvas.set_image(img)
    sel._refresh_warp()
    sel.set_mode("circle")
    sel.canvas.vb.clicked.emit(100.0, 100.0)
    m = sel.pattern.combined_mask()
    assert m[100, 100] and abs(m.sum() - np.pi * (8 / 2.5) ** 2) < 10


def test_stale_calibration_warning(window, tmp_path):
    from patternstim.calibration import Calibration
    cal = Calibration.identity(window.config)
    cal.method = "template"
    cal.created = "2026-01-01T09:00:00"
    p = cal.save(tmp_path / "old.json")
    window.calibrate_tab.load_calibration(str(p))
    txt = window.select_tab.lbl_cal.text()
    assert "2026-01-01 09:00" in txt and "consider recalibrating" in txt
    window.config.calibration_max_age_days = 3650
    window.select_tab.set_config(window.config)
    assert "consider recalibrating" not in window.select_tab.lbl_cal.text()
