"""Select tab: draw circles / freehand outlines on the microscope image, preview, export."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..calibration import Calibration
from ..config import Config
from ..geometry import Circle, Polygon, shape_from_dict, shape_in_field
from ..images import load_image, warp_to_dmd, to_uint8
from ..patterns import PatternSet
from .image_view import ImageCanvas, ShapeItem, stroke_to_polygon

AUTOSAVE_PATH = Path.home() / ".patternstim" / "autosave.patterns.json"


class SelectTab(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)
    calibrationTabRequested = QtCore.pyqtSignal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.calibration: Calibration | None = None
        self.pattern = PatternSet(config=config)
        self.items: list[ShapeItem] = []
        self.selected: ShapeItem | None = None
        self.undo_stack: list[list[dict]] = []
        self.channels: list[np.ndarray] = []
        self.channel_names: list[str] = []
        self.warped: np.ndarray | None = None
        self._loading = False

        self._build_ui()
        self.set_mode("circle")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        tb = QtWidgets.QToolBar()
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        root.addWidget(tb)

        self.act_open = tb.addAction("Load image…")
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self.load_image_dialog)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.setMinimumWidth(110)
        self.channel_combo.currentIndexChanged.connect(self._on_channel)
        tb.addWidget(self.channel_combo)
        tb.addSeparator()

        self.tool_group = QtWidgets.QActionGroup(self)
        self.act_select = tb.addAction("Select (S)")
        self.act_circle = tb.addAction("Circle (C)")
        self.act_free = tb.addAction("Freehand (F)")
        for a, key, mode in ((self.act_select, "S", "select"), (self.act_circle, "C", "circle"),
                             (self.act_free, "F", "freehand")):
            a.setCheckable(True)
            a.setShortcut(key)
            self.tool_group.addAction(a)
            a.triggered.connect(lambda _=False, m=mode: self.set_mode(m))
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" radius (µm) "))
        self.spin_radius = QtWidgets.QDoubleSpinBox()
        self.spin_radius.setRange(0.5, 1000)
        self.spin_radius.setValue(8.0)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setSingleStep(1.0)
        self.spin_radius.valueChanged.connect(self._on_radius_changed)
        tb.addWidget(self.spin_radius)
        tb.addWidget(QtWidgets.QLabel(" margin (µm) "))
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setRange(0, 200)
        self.spin_margin.setValue(0.0)
        self.spin_margin.setDecimals(1)
        self.spin_margin.valueChanged.connect(self._on_margin_changed)
        tb.addWidget(self.spin_margin)
        tb.addSeparator()

        self.act_undo = tb.addAction("Undo")
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_undo.triggered.connect(self.undo)
        self.act_delete = tb.addAction("Delete")
        self.act_delete.setShortcuts([QtGui.QKeySequence.Delete, QtGui.QKeySequence("Backspace")])
        self.act_delete.triggered.connect(self.delete_selected)
        self.act_clear = tb.addAction("Clear all")
        self.act_clear.triggered.connect(self.clear_all)

        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, stretch=1)
        self.canvas = ImageCanvas()
        self.canvas.vb.clicked.connect(self._on_click)
        self.canvas.vb.strokeFinished.connect(self._on_stroke_finished)
        self.canvas.mouseMoved.connect(self._on_mouse)
        body.addWidget(self.canvas, stretch=1)

        right = QtWidgets.QVBoxLayout()
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(380)
        body.addWidget(panel)

        self.lbl_cal = QtWidgets.QLabel()
        self.lbl_cal.setWordWrap(True)
        self.lbl_cal.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbl_cal.linkActivated.connect(lambda _: self.calibrationTabRequested.emit())
        right.addWidget(self.lbl_cal)
        self._refresh_cal_label()

        lbl = QtWidgets.QLabel("<b>Shapes</b> (double-click to rename, untick to disable)")
        lbl.setWordWrap(True)
        right.addWidget(lbl)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.itemSelectionChanged.connect(self._on_list_selection)
        self.list.itemChanged.connect(self._on_list_item_changed)
        right.addWidget(self.list, stretch=2)

        right.addWidget(QtWidgets.QLabel("<b>DMD frame preview</b>"))
        self.preview = pg.GraphicsLayoutWidget()
        self.preview.setFixedHeight(240)
        self.preview_vb = self.preview.addViewBox(lockAspect=True, invertY=True)
        self.preview_item = pg.ImageItem()
        self.preview_vb.addItem(self.preview_item)
        right.addWidget(self.preview)
        prow = QtWidgets.QHBoxLayout()
        self.chk_neg = QtWidgets.QCheckBox("show negative")
        self.chk_neg.toggled.connect(self.update_preview)
        self.chk_bg = QtWidgets.QCheckBox("image background")
        self.chk_bg.setChecked(True)
        self.chk_bg.toggled.connect(self.update_preview)
        prow.addWidget(self.chk_neg)
        prow.addWidget(self.chk_bg)
        right.addLayout(prow)
        self.lbl_preview = QtWidgets.QLabel("")
        self.lbl_preview.setWordWrap(True)
        right.addWidget(self.lbl_preview)

        grp = QtWidgets.QGroupBox("Export pattern set")
        g = QtWidgets.QGridLayout(grp)
        g.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        self.edit_name = QtWidgets.QLineEdit("pattern")
        self.edit_name.textChanged.connect(self._on_name_changed)
        g.addWidget(self.edit_name, 0, 1)
        self.chk_combined = QtWidgets.QCheckBox("combined (all shapes in one frame)")
        self.chk_combined.setChecked(True)
        self.chk_per_shape = QtWidgets.QCheckBox("per shape (one frame per shape)")
        g.addWidget(self.chk_combined, 1, 0, 1, 2)
        g.addWidget(self.chk_per_shape, 2, 0, 1, 2)
        self.btn_export = QtWidgets.QPushButton("Export…  (Ctrl+S)")
        self.btn_export.setShortcut("Ctrl+S")
        self.btn_export.clicked.connect(self.export_dialog)
        g.addWidget(self.btn_export, 3, 0, 1, 2)
        self.btn_load_set = QtWidgets.QPushButton("Load pattern set…")
        self.btn_load_set.clicked.connect(self.load_pattern_dialog)
        g.addWidget(self.btn_load_set, 4, 0, 1, 2)
        right.addWidget(grp)
        self.lbl_mouse = QtWidgets.QLabel("")
        self.lbl_mouse.setWordWrap(True)
        right.addWidget(self.lbl_mouse)

    # ------------------------------------------------------------- calibration
    def set_calibration(self, cal: Calibration | None):
        self.calibration = cal
        self.pattern.calibration = cal
        self._refresh_cal_label()
        self.canvas.set_field_outline(None if cal is None else cal.dmd_field_in_camera())
        self._refresh_warp()
        self.update_preview()

    def _refresh_cal_label(self):
        cal = self.calibration
        link = '<a href="#calibrate">Calibration tab</a>'
        if cal is None:
            colour, text = "#c62828", f"<b>No calibration.</b> Calibrate first in the {link}."
        elif cal.method == "identity":
            colour, text = "#555555", f"Calibration: <b>identity</b> (image already in DMD space). {link}"
        else:
            age = cal.age_days
            stale = age is not None and age > self.config.calibration_max_age_days
            colour = "#e65100" if stale else "#2e7d32"
            src = Path(cal.path).name if cal.path else "unsaved"
            text = (f"Last calibration: <b>{cal.created_text()}</b><br>{src}, "
                    f"{cal.camera_px_to_um(1):.3f} µm / camera px. {link}")
            if stale:
                text += f"<br><b>Older than {self.config.calibration_max_age_days} days: consider recalibrating.</b>"
        self.lbl_cal.setStyleSheet(f"QLabel {{ border: 2px solid {colour}; padding: 4px; }}")
        self.lbl_cal.setText(text)

    def set_config(self, config: Config):
        self.config = config
        self.pattern.config = config
        self._refresh_cal_label()
        self._refresh_warp()
        self.update_preview()

    # ------------------------------------------------------------------ image
    def load_image_dialog(self):
        start = self.config.last_dir or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Microscope image", start, "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            self.channels, self.channel_names = load_image(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Cannot load image", f"{path}\n{e}")
            return
        self.pattern.image_path = str(path)
        self.config.last_dir = str(Path(path).parent)
        self.config.save()
        self._loading = True
        self.channel_combo.clear()
        self.channel_combo.addItems(self.channel_names)
        self._loading = False
        self.channel_combo.setCurrentIndex(0)
        self._show_channel(0)
        self.status.emit(f"Loaded {Path(path).name} ({self.channels[0].shape[1]}x{self.channels[0].shape[0]})")

    def _on_channel(self, idx):
        if not self._loading and 0 <= idx < len(self.channels):
            self._show_channel(idx)

    def _show_channel(self, idx):
        self.canvas.set_image(self.channels[idx])
        self._refresh_warp()
        self.update_preview()

    def _refresh_warp(self):
        img = self.canvas.image
        if img is None or self.calibration is None:
            self.warped = None
            return
        self.warped = to_uint8(warp_to_dmd(img, self.calibration.matrix, self.config.dmd_shape))

    # ------------------------------------------------------------------ tools
    def set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        {"select": self.act_select, "circle": self.act_circle, "freehand": self.act_free}[mode].setChecked(True)
        for it in self.items:
            it.set_movable(mode != "freehand")
        self.status.emit({"select": "Select: click a shape, drag to move, handles to resize. Del removes it.",
                          "circle": "Circle: click to drop a spot of the current radius.",
                          "freehand": "Freehand: drag around a cell; release to close the outline."}[mode])

    def _require_calibration(self) -> bool:
        if self.calibration is None:
            QtWidgets.QMessageBox.warning(self, "No calibration",
                                          "Load or make a calibration in the Calibration tab first.")
            return False
        if self.canvas.image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load a microscope image first.")
            return False
        return True

    def _on_click(self, x, y):
        if self.canvas.mode != "circle" or not self._require_calibration():
            return
        r = self.calibration.um_to_camera_px(self.spin_radius.value())
        self._push_undo()
        self._add_shape(Circle(x, y, r))

    def _on_stroke_finished(self, stroke):
        self.canvas.clear_stroke()
        if not self._require_calibration():
            return
        pts = stroke_to_polygon(stroke, tolerance=max(0.5, self.calibration.um_to_camera_px(0.5)))
        if pts is None:
            self.status.emit("Stroke too short; drag around the cell.")
            return
        self._push_undo()
        self._add_shape(Polygon(pts))

    def _add_shape(self, shape, select=True):
        self.pattern.add(shape)
        item = ShapeItem(shape, self.canvas.vb)
        item.set_movable(self.canvas.mode != "freehand")
        item.changed.connect(self._on_item_changed)
        item.clicked.connect(self._select_item)
        self.items.append(item)
        self._rebuild_list()
        if select:
            self._select_item(item)
        self._changed()

    def _on_item_changed(self, item):
        self._changed()

    def _select_item(self, item: ShapeItem | None):
        if self.selected is not None:
            self.selected.set_selected(False)
        self.selected = item
        if item is not None:
            item.set_selected(True)
            row = self.items.index(item)
            self.list.blockSignals(True)
            self.list.setCurrentRow(row)
            self.list.blockSignals(False)
            if isinstance(item.shape, Circle):
                self.spin_radius.blockSignals(True)
                self.spin_radius.setValue(self.calibration.camera_px_to_um(item.shape.r))
                self.spin_radius.blockSignals(False)

    def _on_radius_changed(self, um):
        if self.selected is not None and isinstance(self.selected.shape, Circle) and self.calibration:
            self._push_undo()
            self.selected.set_radius(self.calibration.um_to_camera_px(um))

    def _on_margin_changed(self, um):
        self.pattern.margin_um = um
        self._changed()

    def _on_name_changed(self, text):
        self.pattern.name = text

    def delete_selected(self):
        item = self.selected if self.selected is not None else (self.items[-1] if self.items else None)
        if item is None:
            return
        self._push_undo()
        item.remove()
        self.items.remove(item)
        self.pattern.shapes.remove(item.shape)
        self.selected = None
        self._rebuild_list()
        self._changed()

    def clear_all(self):
        if not self.items:
            return
        if QtWidgets.QMessageBox.question(self, "Clear all shapes", "Remove all shapes?") != QtWidgets.QMessageBox.Yes:
            return
        self._push_undo()
        self._set_shapes([])

    # ------------------------------------------------------------------- undo
    def _snapshot(self) -> list[dict]:
        return [copy.deepcopy(s.to_dict()) for s in self.pattern.shapes]

    def _push_undo(self):
        self.undo_stack.append(self._snapshot())
        del self.undo_stack[:-50]

    def undo(self):
        if not self.undo_stack:
            return
        self._set_shapes(self.undo_stack.pop())

    def _set_shapes(self, shape_dicts: list[dict]):
        for it in self.items:
            it.remove()
        self.items = []
        self.selected = None
        self.pattern.shapes = []
        for d in shape_dicts:
            self._add_shape(shape_from_dict(d), select=False)
        self._rebuild_list()
        self._changed()

    # ------------------------------------------------------------------- list
    def _rebuild_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for it in self.items:
            li = QtWidgets.QListWidgetItem(it.shape.name)
            li.setFlags(li.flags() | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
            li.setCheckState(QtCore.Qt.Checked if it.shape.enabled else QtCore.Qt.Unchecked)
            kind = "circle" if isinstance(it.shape, Circle) else "freehand"
            li.setToolTip(kind)
            self.list.addItem(li)
        if self.selected in self.items:
            self.list.setCurrentRow(self.items.index(self.selected))
        self.list.blockSignals(False)

    def _on_list_selection(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.items):
            self._select_item(self.items[row])

    def _on_list_item_changed(self, li: QtWidgets.QListWidgetItem):
        row = self.list.row(li)
        if not (0 <= row < len(self.items)):
            return
        it = self.items[row]
        name = li.text().strip() or it.shape.name
        if name != it.shape.name:
            it.set_name(name)
        enabled = li.checkState() == QtCore.Qt.Checked
        if enabled != it.shape.enabled:
            it.set_enabled(enabled)
        self._changed()

    # ---------------------------------------------------------------- preview
    def _changed(self):
        self.update_preview()
        self.autosave()

    def update_preview(self):
        if self.calibration is None:
            self.preview_item.clear()
            self.lbl_preview.setText("")
            return
        mask = self.pattern.combined_mask()
        show = ~mask if self.chk_neg.isChecked() else mask
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if self.chk_bg.isChecked() and self.warped is not None:
            rgb[...] = (self.warped // 2)[..., None]
        rgb[show, 0] = 255
        rgb[show, 1] = np.maximum(rgb[show, 1], 60)
        self.preview_item.setImage(rgb, autoLevels=False)
        self.preview_vb.autoRange()
        n_on = int(mask.sum())
        outside = [s.name for s in self.pattern.shapes
                   if s.enabled and not shape_in_field(s, self.calibration.matrix, self.config.dmd_shape)]
        txt = f"{len([s for s in self.pattern.shapes if s.enabled])} shapes, {n_on} px on " \
              f"({100 * n_on / mask.size:.2f} % of the frame)"
        if outside:
            txt += f"<br><span style='color:#d33'>outside DMD field: {', '.join(outside)}</span>"
        self.lbl_preview.setText(txt)

    def _on_mouse(self, x, y):
        txt = f"camera x={x:.0f} y={y:.0f}"
        if self.calibration is not None:
            dx, dy = self.calibration.camera_to_dmd([(x, y)])[0]
            txt += f"  |  DMD x={dx:.0f} y={dy:.0f} px = {dx * self.config.pixel_size_um:.0f}, " \
                   f"{dy * self.config.pixel_size_um:.0f} µm"
        self.lbl_mouse.setText(txt)

    # ------------------------------------------------------------ persistence
    def autosave(self):
        try:
            self.pattern.save_json(AUTOSAVE_PATH)
        except Exception:  # noqa: BLE001
            pass

    def export_dialog(self):
        if not self._require_calibration():
            return
        if not any(s.enabled for s in self.pattern.shapes):
            QtWidgets.QMessageBox.warning(self, "Nothing to export", "Draw at least one shape.")
            return
        if not (self.chk_combined.isChecked() or self.chk_per_shape.isChecked()):
            QtWidgets.QMessageBox.warning(self, "Nothing to export", "Tick 'combined' and/or 'per shape'.")
            return
        start = self.config.last_dir or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export folder", start)
        if not directory:
            return
        self.pattern.name = self.edit_name.text().strip() or "pattern"
        self.pattern.export_combined = self.chk_combined.isChecked()
        self.pattern.export_per_shape = self.chk_per_shape.isChecked()
        try:
            res = self.pattern.export(directory)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        self.config.last_dir = directory
        self.config.save()
        self.status.emit(f"Exported {len(res['frames'])} frames to {res['npz']}")
        QtWidgets.QMessageBox.information(
            self, "Pattern set exported",
            f"Masks: {res['npz']}\nDescription: {res['json']}\nPreviews: {res['previews']}\n\n"
            f"Frames: {', '.join(res['frames'])}\n\nUse the Merge tab (or the CLI) to append them to a bin.")

    def load_pattern_dialog(self):
        start = self.config.last_dir or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pattern set", start, "Pattern sets (*.patterns.json);;JSON (*.json)")
        if path:
            self.load_pattern(path)

    def load_pattern(self, path):
        try:
            ps = PatternSet.load(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Cannot load pattern set", f"{path}\n{e}")
            return
        if ps.image_path and Path(ps.image_path).exists() and ps.image_path != self.pattern.image_path:
            self.load_image(ps.image_path)
        if ps.calibration is not None and self.calibration is None:
            self.set_calibration(ps.calibration)
        self._push_undo()
        self.pattern.name = ps.name
        self.edit_name.setText(ps.name)
        self.spin_margin.setValue(ps.margin_um)
        self.chk_combined.setChecked(ps.export_combined)
        self.chk_per_shape.setChecked(ps.export_per_shape)
        self._set_shapes([s.to_dict() for s in ps.shapes])
        self.status.emit(f"Loaded pattern set {Path(path).name} ({len(ps.shapes)} shapes)")
