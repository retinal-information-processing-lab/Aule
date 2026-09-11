"""Calibration tab: align a template of the projected target on a camera image.

The template (magenta) is a pyqtgraph ROI covering the whole DMD field: drag it to centre,
corner handles scale + rotate, the right-edge handle rotates only. Its placement (centre,
scale, angle, mirror) is the calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from ..calibration import (Calibration, make_calibration_bin, reference_image,
                           refine_fit, target_geometry, template_signal, TARGET_FRAME_INDEX)
from ..config import Config
from ..images import load_image
from .image_view import ImageCanvas

TEMPLATE_COLOR = (255, 0, 255)
PEN_TEMPLATE_FIELD = pg.mkPen(TEMPLATE_COLOR + (160,), width=1, style=QtCore.Qt.PenStyle.DashLine)

INSTRUCTIONS = (
    "<b>When to calibrate</b><br>"
    "After any change of the optics (camera, objective, DMD, tube lens) and at regular intervals. "
    "Rotation and scale should barely move; centring is the main drift to check.<br><br>"
    f"1. Display frame <b>{TARGET_FRAME_INDEX}</b> of the calibration bin "
    "(<i>calibration_target_MEA&lt;rig&gt;.bin</i>, in the common stimulus folder; "
    "<i>Export calibration bin</i> regenerates it) with the DMD software.<br>"
    "2. Image the projected target with the camera <b>in the same configuration</b> "
    "(objective, binning, orientation) as the images used to select cells. <i>Load camera image</i>.<br>"
    "3. Align the magenta template on the projected light: drag the template to centre the cross, "
    "use a <b>corner handle</b> to scale (ring) and rotate (bar must point along the projected bar), "
    "the <b>right-edge handle</b> rotates only. Tick <i>mirror</i> if the dot ends up on the wrong side "
    "of the bar. Fine-tune with the boxes below, or click <i>Refine automatically</i>.<br>"
    "4. <i>Save calibration</i>. Its date is shown in the <i>Select patterns</i> tab."
)


def _qtransform(m: np.ndarray) -> QtGui.QTransform:
    """QTransform for a 3x3 matrix acting on column vectors [x, y, 1]."""
    return QtGui.QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])


class TemplateROI(pg.ROI):
    """ROI spanning the DMD field with the target drawn inside it."""

    def __init__(self, dmd_shape):
        self.dmd_shape = tuple(dmd_shape)
        self.mirror = False
        pen = pg.mkPen(TEMPLATE_COLOR + (200,), width=1, style=QtCore.Qt.PenStyle.DashLine)
        super().__init__([0, 0], [10, 10], pen=pen, hoverPen=pg.mkPen(TEMPLATE_COLOR, width=2),
                         aspectLocked=True, rotatable=True, resizable=True, removable=False)
        for corner in ([0, 0], [1, 0], [0, 1], [1, 1]):
            self.addScaleRotateHandle(corner, [0.5, 0.5])
        self.addRotateHandle([1, 0.5], [0.5, 0.5])
        self.setZValue(25)

        self.stroke_item = QtWidgets.QGraphicsPathItem(self)
        self.fill_item = QtWidgets.QGraphicsPathItem(self)
        self.fill_item.setBrush(pg.mkBrush(TEMPLATE_COLOR + (110,)))
        self.fill_item.setPen(pg.mkPen(None))
        self.build_paths()
        self.sigRegionChanged.connect(self._update_children)

    # -- template drawing (DMD coordinates, mapped by the child transform) -------------
    def set_dmd_shape(self, dmd_shape):
        self.dmd_shape = tuple(dmd_shape)
        self.build_paths()
        self._update_children()

    def build_paths(self):
        g = target_geometry(Config(dmd_height=self.dmd_shape[0], dmd_width=self.dmd_shape[1]))
        stroke = QtGui.QPainterPath()
        stroke.moveTo(g.cx - g.arm, g.cy); stroke.lineTo(g.cx + g.arm, g.cy)
        stroke.moveTo(g.cx, g.cy - g.arm); stroke.lineTo(g.cx, g.cy + g.arm)
        stroke.addEllipse(QtCore.QPointF(g.cx, g.cy), g.ring_r, g.ring_r)
        fill = QtGui.QPainterPath()
        x0, y0, x1, y1 = g.bar
        fill.addRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        fill.addEllipse(QtCore.QPointF(*g.dot_c), g.dot_r, g.dot_r)
        self.stroke_item.setPath(stroke)
        self.fill_item.setPath(fill)
        pen = pg.mkPen(TEMPLATE_COLOR + (170,), width=g.thickness)
        pen.setCosmetic(False)
        self.stroke_item.setPen(pen)

    def _update_children(self):
        h, w = self.dmd_shape
        s = float(self.size()[0]) / w
        m = np.diag([s, s, 1.0])
        if self.mirror:
            m = m @ np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        tr = _qtransform(m)
        self.stroke_item.setTransform(tr)
        self.fill_item.setTransform(tr)

    # -- placement <-> parameters ------------------------------------------------------
    def params(self) -> dict:
        h, w = self.dmd_shape
        s = float(self.size()[0]) / w
        th = np.deg2rad(float(self.angle()))
        pos = np.array([float(self.pos()[0]), float(self.pos()[1])])
        half = np.array([s * w / 2, s * h / 2])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c = pos + R @ half
        return {"center_camera": [float(c[0]), float(c[1])], "scale": s,
                "angle_deg": float(self.angle()), "mirror": self.mirror}

    def set_params(self, p: dict, finish: bool = True):
        h, w = self.dmd_shape
        s = float(p["scale"])
        th = np.deg2rad(float(p["angle_deg"]))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c = np.asarray(p["center_camera"], float)
        pos = c - R @ np.array([s * w / 2, s * h / 2])
        self.mirror = bool(p["mirror"])
        self.setState({"pos": pg.Point(pos[0], pos[1]), "size": pg.Point(s * w, s * h),
                       "angle": float(p["angle_deg"])}, update=False)
        self._update_children()
        self.stateChanged(finish=finish)


class CalibrateTab(QtWidgets.QWidget):
    calibrationChanged = QtCore.pyqtSignal(object)   # Calibration
    status = QtCore.pyqtSignal(str)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.calibration: Calibration | None = None
        self.image_path = ""
        self._syncing = False

        self.canvas = ImageCanvas()
        self.canvas.set_mode("select")
        self.canvas.mouseMoved.connect(self._on_mouse)
        self.roi = TemplateROI(config.dmd_shape)
        self.roi.hide()
        self.canvas.add_item(self.roi)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.roi.sigRegionChangeFinished.connect(self._on_roi_finished)

        # ---------------------------------------------------------------- right panel
        right = QtWidgets.QVBoxLayout()
        self.info_group = QtWidgets.QGroupBox("Procedure (untick to hide)")
        self.info_group.setCheckable(True)
        self.info_group.setChecked(True)
        info = QtWidgets.QLabel(INSTRUCTIONS)
        info.setWordWrap(True)
        info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        info_lay = QtWidgets.QVBoxLayout(self.info_group)
        info_lay.addWidget(info)
        self.info_group.toggled.connect(info.setVisible)
        right.addWidget(self.info_group)

        ref_row = QtWidgets.QHBoxLayout()
        self.ref_view = pg.GraphicsLayoutWidget()
        self.ref_view.setFixedSize(180, 180)
        vb = self.ref_view.addViewBox(lockAspect=True, invertY=True, enableMouse=False)
        self.ref_item = pg.ImageItem()
        vb.addItem(self.ref_item)
        self.ref_vb = vb
        ref_row.addWidget(self.ref_view)
        ref_col = QtWidgets.QVBoxLayout()
        ref_col.addWidget(QtWidgets.QLabel("<b>Projected target</b><br>(logical DMD frame)"))
        self.btn_gen = QtWidgets.QPushButton("Export calibration bin…")
        self.btn_load_img = QtWidgets.QPushButton("Load camera image…")
        ref_col.addWidget(self.btn_gen)
        ref_col.addWidget(self.btn_load_img)
        ref_col.addStretch(1)
        ref_row.addLayout(ref_col)
        right.addLayout(ref_row)

        grp = QtWidgets.QGroupBox("Template placement (camera pixels)")
        form = QtWidgets.QGridLayout(grp)
        self.spin_cx = QtWidgets.QDoubleSpinBox(); self.spin_cx.setRange(-1e6, 1e6); self.spin_cx.setDecimals(1)
        self.spin_cy = QtWidgets.QDoubleSpinBox(); self.spin_cy.setRange(-1e6, 1e6); self.spin_cy.setDecimals(1)
        self.spin_scale = QtWidgets.QDoubleSpinBox(); self.spin_scale.setRange(1e-4, 1e4)
        self.spin_scale.setDecimals(4); self.spin_scale.setSingleStep(0.01)
        self.spin_angle = QtWidgets.QDoubleSpinBox(); self.spin_angle.setRange(-360, 360)
        self.spin_angle.setDecimals(2); self.spin_angle.setSingleStep(0.5); self.spin_angle.setSuffix(" °")
        self.chk_mirror = QtWidgets.QCheckBox("mirror (flip DMD x)")
        form.addWidget(QtWidgets.QLabel("centre x"), 0, 0); form.addWidget(self.spin_cx, 0, 1)
        form.addWidget(QtWidgets.QLabel("centre y"), 0, 2); form.addWidget(self.spin_cy, 0, 3)
        form.addWidget(QtWidgets.QLabel("scale (cam px / DMD px)"), 1, 0, 1, 2); form.addWidget(self.spin_scale, 1, 2, 1, 2)
        form.addWidget(QtWidgets.QLabel("angle"), 2, 0); form.addWidget(self.spin_angle, 2, 1)
        form.addWidget(self.chk_mirror, 2, 2, 1, 2)
        self.lbl_scale_um = QtWidgets.QLabel("")
        form.addWidget(self.lbl_scale_um, 3, 0, 1, 4)
        self.btn_reset = QtWidgets.QPushButton("Reset template")
        self.btn_refine = QtWidgets.QPushButton("Refine automatically")
        self.btn_refine.setToolTip("Local optimisation of centre / scale / angle on the image intensity, "
                                   "starting from the current placement (mirror unchanged).")
        form.addWidget(self.btn_reset, 4, 0, 1, 2)
        form.addWidget(self.btn_refine, 4, 2, 1, 2)
        self.lbl_signal = QtWidgets.QLabel("")
        form.addWidget(self.lbl_signal, 5, 0, 1, 4)
        right.addWidget(grp)

        btns = QtWidgets.QGridLayout()
        self.btn_save = QtWidgets.QPushButton("Save calibration…")
        self.btn_save.setStyleSheet("font-weight: bold")
        self.btn_load_cal = QtWidgets.QPushButton("Load calibration…")
        self.btn_identity = QtWidgets.QPushButton("Use identity (image already in DMD space)")
        btns.addWidget(self.btn_save, 0, 0)
        btns.addWidget(self.btn_load_cal, 0, 1)
        btns.addWidget(self.btn_identity, 1, 0, 1, 2)
        right.addLayout(btns)

        self.lbl_result = QtWidgets.QLabel("No calibration loaded.")
        self.lbl_result.setWordWrap(True)
        self.lbl_mouse = QtWidgets.QLabel("")
        right.addWidget(self.lbl_result)
        right.addWidget(self.lbl_mouse)
        right.addStretch(1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.canvas, stretch=1)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(460)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(480)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        self.btn_gen.clicked.connect(self.generate_bin)
        self.btn_load_img.clicked.connect(self.load_image_dialog)
        self.btn_reset.clicked.connect(self.reset_template)
        self.btn_refine.clicked.connect(self.refine)
        self.btn_save.clicked.connect(self.save_dialog)
        self.btn_load_cal.clicked.connect(self.load_dialog)
        self.btn_identity.clicked.connect(self.use_identity)
        for sp in (self.spin_cx, self.spin_cy, self.spin_scale, self.spin_angle):
            sp.valueChanged.connect(self._on_spin_changed)
        self.chk_mirror.toggled.connect(self._on_spin_changed)
        self.refresh_reference()
        self._set_controls_enabled(False)

    # -- config / reference ------------------------------------------------------------
    def refresh_reference(self):
        self.ref_item.setImage(reference_image(self.config, annotate=False))
        self.ref_vb.autoRange()

    def set_config(self, config: Config):
        self.config = config
        self.roi.set_dmd_shape(config.dmd_shape)
        self.refresh_reference()
        if self.roi.isVisible():
            self._on_roi_changed()

    def _set_controls_enabled(self, on: bool):
        for w in (self.spin_cx, self.spin_cy, self.spin_scale, self.spin_angle, self.chk_mirror,
                  self.btn_reset, self.btn_refine):
            w.setEnabled(on)

    # -- calibration bin ----------------------------------------------------------------
    def generate_bin(self):
        start = str(Path(self.config.last_dir or Path.home()) / f"calibration_target_MEA{self.config.rig_id}.bin")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Calibration bin", start, "Bin files (*.bin)")
        if not path:
            return
        res = make_calibration_bin(path, self.config)
        self.config.last_dir = str(Path(path).parent)
        QtWidgets.QMessageBox.information(
            self, "Calibration bin written",
            f"{res['bin']}\n\nFrames: {', '.join(f'{i}={n}' for i, n in enumerate(res['frames']))}\n"
            f"Reference image: {res['png']}\n\nDisplay frame {res['target_frame']} (target) with the DMD "
            "software and image it with the camera.")

    # -- image ------------------------------------------------------------------------------
    def load_image_dialog(self):
        start = self.config.last_dir or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Camera image of the projected target", start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        chans, names = load_image(path)
        img = chans[0]
        if len(chans) > 1:
            name, ok = QtWidgets.QInputDialog.getItem(self, "Channel", "Use channel:", names, 0, False)
            if ok:
                img = chans[names.index(name)]
        self.image_path = str(path)
        self.config.last_dir = str(Path(path).parent)
        self.set_image(img)
        self.status.emit(f"Loaded {Path(path).name} ({img.shape[1]}x{img.shape[0]})")

    def set_image(self, img: np.ndarray):
        self.canvas.set_image(img)
        self._set_controls_enabled(True)
        self.roi.show()
        # start from the current calibration if it fits this image, else a default placement
        if self.calibration is not None and self.calibration.method != "identity":
            self.roi.set_params(self.calibration.similarity_params())
        else:
            self.reset_template()
        self.canvas.vb.autoRange()

    def reset_template(self):
        img = self.canvas.image
        if img is None:
            return
        h, w = self.config.dmd_shape
        ih, iw = img.shape[:2]
        self.roi.set_params({"center_camera": [iw / 2, ih / 2],
                             "scale": 0.6 * min(iw / w, ih / h), "angle_deg": 0.0, "mirror": False})

    # -- template placement sync -----------------------------------------------------
    def _on_roi_changed(self):
        if self._syncing:
            return
        self._syncing = True
        p = self.roi.params()
        self.spin_cx.setValue(p["center_camera"][0])
        self.spin_cy.setValue(p["center_camera"][1])
        self.spin_scale.setValue(p["scale"])
        self.spin_angle.setValue(p["angle_deg"])
        self.chk_mirror.setChecked(p["mirror"])
        self._syncing = False
        um = self.config.pixel_size_um / p["scale"]
        self.lbl_scale_um.setText(f"= {um:.4f} µm per camera px ({1 / um:.3f} camera px per µm)")

    def _on_roi_finished(self):
        self._update_signal()

    def _on_spin_changed(self, *_):
        if self._syncing:
            return
        self._syncing = True
        self.roi.set_params(self.params(), finish=False)
        self._syncing = False
        self._on_roi_changed()
        self._update_signal()

    def params(self) -> dict:
        return {"center_camera": [self.spin_cx.value(), self.spin_cy.value()],
                "scale": self.spin_scale.value(), "angle_deg": self.spin_angle.value(),
                "mirror": self.chk_mirror.isChecked()}

    def _update_signal(self):
        if self.canvas.image is None or not self.roi.isVisible():
            self.lbl_signal.setText("")
            return
        sig = template_signal(self.canvas.image, self.params(), self.config)
        self.lbl_signal.setText(f"template signal: <b>{sig:.2f}</b>  (1 = on the projected light, 0 = background)")

    def refine(self):
        if self.canvas.image is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            before = template_signal(self.canvas.image, self.params(), self.config)
            p = refine_fit(self.canvas.image, self.params(), self.config)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if p["signal"] < before:
            self.status.emit(f"Refinement did not improve the fit (signal {before:.2f} -> {p['signal']:.2f}); kept.")
            return
        self.roi.set_params(p)
        self.status.emit(f"Refined: template signal {before:.2f} -> {p['signal']:.2f}")

    # -- calibration ----------------------------------------------------------------------
    def current_fit(self) -> Calibration:
        p = self.params()
        p["signal"] = template_signal(self.canvas.image, p, self.config) if self.canvas.image is not None else None
        shape = self.canvas.image.shape if self.canvas.image is not None else ()
        return Calibration.from_template_fit(p, self.config, image_shape=shape, image_path=self.image_path)

    def use_identity(self):
        self._set_calibration(Calibration.identity(self.config), announce=True)

    def _set_calibration(self, cal: Calibration, announce: bool):
        self.calibration = cal
        if cal.method == "identity":
            self.canvas.set_field_outline(cal.dmd_field_in_camera())
        else:
            self.canvas.set_field_outline(None)
            if self.canvas.image is not None:
                self.roi.set_params(cal.similarity_params())
        src = f"file: {cal.path}" if cal.path else ("identity" if cal.method == "identity" else "unsaved")
        p = cal.similarity_params()
        sig = cal.fit.get("signal")
        self.lbl_result.setText(
            f"<b>Current calibration</b> ({src})<br>"
            f"Date: <b>{cal.created_text()}</b><br>"
            f"Scale: {p['scale']:.4f} camera px per DMD px = {cal.camera_px_to_um(1):.4f} µm per camera px<br>"
            f"Angle: {p['angle_deg']:.2f}°, mirror: {'yes' if p['mirror'] else 'no'}, "
            f"DMD centre at camera ({p['center_camera'][0]:.1f}, {p['center_camera'][1]:.1f})"
            + (f"<br>Template signal: {sig:.2f}" if sig is not None else ""))
        if announce:
            self.calibrationChanged.emit(cal)

    def save_dialog(self):
        if not self.roi.isVisible():
            QtWidgets.QMessageBox.warning(self, "No image", "Load a camera image of the target and align the "
                                                            "template first.")
            return
        cal = self.current_fit()
        day = cal.created[:10]
        start_dir = Path(self.config.last_calibration).parent if self.config.last_calibration else \
            Path(self.config.last_dir or Path.home())
        start = str(start_dir / f"calibration_MEA{self.config.rig_id}_{day}.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration", start, "JSON (*.json)")
        if not path:
            return
        self.save_calibration(cal, path)

    def save_calibration(self, cal: Calibration, path) -> Calibration:
        cal.save(path)
        self.config.last_calibration = str(path)
        self.config.last_dir = str(Path(path).parent)
        self.config.save()
        self._set_calibration(cal, announce=True)
        self.status.emit(f"Calibration saved to {path}")
        return cal

    def load_dialog(self):
        start = self.config.last_dir or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load calibration", start, "JSON (*.json)")
        if path:
            self.load_calibration(path)

    def load_calibration(self, path) -> bool:
        try:
            cal = Calibration.load(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Cannot load calibration", f"{path}\n{e}")
            return False
        if tuple(cal.dmd_shape) != tuple(self.config.dmd_shape):
            QtWidgets.QMessageBox.warning(
                self, "DMD size mismatch",
                f"Calibration was made for DMD {cal.dmd_shape}, current config is {self.config.dmd_shape}.")
        self.config.last_calibration = str(path)
        self.config.save()
        self._set_calibration(cal, announce=True)
        self.status.emit(f"Calibration loaded from {path}")
        return True

    def _on_mouse(self, x, y):
        txt = f"camera x={x:.1f} y={y:.1f}"
        if self.roi.isVisible():
            M = Calibration.from_template_fit(self.params(), self.config).matrix
            dx, dy = (M @ np.array([x, y, 1.0]))[:2]
            txt += f"   DMD x={dx:.1f} y={dy:.1f} px"
        elif self.calibration is not None:
            dx, dy = self.calibration.camera_to_dmd([(x, y)])[0]
            txt += f"   DMD x={dx:.1f} y={dy:.1f} px"
        self.lbl_mouse.setText(txt)
