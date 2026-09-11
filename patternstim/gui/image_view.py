"""pyqtgraph image canvas with tool modes (select / circle / freehand / point) and shape ROIs."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..geometry import Circle, Polygon, Shape, simplify_polyline

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

PEN_NORMAL = pg.mkPen((255, 220, 0), width=2)
PEN_SELECTED = pg.mkPen((0, 255, 255), width=3)
PEN_DISABLED = pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DashLine)
PEN_FIELD = pg.mkPen((0, 200, 255), width=1, style=QtCore.Qt.DashLine)
PEN_STROKE = pg.mkPen((255, 120, 0), width=2)
HOVER_PEN = pg.mkPen((255, 255, 255), width=2)


class ToolViewBox(pg.ViewBox):
    """ViewBox whose left-button behaviour depends on ``mode``."""

    clicked = QtCore.pyqtSignal(float, float)         # view coords, modes circle/point
    strokeChanged = QtCore.pyqtSignal(list)           # freehand in progress
    strokeFinished = QtCore.pyqtSignal(list)          # freehand released

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "select"
        self._stroke: list[tuple[float, float]] = []
        self.setAspectLocked(True)
        self.invertY(True)
        self.setMouseMode(pg.ViewBox.PanMode)

    def mouseClickEvent(self, ev):
        if self.mode in ("circle", "point") and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            p = self.mapToView(ev.pos())
            self.clicked.emit(p.x(), p.y())
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self.mode == "freehand" and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            p = self.mapToView(ev.pos())
            if ev.isStart():
                self._stroke = [(p.x(), p.y())]
            else:
                self._stroke.append((p.x(), p.y()))
            self.strokeChanged.emit(list(self._stroke))
            if ev.isFinish():
                self.strokeFinished.emit(list(self._stroke))
                self._stroke = []
            return
        super().mouseDragEvent(ev, axis)


class ShapeItem(QtCore.QObject):
    """Binds a geometry Shape to a pyqtgraph ROI living in the view."""

    changed = QtCore.pyqtSignal(object)      # self, after user edit
    clicked = QtCore.pyqtSignal(object)      # self

    def __init__(self, shape: Shape, view: pg.ViewBox):
        super().__init__()
        self.shape = shape
        self.view = view
        self.selected = False
        if isinstance(shape, Circle):
            d = 2 * shape.r
            self.roi = pg.CircleROI([shape.cx - shape.r, shape.cy - shape.r], [d, d],
                                    pen=PEN_NORMAL, hoverPen=HOVER_PEN, removable=False)
        else:
            self.roi = pg.PolyLineROI(shape.points.tolist(), closed=True,
                                      pen=PEN_NORMAL, hoverPen=HOVER_PEN, removable=False)
        self.label = pg.TextItem(shape.name, color=(255, 220, 0), anchor=(0.5, 1.2))
        self.label.setZValue(20)
        view.addItem(self.roi)
        view.addItem(self.label)
        self.roi.sigRegionChanged.connect(self._on_region_changed)
        self.roi.sigRegionChangeFinished.connect(self._on_region_finished)
        self.roi.sigClicked.connect(lambda *_: self.clicked.emit(self))
        self._sync_from_roi()
        self.update_style()

    # -- state ---------------------------------------------------------------
    def _sync_from_roi(self):
        if isinstance(self.shape, Circle):
            pos = self.roi.pos()
            size = self.roi.size()
            r = float(size[0]) / 2
            self.shape.cx, self.shape.cy, self.shape.r = float(pos[0]) + r, float(pos[1]) + r, r
        else:
            pts = [self.roi.mapToParent(p) for _, p in self.roi.getLocalHandlePositions()]
            self.shape.points = np.array([[q.x(), q.y()] for q in pts], dtype=float)
        self._place_label()

    def _place_label(self):
        poly = self.shape.to_polygon()
        if len(poly):
            cx, cy = poly[:, 0].mean(), poly[:, 1].min() if isinstance(self.shape, Polygon) else self.shape.cy - self.shape.r
            self.label.setPos(cx, cy)

    def _on_region_changed(self):
        self._sync_from_roi()

    def _on_region_finished(self):
        self._sync_from_roi()
        self.changed.emit(self)

    def set_radius(self, r: float):
        if isinstance(self.shape, Circle):
            self.roi.setSize([2 * r, 2 * r], center=[0.5, 0.5], finish=True)

    def set_name(self, name: str):
        self.shape.name = name
        self.label.setText(name)

    def set_enabled(self, enabled: bool):
        self.shape.enabled = enabled
        self.update_style()

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update_style()

    def set_movable(self, movable: bool):
        self.roi.translatable = movable

    def update_style(self):
        pen = PEN_SELECTED if self.selected else (PEN_NORMAL if self.shape.enabled else PEN_DISABLED)
        self.roi.setPen(pen)
        self.label.setColor(pen.color())

    def remove(self):
        self.view.removeItem(self.roi)
        self.view.removeItem(self.label)


class ImageCanvas(QtWidgets.QWidget):
    """Image + histogram contrast + overlays. Emits mouse position in view coords."""

    mouseMoved = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, stretch=1)
        self.vb = ToolViewBox()
        self.glw.addItem(self.vb)
        self.image_item = pg.ImageItem()
        self.image_item.setZValue(-10)
        self.vb.addItem(self.image_item)
        self.hist = pg.HistogramLUTWidget()
        self.hist.setImageItem(self.image_item)
        self.hist.setMaximumWidth(120)
        layout.addWidget(self.hist)

        self.field_item = pg.PlotCurveItem(pen=PEN_FIELD)
        self.field_item.setZValue(5)
        self.vb.addItem(self.field_item)
        self.stroke_item = pg.PlotCurveItem(pen=PEN_STROKE)
        self.stroke_item.setZValue(15)
        self.vb.addItem(self.stroke_item)
        self.vb.strokeChanged.connect(self._on_stroke)

        self.image: np.ndarray | None = None
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # -- image ------------------------------------------------------------------
    def set_image(self, image: np.ndarray | None, auto_levels: bool = True):
        self.image = image
        if image is None:
            self.image_item.clear()
            return
        self.image_item.setImage(image, autoLevels=False)
        if auto_levels:
            lo, hi = np.percentile(image, [1, 99.7])
            if hi <= lo:
                hi = lo + 1
            self.hist.setLevels(lo, hi)
        self.hist.setHistogramRange(float(image.min()), float(image.max()))
        self.vb.autoRange()

    @property
    def mode(self) -> str:
        return self.vb.mode

    def set_mode(self, mode: str):
        assert mode in ("select", "circle", "freehand", "point")
        self.vb.mode = mode
        cursor = {"select": QtCore.Qt.ArrowCursor, "circle": QtCore.Qt.CrossCursor,
                  "freehand": QtCore.Qt.CrossCursor, "point": QtCore.Qt.CrossCursor}[mode]
        self.glw.setCursor(cursor)

    def set_field_outline(self, corners: np.ndarray | None):
        if corners is None or len(corners) == 0:
            self.field_item.setData([], [])
            return
        c = np.vstack([corners, corners[:1]])
        self.field_item.setData(c[:, 0], c[:, 1])

    def _on_stroke(self, pts):
        arr = np.asarray(pts)
        if len(arr):
            self.stroke_item.setData(arr[:, 0], arr[:, 1])
        else:
            self.stroke_item.setData([], [])

    def clear_stroke(self):
        self.stroke_item.setData([], [])

    def _on_mouse_moved(self, scene_pos):
        if self.vb.sceneBoundingRect().contains(scene_pos):
            p = self.vb.mapSceneToView(scene_pos)
            self.mouseMoved.emit(p.x(), p.y())

    def add_item(self, item):
        self.vb.addItem(item)

    def remove_item(self, item):
        self.vb.removeItem(item)


def stroke_to_polygon(stroke, tolerance: float = 1.0) -> np.ndarray | None:
    pts = simplify_polyline(np.asarray(stroke, float), tolerance)
    if len(pts) < 3:
        return None
    return pts
