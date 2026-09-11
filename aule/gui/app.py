"""Main window: Select | Merge | Calibration tabs, settings dialog."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6 import QtWidgets

from ..config import Config, DEFAULT_CONFIG_PATH
from .calibrate_tab import CalibrateTab
from .merge_tab import MergeTab
from .select_tab import SelectTab


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rig settings")
        form = QtWidgets.QFormLayout(self)
        self.rig = QtWidgets.QComboBox()
        self.rig.addItems(["2", "3"])
        self.rig.setCurrentText(str(config.rig_id))
        self.h = QtWidgets.QSpinBox(); self.h.setRange(8, 4096); self.h.setValue(config.dmd_height)
        self.w = QtWidgets.QSpinBox(); self.w.setRange(8, 4096); self.w.setValue(config.dmd_width)
        self.px = QtWidgets.QDoubleSpinBox(); self.px.setRange(0.01, 100); self.px.setDecimals(3)
        self.px.setValue(config.pixel_size_um)
        form.addRow("rig_id", self.rig)
        form.addRow("DMD frame height (rows)", self.h)
        form.addRow("DMD frame width (cols)", self.w)
        form.addRow("pixel size (µm)", self.px)
        self.age = QtWidgets.QSpinBox(); self.age.setRange(1, 3650); self.age.setSuffix(" days")
        self.age.setValue(config.calibration_max_age_days)
        form.addRow("warn if calibration older than", self.age)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def apply(self, config: Config):
        config.rig_id = int(self.rig.currentText())
        config.dmd_height = self.h.value()
        config.dmd_width = self.w.value()
        config.pixel_size_um = self.px.value()
        config.calibration_max_age_days = self.age.value()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Config, config_path=DEFAULT_CONFIG_PATH):
        super().__init__()
        self.config = config
        self.config_path = config_path
        if not config.path:
            config.path = str(config_path)
        self.setWindowTitle("Aule - patterned OSS stimulation")
        self.resize(1400, 900)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.calibrate_tab = CalibrateTab(config)
        self.select_tab = SelectTab(config)
        self.merge_tab = MergeTab(config)
        self.tabs.addTab(self.select_tab, "1. Select patterns")
        self.tabs.addTab(self.merge_tab, "2. Merge into bin")
        self.tabs.addTab(self.calibrate_tab, "Calibration")

        self.calibrate_tab.calibrationChanged.connect(self.select_tab.set_calibration)
        self.select_tab.calibrationTabRequested.connect(
            lambda: self.tabs.setCurrentWidget(self.calibrate_tab))
        for t in (self.calibrate_tab, self.select_tab, self.merge_tab):
            t.status.connect(self.statusBar().showMessage)

        menu = self.menuBar().addMenu("&Settings")
        act = menu.addAction("Rig / DMD settings…")
        act.triggered.connect(self.open_settings)
        self.statusBar().showMessage(f"rig {config.rig_id}, DMD {config.dmd_width}x{config.dmd_height} px, "
                                     f"{config.pixel_size_um} µm/px")

        self.tabs.setCurrentIndex(0)
        if config.last_calibration and Path(config.last_calibration).exists():
            self.calibrate_tab.load_calibration(config.last_calibration)

    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            dlg.apply(self.config)
            self.config.save(self.config_path)
            self.calibrate_tab.set_config(self.config)
            self.select_tab.set_config(self.config)
            self.statusBar().showMessage(f"rig {self.config.rig_id}, DMD {self.config.dmd_width}x"
                                         f"{self.config.dmd_height} px, {self.config.pixel_size_um} µm/px")

    def closeEvent(self, ev):
        self.config.save(self.config_path)
        super().closeEvent(ev)


def run(config: Config | None = None, config_path=DEFAULT_CONFIG_PATH) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    config = config or Config.load(config_path)
    win = MainWindow(config, config_path)
    win.show()
    return app.exec()
