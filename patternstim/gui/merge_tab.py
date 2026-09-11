"""Merge tab: append pattern sets (pos + neg frames) to an existing bin file."""
from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from ..binfile import read_header
from ..config import Config
from ..merge import merge_bin, plan_merge
from ..patterns import pattern_stem, NPZ_SUFFIX


class MergeTab(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Appends the <b>pos</b> and <b>neg</b> frame of every mask in the selected pattern sets to the "
            "end of the original bin. The original file is never modified and its frame indices are "
            "unchanged. An <code>.indices.json</code> next to the output lists the new indices for your vec script.")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QtWidgets.QGridLayout()
        layout.addLayout(form)
        form.addWidget(QtWidgets.QLabel("Original bin"), 0, 0)
        self.edit_bin = QtWidgets.QLineEdit()
        self.edit_bin.textChanged.connect(self.refresh)
        form.addWidget(self.edit_bin, 0, 1)
        b = QtWidgets.QPushButton("Browse…")
        b.clicked.connect(self.browse_bin)
        form.addWidget(b, 0, 2)
        self.lbl_header = QtWidgets.QLabel("")
        form.addWidget(self.lbl_header, 1, 1)

        form.addWidget(QtWidgets.QLabel("Pattern sets"), 2, 0, QtCore.Qt.AlignTop)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(110)
        form.addWidget(self.list, 2, 1)
        vb = QtWidgets.QVBoxLayout()
        b_add = QtWidgets.QPushButton("Add…")
        b_add.clicked.connect(self.add_patterns)
        b_rm = QtWidgets.QPushButton("Remove")
        b_rm.clicked.connect(self.remove_pattern)
        vb.addWidget(b_add)
        vb.addWidget(b_rm)
        vb.addStretch(1)
        form.addLayout(vb, 2, 2)

        form.addWidget(QtWidgets.QLabel("Output bin"), 3, 0)
        self.edit_out = QtWidgets.QLineEdit()
        form.addWidget(self.edit_out, 3, 1)
        b = QtWidgets.QPushButton("Browse…")
        b.clicked.connect(self.browse_out)
        form.addWidget(b, 3, 2)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["frame", "pos_idx", "neg_idx", "pixels on"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.lbl_summary = QtWidgets.QLabel("")
        row.addWidget(self.lbl_summary, stretch=1)
        self.btn_merge = QtWidgets.QPushButton("Merge")
        self.btn_merge.setEnabled(False)
        self.btn_merge.clicked.connect(self.merge)
        row.addWidget(self.btn_merge)
        layout.addLayout(row)

    # ------------------------------------------------------------------
    def pattern_files(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def browse_bin(self):
        start = self.config.last_dir or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Original bin", start, "Bin files (*.bin)")
        if path:
            self.edit_bin.setText(path)
            self.config.last_dir = str(Path(path).parent)

    def add_patterns(self, paths=None):
        if not paths:
            start = self.config.last_dir or str(Path.home())
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Pattern sets", start, f"Pattern masks (*{NPZ_SUFFIX});;NPZ (*.npz);;All files (*)")
        for p in paths or []:
            p = str(Path(str(pattern_stem(p)) + NPZ_SUFFIX)) if not str(p).endswith(".npz") else str(p)
            if p not in self.pattern_files():
                self.list.addItem(p)
        self.refresh()

    def remove_pattern(self):
        for li in self.list.selectedItems():
            self.list.takeItem(self.list.row(li))
        self.refresh()

    def browse_out(self):
        start = self.edit_out.text() or self.edit_bin.text() or self.config.last_dir or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Output bin", start, "Bin files (*.bin)")
        if path:
            self.edit_out.setText(path)

    def _suggest_out(self):
        src = self.edit_bin.text().strip()
        files = self.pattern_files()
        if src and files and not self.edit_out.text().strip():
            names = "_".join(pattern_stem(f).name for f in files)
            self.edit_out.setText(str(Path(src).with_name(f"{Path(src).stem}_{names}.bin")))

    def refresh(self):
        src = self.edit_bin.text().strip()
        self.table.setRowCount(0)
        self.btn_merge.setEnabled(False)
        if not src or not Path(src).exists():
            self.lbl_header.setText("" if not src else "<span style='color:#d33'>file not found</span>")
            self.lbl_summary.setText("")
            return
        try:
            h = read_header(src)
        except Exception as e:  # noqa: BLE001
            self.lbl_header.setText(f"<span style='color:#d33'>{e}</span>")
            return
        self.lbl_header.setText(f"{h['nb_images']} frames of {h['xsize']}x{h['ysize']} px, {h['nb_bits']} bits")
        files = self.pattern_files()
        if not files:
            self.lbl_summary.setText("Add at least one pattern set.")
            return
        try:
            plan = plan_merge(src, files)
        except Exception as e:  # noqa: BLE001
            self.lbl_summary.setText(f"<span style='color:#d33'>{e}</span>")
            return
        self.table.setRowCount(len(plan["entries"]))
        for r, e in enumerate(plan["entries"]):
            for c, v in enumerate([e["frame"], e["pos_idx"], e["neg_idx"], e["n_pixels"]]):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.lbl_summary.setText(f"{plan['n_original']} original + {plan['n_new']} new = "
                                 f"{plan['n_original'] + plan['n_new']} frames")
        self._suggest_out()
        self.btn_merge.setEnabled(True)

    def merge(self):
        src, out = self.edit_bin.text().strip(), self.edit_out.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(self, "No output", "Choose an output bin path.")
            return
        if Path(out).exists():
            if QtWidgets.QMessageBox.question(self, "Overwrite?", f"{out} exists. Overwrite it?") != QtWidgets.QMessageBox.Yes:
                return
        try:
            m = merge_bin(src, self.pattern_files(), out)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Merge failed", str(e))
            return
        self.config.last_dir = str(Path(out).parent)
        self.config.save()
        self.status.emit(f"Merged: {m['output_bin']} ({m['n_total_frames']} frames)")
        QtWidgets.QMessageBox.information(
            self, "Merged",
            f"{m['output_bin']}\n{m['n_original_frames']} original + "
            f"{m['n_total_frames'] - m['n_original_frames']} new frames\n\nIndices: {m['indices_json']}")
