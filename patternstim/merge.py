"""Append pattern masks (pos + neg frames) to an existing bin file."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .binfile import BinFile, read_header, read_raw_frame_bytes
from .patterns import load_masks


def plan_merge(original_bin, pattern_files) -> dict:
    """Describe what a merge would do without writing anything."""
    header = read_header(original_bin)
    n0 = header["nb_images"]
    entries = []
    idx = n0
    for pf in pattern_files:
        names, masks, meta = load_masks(pf)
        if (header["ysize"], header["xsize"]) != tuple(meta["dmd_shape"]):
            raise ValueError(f"{Path(pf).name}: mask shape {meta['dmd_shape']} does not match bin "
                             f"({header['ysize']}, {header['xsize']})")
        for n, m in zip(names, masks):
            entries.append({"frame": n, "pattern_file": str(pf), "pos_idx": idx, "neg_idx": idx + 1,
                            "n_pixels": int(m.sum())})
            idx += 2
    return {"header": header, "n_original": n0, "n_new": idx - n0, "entries": entries}


def merge_bin(original_bin, pattern_files, out_bin, rig_id: int | None = None) -> dict:
    """Write ``out_bin`` = original frames (byte-exact) + pos/neg frames of each mask.

    Returns the index mapping (also written to ``<out_bin>.indices.json``).
    """
    original_bin, out_bin = Path(original_bin), Path(out_bin)
    if not original_bin.exists():
        raise FileNotFoundError(original_bin)
    if out_bin.resolve() == original_bin.resolve():
        raise ValueError("output must differ from the original bin (the original is never modified)")
    pattern_files = [Path(p) for p in pattern_files]
    if not pattern_files:
        raise ValueError("no pattern file given")

    plan = plan_merge(original_bin, pattern_files)
    header = plan["header"]
    if header["nb_bits"] != 8:
        raise ValueError(f"unsupported nb_bits={header['nb_bits']}")
    n_total = plan["n_original"] + plan["n_new"]
    if n_total > 65535:
        raise ValueError("resulting bin would exceed 65535 frames (int16 header)")

    loaded = [load_masks(pf) for pf in pattern_files]
    rig = rig_id
    if rig is None:
        rigs = {m["rig_id"] for _, _, m in loaded}
        if len(rigs) != 1:
            raise ValueError(f"pattern files come from different rigs: {sorted(rigs)}")
        rig = rigs.pop()

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_bin.with_suffix(out_bin.suffix + ".tmp")
    with BinFile(str(tmp), header["xsize"], header["ysize"], rig_id=rig, nb_images=n_total, mode="w") as bf:
        with open(original_bin, "rb") as src:
            src.seek(8)
            frame_bytes = header["xsize"] * header["ysize"]
            for _ in range(plan["n_original"]):
                chunk = src.read(frame_bytes)
                if len(chunk) != frame_bytes:
                    raise ValueError("original bin is truncated")
                bf.append(chunk)
        for names, masks, _ in loaded:
            for m in masks:
                pos = m.astype(float)
                bf.append(pos)
                bf.append(1.0 - pos)
    os.replace(tmp, out_bin)

    mapping = {
        "output_bin": str(out_bin),
        "original_bin": str(original_bin),
        "pattern_files": [str(p) for p in pattern_files],
        "rig_id": rig,
        "n_original_frames": plan["n_original"],
        "n_total_frames": n_total,
        "frames": {e["frame"]: {"pos_idx": e["pos_idx"], "neg_idx": e["neg_idx"],
                                "n_pixels": e["n_pixels"], "pattern_file": e["pattern_file"]}
                   for e in plan["entries"]},
    }
    json_path = Path(str(out_bin) + ".indices.json")
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)
    mapping["indices_json"] = str(json_path)
    return mapping
