"""Command line entry points: ``python -m patternstim <command>``."""
from __future__ import annotations

import argparse
import sys

from .config import Config, DEFAULT_CONFIG_PATH


def _add_config_args(p: argparse.ArgumentParser):
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config JSON (rig, DMD size, pixel size)")
    p.add_argument("--rig", type=int, help="override rig_id")
    p.add_argument("--dmd-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="override DMD frame size")
    p.add_argument("--pixel-size", type=float, help="override DMD pixel size (um)")


def _config_from_args(a) -> Config:
    cfg = Config.load(a.config)
    if a.rig is not None:
        cfg.rig_id = a.rig
    if a.dmd_size is not None:
        cfg.dmd_height, cfg.dmd_width = a.dmd_size
    if a.pixel_size is not None:
        cfg.pixel_size_um = a.pixel_size
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="patternstim", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gui = sub.add_parser("gui", help="launch the graphical interface")
    _add_config_args(p_gui)

    p_cal = sub.add_parser("calib-bin", help="write the calibration target bin (+ reference PNG)")
    p_cal.add_argument("--out", required=True, help="output .bin path")
    _add_config_args(p_cal)

    p_merge = sub.add_parser("merge", help="append pattern frames (pos+neg) to a bin file")
    p_merge.add_argument("--bin", required=True, dest="original", help="original bin (never modified)")
    p_merge.add_argument("--patterns", required=True, nargs="+", help="one or more .patterns.npz files")
    p_merge.add_argument("--out", required=True, help="output bin path")
    p_merge.add_argument("--rig", type=int, help="override rig_id used for the appended frames")

    p_info = sub.add_parser("info", help="print the header of a bin file")
    p_info.add_argument("bin")

    a = parser.parse_args(argv)

    if a.cmd == "gui":
        from .gui.app import run
        return run(_config_from_args(a), config_path=a.config)

    if a.cmd == "calib-bin":
        from .calibration import make_calibration_bin
        res = make_calibration_bin(a.out, _config_from_args(a))
        print(f"wrote {res['bin']} ({len(res['frames'])} frames: {', '.join(res['frames'])})")
        print(f"reference image: {res['png']}")
        print(f"display frame {res['target_frame']} (target) with the DMD software")
        return 0

    if a.cmd == "merge":
        from .merge import merge_bin
        m = merge_bin(a.original, a.patterns, a.out, rig_id=a.rig)
        print(f"wrote {m['output_bin']}: {m['n_original_frames']} original + "
              f"{m['n_total_frames'] - m['n_original_frames']} new frames")
        for name, e in m["frames"].items():
            print(f"  {name}: pos_idx={e['pos_idx']} neg_idx={e['neg_idx']} ({e['n_pixels']} px)")
        print(f"indices: {m['indices_json']}")
        return 0

    if a.cmd == "info":
        from .binfile import read_header
        h = read_header(a.bin)
        print(f"{a.bin}: {h['nb_images']} frames of {h['xsize']}x{h['ysize']} (x,y), {h['nb_bits']} bits")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
