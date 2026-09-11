# Aule — patterned OSS stimulation

Desktop tool to turn a microscope image (epi / 2p, e.g. GFP cells) into DMD frames for
patterned optogenetic stimulation, and to append those frames to an existing stimulus `.bin`.

For every drawn pattern the tool produces the **pattern frame** (light on the selected cells,
to be shown with the activating LED) and its **negative** (light everywhere else, to be shown
with the inactive LED) so that the total luminance is uniform over the retina.

## Install

One-click launchers (they create the `aule` conda env from `environment.yml` on first
use, then start the GUI): double-click `Aule.bat` on Windows, run `./Aule.sh` on
Linux. `Aule.desktop` can be copied to the desktop / `~/.local/share/applications` for a
menu entry (its `Exec` / `Path` lines hold the absolute path of this folder: edit them if it moves).

Manual install, dedicated conda environment (Python 3.14, numpy 2, PyQt6; Linux / Windows / macOS):

```
conda env create -f environment.yml
conda activate aule
```

Or with pip in any Python 3.11+ (virtualenv recommended):

```
pip install -r requirements.txt
```

Run everything from this folder:

```
python -m aule gui                 # the application
python -m aule calib-bin --out calibration_target_MEA3.bin
python -m aule merge --bin orig.bin --patterns cells.patterns.npz --out orig_cells.bin
python -m aule info some.bin
python -m pytest tests                    # self-tests
```

Rig defaults (rig 3, 672x672 frame, 2.5 µm/px) are editable in *Settings > Rig / DMD settings*
and stored in `~/.aule/config.json` (or `--config` / `--rig` / `--dmd-size` / `--pixel-size`
on the CLI).

## Workflow

The application opens on **1. Select patterns**; the calibration lives in its own tab and only
needs redoing after the optics changed (or at regular intervals). The date of the last
calibration is shown at the top right of the Select tab (green = recent, orange = older than
*warn if calibration older than* in the settings, red = none).

1. **Calibration** (tab *Calibration*)
   * Display frame 0 of the calibration bin (`calibration_target_MEA3.bin`, kept in the common
     stimulus folder; *Export calibration bin* regenerates it, or `python -m aule calib-bin`).
     The target is a cross at the centre (centring), a ring (scale), a bar towards +x (rotation)
     and a dot in the upper-right quadrant (mirror check).
   * Image the projected target with the camera in the same configuration as the images used to
     select cells, then *Load camera image*.
   * Align the magenta template on the projected light: drag it to centre the cross, corner
     handles scale + rotate, the right-edge handle rotates only, tick *mirror* if the dot lands on
     the wrong side of the bar. Fine-tune with the centre / scale / angle boxes or *Refine
     automatically* (local optimisation of the placement on the image intensity; the *template
     signal* should get close to 1).
   * *Save calibration*: the placement (centre, scale, angle, mirror) is the calibration, saved
     as JSON with its date. It reloads automatically next time. *Use identity* if the image is
     already in DMD frame coordinates.
2. **Select patterns** (live)
   * *Load image* (TIFF 8/16-bit, PNG, JPG; channels selectable, z-stacks max-projected).
     Contrast with the histogram on the right of the image, zoom with the wheel, pan with the
     right mouse button or in *Select* mode.
   * `C` circle: click drops a spot of the chosen radius (µm). `F` freehand: drag around a cell.
     `S` select: drag shapes, handles to resize / move vertices. `Del` removes, `Ctrl+Z` undoes.
   * Shapes can be renamed (double-click) and disabled (untick) in the list. *margin* dilates
     every shape by that many µm in the DMD frame.
   * The preview shows the real DMD frame (image warped into DMD space, pattern in red, toggle
     *show negative*). Shapes leaving the DMD field are flagged.
   * *Export*: choose a folder; writes `<name>.patterns.npz` (the masks — source of truth),
     `<name>.patterns.json` (shapes, calibration, image path; reloadable with *Load pattern set*)
     and `<name>_previews/` PNGs. Tick *combined* for one frame with all shapes and/or
     *per shape* for one frame per shape.
   * Everything is autosaved to `~/.aule/autosave.patterns.json`.
3. **Merge into bin** (GUI tab or CLI, usable on any number of bins with the exact same masks)
   * Pick the original bin, add one or more `.patterns.npz`, pick an output path.
   * The output is the original frames byte for byte, followed by `pos, neg` for each mask.
     The original bin is never modified, so all existing vec indices stay valid.
   * `<output>.indices.json` lists `pos_idx` / `neg_idx` for every frame — use these in the vec
     generation script in place of the full-field background index.

## Frame conventions

* Frames are written through the same `BinFile.append` path as the grating stimuli (rig 3:
  polarity inverted then flipped horizontally), so a pattern mask value of 1 means light on.
* `aule/binfile.py` is the lab reader/writer updated for numpy 2 (`np.float` and
  `np.fromstring` were removed) with `read_frame_uint8()` to get back the logical frame.

## Layout

```
aule/
  config.py       rig / DMD settings
  binfile.py      bin reader / writer (+ raw byte copy)
  calibration.py  calibration target, similarity fit / refinement, Calibration JSON
  geometry.py     Circle / Polygon shapes, simplification, rasterisation to DMD masks
  patterns.py     PatternSet: frames (combined / per shape), export / load
  merge.py        append masks as pos/neg frames to a bin, index JSON
  images.py       image loading, warp to DMD space
  cli.py          gui | calib-bin | merge | info
  gui/            PyQt6 + pyqtgraph application (select / merge / calibration tabs)
tests/            pytest (core + offscreen GUI end-to-end)
```
