"""Bin file reader/writer for the DMD stimulation software.

Format: header of 4 little-endian int16 (xsize, ysize, nb_images, nb_bits=8) followed by
raw uint8 frames of xsize*ysize bytes each.

This is a copy of the lab's ``binfile.py`` with two fixes for numpy >= 1.24
(``np.float`` -> ``float``, ``np.fromstring`` -> ``np.frombuffer``) and a helper to copy
frames byte for byte. The rig transforms on write/read are kept identical so that frames
appended here look exactly like frames produced by the existing stimulus scripts.
"""
from __future__ import annotations

import os
import numpy as np

HEADER_BYTES = 2 * 4
RIG_MAX_DIMS = {2: (1920, 1080), 3: (1024, 768)}


def read_header(path) -> dict:
    with open(path, mode="rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: file too short to contain a bin header")
    xsize, ysize, nb_images, nb_bits = np.frombuffer(raw, dtype="<u2")
    return {"xsize": int(xsize), "ysize": int(ysize),
            "nb_images": int(nb_images), "nb_bits": int(nb_bits)}


def read_raw_frame_bytes(path, frame_nb: int) -> bytes:
    """Return the stored bytes of one frame, untouched (no rig transform)."""
    h = read_header(path)
    assert 0 <= frame_nb < h["nb_images"], frame_nb
    frame_bytes = h["xsize"] * h["ysize"]
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES + frame_bytes * frame_nb)
        return f.read(frame_bytes)


def apply_write_transform(frame: np.ndarray, rig_id: int) -> np.ndarray:
    """Float frame in [0, 1] -> uint8 array as stored in the file (rig polarity + flip)."""
    frame = np.asarray(frame, dtype=np.float64)
    if rig_id == 3:
        frame = 1 - frame
        frame = np.fliplr(frame)
    elif rig_id == 2:
        frame = np.rot90(np.flipud(frame), k=3)
    else:
        raise ValueError(f"unknown rig_id: {rig_id}")
    return (frame * 255).astype(np.uint8)


class BinFile:

    @classmethod
    def read_header(cls, path):
        return read_header(path)

    @classmethod
    def read_nb_images(cls, path):
        return read_header(path)["nb_images"]

    def __init__(self, path, frame_xsize=None, frame_ysize=None, rig_id=3,
                 nb_images=0, reverse=False, mode="r"):
        self._path = path
        self._reverse = reverse
        self._mode = mode

        assert rig_id in RIG_MAX_DIMS, f"unknown rig_id: {rig_id}"
        self._rig_id = rig_id
        self._max_dimension_x, self._max_dimension_y = RIG_MAX_DIMS[rig_id]

        if self._mode == "r":
            header = read_header(self._path)
            self._nb_images = header["nb_images"]
            assert header["xsize"] <= self._max_dimension_x, \
                f"image is too big on x axis for RIG {self._rig_id} ({header['xsize']} > {self._max_dimension_x})"
            self._frame_xsize = header["xsize"]
            assert header["ysize"] <= self._max_dimension_y, \
                f"image is too big on y axis for RIG {self._rig_id} ({header['ysize']} > {self._max_dimension_y})"
            self._frame_ysize = header["ysize"]
            self._nb_bits = header["nb_bits"]
            self._file = open(self._path, mode="rb")
            self._frame_nb = self._nb_images - 1
        elif self._mode == "w":
            assert frame_xsize is not None and frame_ysize is not None
            self._nb_images = nb_images
            assert frame_xsize <= self._max_dimension_x, \
                f"image is too big on x axis for RIG {self._rig_id}: {frame_xsize} > {self._max_dimension_x}"
            self._frame_xsize = frame_xsize
            assert frame_ysize <= self._max_dimension_y, \
                f"image is too big on y axis for RIG {self._rig_id}: {frame_ysize} > {self._max_dimension_y}"
            self._frame_ysize = frame_ysize
            self._nb_bits = 8
            self._file = open(self._path, mode="wb")
            self._write_header()
            self._frame_nb = -1
        else:
            raise ValueError(f"unknown mode value: {self._mode}")

        self._counter = 0

    # -- context manager -------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- sequence protocol ----------------------------------------------
    def __len__(self):
        return self._nb_images

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter < len(self):
            frame = self.read_frame(self._counter)
            self._counter += 1
            return frame
        raise StopIteration

    # -- properties -------------------------------------------------------
    @property
    def _frame_shape(self):
        return self._frame_xsize, self._frame_ysize

    @property
    def ysize(self):
        return self._frame_ysize

    @property
    def xsize(self):
        return self._frame_xsize

    @property
    def nb_frames(self):
        return self._nb_images

    @property
    def nb_bits(self):
        return self._nb_bits

    @property
    def rig_id(self):
        return self._rig_id

    def is_readable(self):
        return self._mode == "r"

    def is_writeable(self):
        return self._mode == "w"

    def get_frame_nb(self):
        """Get the number of the latest frame appended."""
        return self._frame_nb

    def get_frame_nbs(self):
        return np.arange(0, len(self))

    # -- reading ----------------------------------------------------------
    def read_frame_as_bytes(self, frame_nb):
        assert self.is_readable(), "not readable"
        assert 0 <= frame_nb < len(self), frame_nb
        assert self._nb_bits == 8, self._nb_bits
        frame_byte_size = self._frame_ysize * self._frame_xsize
        self._file.seek(HEADER_BYTES + frame_byte_size * frame_nb)
        return self._file.read(frame_byte_size)

    def read_frame(self, frame_nb, verbose=0):
        """Read, convert to float in [0, 1) and undo the rig transform."""
        frame_bytes = self.read_frame_as_bytes(frame_nb)
        frame_data = np.frombuffer(frame_bytes, dtype=np.uint8)
        if verbose > 1:
            print("max value = ", np.max(frame_data))

        frame_data = frame_data.astype(float)
        dinfo = np.iinfo(np.uint8)
        frame_data = frame_data / float(dinfo.max - dinfo.min + 1)

        # NOTE: the original reader reshapes to (xsize, ysize); kept for compatibility.
        frame_data = np.reshape(frame_data, (self._frame_xsize, self._frame_ysize))

        if self._rig_id == 3:
            frame_data = 1 - frame_data
            frame_data = np.fliplr(frame_data)
        elif self._rig_id == 2:
            frame_data = np.flipud(np.rot90(frame_data))
        return frame_data

    def read_frame_uint8(self, frame_nb) -> np.ndarray:
        """Read a frame as the logical uint8 array (0..255) that was passed to append.

        Undoes the rig polarity/flip so that ``read_frame_uint8(i)`` equals
        ``(frame * 255).astype(uint8)`` for the ``frame`` given to ``append``.
        """
        stored = np.frombuffer(self.read_frame_as_bytes(frame_nb), dtype=np.uint8)
        stored = stored.reshape((self._frame_ysize, self._frame_xsize))
        if self._rig_id == 3:
            return np.fliplr(255 - stored)
        if self._rig_id == 2:
            return np.flipud(np.rot90(stored))
        raise ValueError(self._rig_id)

    # -- writing ----------------------------------------------------------
    def _write_header(self):
        header_list = [self._frame_xsize, self._frame_ysize, self._nb_images, self._nb_bits]
        header_array = np.array(header_list, dtype="<i2")
        self._file.write(header_array.tobytes())

    def append(self, frame):
        """Append a frame. ``bytes`` are written verbatim; arrays are float in [0, 1]
        and go through the rig transform (polarity reversal + flip) then x255."""
        if isinstance(frame, (bytes, bytearray)):
            assert len(frame) == self._frame_ysize * self._frame_xsize, len(frame)
            self._file.write(bytes(frame))
        else:
            frame = np.asarray(frame)
            assert frame.shape == (self._frame_ysize, self._frame_xsize), \
                f"frame shape {frame.shape} != (ysize, xsize) = {(self._frame_ysize, self._frame_xsize)}"
            self._file.write(apply_write_transform(frame, self._rig_id).tobytes())
        self._frame_nb += 1

    def flush(self):
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        if self._file.closed:
            return
        if self.is_writeable():
            self.flush()
        self._file.close()
