"""Fast chunked LAS reading plus laspy-based writing.

The training files are LAS 1.4, point format 7, record length 48:
  offset 0  : X int32
  offset 4  : Y int32
  offset 8  : Z int32
  offset 12 : intensity uint16
  offset 14 : return flags uint8
  offset 15 : classification flags / scanner channel uint8
  offset 16 : classification uint8
  offset 17 : user data uint8
  offset 18 : scan angle int16
  offset 20 : point source id uint16
  offset 22 : GPS time float64
  offset 30 : Red uint16, Green uint16, Blue uint16
  offset 36 : extra bytes (12 bytes: float32 "Original cloud index" + float64)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class LasHeader:
    path: str
    version: tuple[int, int]
    header_size: int
    point_offset: int
    point_format: int
    record_len: int
    n_points: int
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    bounds: tuple[float, float, float, float, float, float]  # minx,miny,minz,maxx,maxy,maxz
    n_vlr: int


def read_header(path: str | Path) -> LasHeader:
    path = str(path)
    with open(path, "rb") as f:
        # LAS 1.4 header is 375 bytes; LAS 1.0-1.3 is 227 bytes.  Reading the
        # whole file here was unnecessary and could be multi-GB for large LAS.
        h = f.read(375)
        if len(h) < 227 or h[:4] != b"LASF":
            raise ValueError(f"not a LAS file: {path}")

    version = (h[24], h[25])
    header_size = struct.unpack_from("<H", h, 94)[0]
    point_offset = struct.unpack_from("<I", h, 96)[0]
    n_vlr = struct.unpack_from("<I", h, 100)[0]
    point_format = h[104] & 0x3F
    record_len = struct.unpack_from("<H", h, 105)[0]
    n_legacy = struct.unpack_from("<I", h, 107)[0]
    scales = struct.unpack_from("<3d", h, 131)
    offsets = struct.unpack_from("<3d", h, 155)
    maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", h, 179)
    n_ext = struct.unpack_from("<Q", h, 247)[0] if version >= (1, 4) else 0
    n_points = int(n_ext if n_ext else n_legacy)
    return LasHeader(
        path=path,
        version=version,
        header_size=header_size,
        point_offset=point_offset,
        point_format=point_format,
        record_len=record_len,
        n_points=n_points,
        scales=tuple(float(x) for x in scales),  # type: ignore[arg-type]
        offsets=tuple(float(x) for x in offsets),  # type: ignore[arg-type]
        bounds=(float(minx), float(miny), float(minz), float(maxx), float(maxy), float(maxz)),
        n_vlr=int(n_vlr),
    )


def iter_chunks(
    path: str | Path,
    chunk_points: int = 2_000_000,
    with_rgb: bool = True,
    with_extra: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield raw point fields in chunks.

    Returned arrays are views/copies of the raw little-endian values; raw
    coordinates are int32 and must be scaled by the header scales/offsets.
    """
    hdr = read_header(path)
    if hdr.point_format not in (6, 7, 8):
        raise ValueError(
            f"unsupported point format {hdr.point_format} in {path}; "
            "this project expects LAS 1.4 point format 6/7/8"
        )
    # format 6: no RGB; 7: RGB; 8: RGB + NIR
    rgb_off = 30 if hdr.point_format in (7, 8) else None
    extra_off = {6: 30, 7: 36, 8: 38}[hdr.point_format]
    if hdr.record_len < extra_off:
        raise ValueError(f"record_len {hdr.record_len} too small for format {hdr.point_format}")

    with open(path, "rb") as f:
        f.seek(hdr.point_offset)
        done = 0
        while done < hdr.n_points:
            m = min(chunk_points, hdr.n_points - done)
            buf = np.fromfile(f, dtype=np.uint8, count=m * hdr.record_len)
            if buf.size != m * hdr.record_len:
                raise IOError(f"short read in {path}: got {buf.size}, expected {m * hdr.record_len}")
            buf = buf.reshape(m, hdr.record_len)

            out: Dict[str, np.ndarray] = {
                "x": buf[:, 0:4].copy().view("<i4").ravel(),
                "y": buf[:, 4:8].copy().view("<i4").ravel(),
                "z": buf[:, 8:12].copy().view("<i4").ravel(),
                "intensity": buf[:, 12:14].copy().view("<u2").ravel(),
                "return_byte": buf[:, 14].copy(),
                "flags": buf[:, 15].copy(),
                "classification": buf[:, 16].copy(),
                "row_offset": np.arange(done, done + m, dtype=np.int64),
            }
            if with_rgb and rgb_off is not None:
                out["rgb"] = buf[:, rgb_off:rgb_off + 6].copy().view("<u2").reshape(m, 3)
            else:
                out["rgb"] = np.zeros((m, 3), dtype=np.uint16)
            if with_extra:
                out["extra_f32"] = buf[:, extra_off:extra_off + 4].copy().view("<f4").ravel()
            else:
                out["extra_f32"] = np.zeros(m, dtype=np.float32)
            yield out
            done += m


def read_all(path: str | Path, chunk_points: int = 2_000_000) -> Dict[str, np.ndarray]:
    """Read all points into memory (convenience for smaller files)."""
    parts: Dict[str, list[np.ndarray]] = {}
    for chunk in iter_chunks(path, chunk_points=chunk_points):
        for key, value in chunk.items():
            parts.setdefault(key, []).append(value)
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def to_meters(hdr: LasHeader, raw: np.ndarray, axis: int) -> np.ndarray:
    return raw.astype(np.float64) * hdr.scales[axis] + hdr.offsets[axis]


def point_keys(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Encode raw int32 LAS coordinates into a uint64 key.

    The raw coordinates in this dataset are non-negative and fit in 24/20/20
    bits respectively (X ~ 6.8e6, Y ~ 1.1e6, Z ~ 1.6e4).  The encoding is
    exact and is only used for duplicate/conflict detection.
    """
    xu = x.astype(np.uint64, copy=False)
    yu = y.astype(np.uint64, copy=False)
    zu = z.astype(np.uint64, copy=False)
    if np.any(xu >= (1 << 24)) or np.any(yu >= (1 << 20)) or np.any(zu >= (1 << 20)):
        raise ValueError("raw coordinates exceed the 24/20/20-bit key encoding range")
    return (xu << 40) | (yu << 20) | zu


def write_predictions_las(
    input_path: str | Path,
    output_path: str | Path,
    classification: np.ndarray,
    prob: Optional[np.ndarray] = None,
    prob_field: str = "pred_prob",
) -> None:
    """Write a copy of the input LAS with classification (and optional prob).

    Requires laspy; imported lazily so the rest of the package does not depend
    on it.  The original CRS, scale/offset, point order, RGB and other fields
    are preserved by laspy.
    """
    import laspy  # lazy import

    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    las = laspy.read(input_path)
    n = len(las.points)
    if len(classification) != n:
        raise ValueError(f"classification length {len(classification)} != LAS points {n}")
    las.classification = np.asarray(classification, dtype=np.uint8)
    existing = {dim.name for dim in las.point_format.extra_dimensions}

    if prob is not None:
        if len(prob) != n:
            raise ValueError(f"prob length {len(prob)} != LAS points {n}")
        if prob_field not in existing:
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=prob_field,
                    type=np.float32,
                    description="PointNet building probability",
                )
            )
            existing.add(prob_field)
        las[prob_field] = np.asarray(prob, dtype=np.float32)

    # LAS classification 0/1 is this project's convention, not a standard LAS
    # building code.  Also write a self-describing extra field.
    if "pred_building" not in existing:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="pred_building",
                type=np.uint8,
                description="1=building, 0=non-building",
            )
        )
    las["pred_building"] = np.asarray(classification, dtype=np.uint8)

    las.write(output_path)


def laspy_extra_dims(path: str | Path) -> list[str]:
    import laspy

    las = laspy.read(path)
    return [dim.name for dim in las.point_format.extra_dimensions]
