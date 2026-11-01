"""Monthly thermal-presence cells from thermal.kk7.ch, for PGXC's route analysis.

KK7's admin agreed to PGXC keeping a monthly copy. The analysis only needs
"is there a detected thermal near this point", so instead of storing images
we keep a yes/no grid: each zoom-10 tile (~27-39 km wide) becomes 64x64
cells (~400-600 m), one bit each.

To avoid asking for a million tiles we look at the 4096 zoom-6 tiles first
and only fetch the zoom-10 tiles under the parts that show any thermal.

  site/kk7/manifest.json            months available, layout, stats
  site/kk7/<month>/<x6>_<y6>.bin    one file per zoom-6 tile with data:
      32 bytes: which of its 16x16 zoom-10 children have data (bit j*16+i)
      then 512 bytes per present child, in that order: 64x64 bits, row-major

  python kk7.py <outdir> [layer ...]    KK7 layers are quarters named after
                                        their first month (jan, apr, jul, oct)
                                        or "all"; default: this quarter, plus
                                        the next one in a quarter's last week
"""

import datetime as dt
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://thermal.kk7.ch/tiles/thermals_{m}_all/{z}/{x}/{y}.png?src=pgxc.app"
UA = "pgxc-weather/1.0 (+https://pgxc.app; monthly copy agreed with thermal.kk7.ch)"
QUARTERS = ["jan", "apr", "jul", "oct"]
Z0, Z = 6, 10
SUB = 1 << (Z - Z0)  # zoom-10 children per zoom-6 tile side (16)
GRID = 64  # cells per zoom-10 tile side
WORKERS = 4  # kept low on purpose: their server, our courtesy
ALPHA_MIN = 32


def get(month: str, z: int, x: int, y: int) -> bytes | None:
    # KK7 serves TMS tiles: y counts from the south.
    url = URL.format(m=month, z=z, x=x, y=(1 << z) - 1 - y)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # no thermals in this tile
            if attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
        time.sleep(3 * (attempt + 1))
    return None


def alpha(png: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    # Empty areas come back as tiny placeholder images; work on 256 px always.
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.NEAREST)
    return np.asarray(im)[:, :, 3]


def children_with_data(a: np.ndarray) -> list[tuple[int, int]]:
    """(i, j) of the zoom-10 children under any painted pixel of a zoom-6 tile."""
    block = a.shape[0] // SUB
    out = []
    for j in range(SUB):
        for i in range(SUB):
            if a[j * block:(j + 1) * block, i * block:(i + 1) * block].max() > 0:
                out.append((i, j))
    return out


def presence_bits(a: np.ndarray) -> bytes:
    """64x64 yes/no grid of a zoom-10 tile, packed row-major into 512 bytes."""
    n = a.shape[0] // GRID
    cells = (a[: n * GRID, : n * GRID] > ALPHA_MIN).reshape(GRID, n, GRID, n).any(axis=(1, 3))
    return np.packbits(cells.astype(np.uint8)).tobytes()


def build_month(month: str, outdir: Path, stats: dict) -> None:
    mdir = outdir / month
    mdir.mkdir(parents=True, exist_ok=True)
    s = stats[month] = {"z6_tiles": 0, "z6_with_data": 0, "z6_png_bytes": 0, "z10_tiles": 0, "z10_png_bytes": 0, "cells_bytes": 0}
    # Rows between ~80N and ~80S; nobody flies closer to the poles.
    rows = range(4, (1 << Z0) - 4)
    parents = [(x, y) for y in rows for x in range(1 << Z0)]
    with ThreadPoolExecutor(WORKERS) as pool:
        tiles6 = list(pool.map(lambda p: (p, get(month, Z0, *p)), parents))
    s["z6_tiles"] = len(parents)
    for (x6, y6), png6 in tiles6:
        if not png6:
            continue
        a6 = alpha(png6)
        kids = children_with_data(a6)
        if not kids:
            continue
        s["z6_with_data"] += 1
        s["z6_png_bytes"] += len(png6)
        with ThreadPoolExecutor(WORKERS) as pool:
            pngs = list(pool.map(lambda c: get(month, Z, x6 * SUB + c[0], y6 * SUB + c[1]), kids))
        header = bytearray(SUB * SUB // 8)
        body = bytearray()
        for (i, j), png in zip(kids, pngs):
            if not png:
                continue
            bits = presence_bits(alpha(png))
            s["z10_tiles"] += 1
            s["z10_png_bytes"] += len(png)
            if not any(bits):
                continue
            k = j * SUB + i
            header[k // 8] |= 0x80 >> (k % 8)
            body += bits
        if body:
            data = bytes(header) + bytes(body)
            (mdir / f"{x6}_{y6}.bin").write_bytes(data)
            s["cells_bytes"] += len(data)
        print(month, x6, y6, f"{len(kids)} children", flush=True)


def default_months(today: dt.date) -> list[str]:
    q = (today.month - 1) // 3
    layers = [QUARTERS[q]]
    # The forecast reaches a few days ahead: near a quarter's end, add the next.
    if today.month % 3 == 0 and today.day >= 24:
        layers.append(QUARTERS[(q + 1) % 4])
    return layers


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "site/kk7")
    months = sys.argv[2:] or default_months(dt.date.today())
    outdir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    for m in months:
        build_month(m, outdir, stats)
        # A month with no data at all means KK7 was unreachable or blocked us:
        # fail rather than publish "no thermals anywhere".
        if stats[m]["z6_with_data"] == 0:
            raise SystemExit(f"{m}: no KK7 data fetched")
    manifest = {
        "months": months,
        "z0": Z0,
        "z": Z,
        "grid": GRID,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stats": stats,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
