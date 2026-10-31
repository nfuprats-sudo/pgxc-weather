"""Daily airspace cells for PGXC from OpenAIP's public per-country exports.

OpenAIP blocked bbox queries on its API (2026-10-24) and asks clients to do
geospatial filtering themselves, so we download every <cc>_asp.ndgeojson once
a day and cut it into 5x5 degree cells:

  site/airspace/<x>_<y>.json  list of raw OpenAIP airspace items whose box
                              touches cell x = floor(lng/5), y = floor(lat/5)

PGXC's /api/public/airspaces reads these cells and maps the items itself.
A missing cell means no airspace there. Data (c) OpenAIP, CC BY-NC 4.0.

  python airspace.py [outdir]     default outdir: site/airspace
"""

import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://storage.openaip.net/openaip-system-exports/"
CELL = 5  # must match CELL in PGXC's src/lib/airspace/openaip.server.ts
KEEP = ("_id", "name", "type", "icaoClass", "lowerLimit", "upperLimit")


def get(url: str, tries: int = 4) -> bytes:
    # The bucket answers 429 when hit too fast; back off and retry.
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise AssertionError


def countries() -> list[str]:
    out, token = [], ""
    while True:
        url = BASE + "?list-type=2" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        xml = get(url).decode()
        out += re.findall(r"<Key>([a-z]{2})_asp\.ndgeojson</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]*)", xml)
        if not m:
            return out
        token = m.group(1)


def rounded(coords):
    # ~10 m precision keeps files small; plenty for airspace edges.
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 4), round(coords[1], 4)]
    return [rounded(c) for c in coords]


def points(coords):
    if isinstance(coords[0], (int, float)):
        yield coords
    else:
        for c in coords:
            yield from points(c)


def cells_of(geometry) -> list[tuple[int, int]]:
    pts = list(points(geometry["coordinates"]))
    w, e = min(p[0] for p in pts), max(p[0] for p in pts)
    s, n = min(p[1] for p in pts), max(p[1] for p in pts)
    # shortcut: antimeridian-crossing shapes (span > 180) are dropped; split them if a Pacific pilot ever needs one.
    if e - w > 180:
        return []
    return [
        (x, y)
        for x in range(math.floor(w / CELL), math.floor(e / CELL) + 1)
        for y in range(math.floor(s / CELL), math.floor(n / CELL) + 1)
    ]


def main(outdir: Path) -> None:
    cells: dict[tuple[int, int], list] = defaultdict(list)
    ccs = countries()
    if len(ccs) < 50:
        # A broken listing must fail the build, not publish an empty world.
        raise SystemExit(f"only {len(ccs)} country exports listed")
    for cc in ccs:
        n = 0
        for line in get(f"{BASE}{cc}_asp.ndgeojson").splitlines():
            if not line.strip():
                continue
            f = json.loads(line)
            geom = f.get("geometry")
            props = f.get("properties") or {}
            if not geom or geom.get("type") not in ("Polygon", "MultiPolygon") or not props.get("_id"):
                continue
            item = {k: props[k] for k in KEEP if k in props}
            item["geometry"] = {"type": geom["type"], "coordinates": rounded(geom["coordinates"])}
            for c in cells_of(geom):
                cells[c].append(item)
            n += 1
        print(cc, n, flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.json"):
        old.unlink()
    for (x, y), items in cells.items():
        (outdir / f"{x}_{y}.json").write_text(json.dumps(items, separators=(",", ":")))
    print(f"{len(ccs)} countries -> {len(cells)} cells in {outdir}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "site/airspace"))
