"""Build PGXC's wind + rain map overlays from Météo-France open data.

Every run (see .github/workflows/build.yml):
  1. pick the newest AROME/ARPEGE run whose files are all published
  2. index the GRIB packages with HTTP range requests and download only the
     messages we need (u/v wind at 10 m + 500..3000 m AGL, accumulated rain)
  3. composite ARPEGE (0.1°, to 102 h) under AROME (0.025°, to 51 h) on one
     0.025° grid, colour it with Meteo-Parapente's palette, resample to Web
     Mercator rows and write PNGs + a coarse u/v grid for the arrows
  4. write site/manifest.json, which PGXC reads to know what exists

Data: Météo-France, Licence Ouverte / Etalab 2.0 (attribution required).

  python build.py                 full build into site/
  python build.py --quick         one AROME group, surface wind only (local test)
"""

import datetime as dt
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import grib2
import grib_index

BASE = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt"
OUT = "site"

# Output grid: PGXC's Meteo-Parapente coverage box, at AROME's 0.025°
BOUNDS = {"south": 34.0, "north": 58.0, "west": -12.0, "east": 20.0}
RES = 0.025
NLAT = round((BOUNDS["north"] - BOUNDS["south"]) / RES) + 1
NLON = round((BOUNDS["east"] - BOUNDS["west"]) / RES) + 1
LATS = BOUNDS["north"] - np.arange(NLAT) * RES
LONS = BOUNDS["west"] + np.arange(NLON) * RES

LEVELS = [10, 500, 1000, 1500, 2000, 2500, 3000]  # m above ground; 10 = "surface"
UTC_HOURS = range(4, 21)  # pilots' day in Europe (06-22 local in summer)
ARROW_STEP = 0.2  # degrees between arrow samples

AROME = {
    "name": "arome",
    "path": "arome/0025",
    "groups": ["00H06H", "07H12H", "13H18H", "19H24H", "25H30H", "31H36H", "37H42H", "43H48H", "49H51H"],
    "runs_every": 3,
}
ARPEGE = {
    "name": "arpege",
    "path": "arpege/01",
    "groups": ["000H012H", "013H024H", "025H036H", "037H048H", "049H060H", "061H072H", "073H084H", "085H096H", "097H102H"],
    "runs_every": 6,
}

# Meteo-Parapente's wind palette (km/h) — same stops as WIND_SPEED_PALETTE in PGXC
WIND = [
    (0, (255, 255, 255)), (3.3, (0, 255, 255)), (6.7, (3, 230, 175)), (10.0, (5, 204, 95)),
    (13.3, (8, 179, 15)), (16.7, (88, 204, 10)), (20.0, (168, 228, 5)), (23.3, (248, 253, 0)),
    (26.7, (255, 228, 0)), (30.0, (255, 198, 0)), (33.3, (255, 168, 0)), (36.7, (255, 115, 0)),
    (40.0, (255, 60, 0)), (43.3, (255, 5, 0)), (46.7, (204, 0, 41)), (50.0, (148, 0, 87)),
]
# mm/h; transparent below the first stop
RAIN = [(0.1, (180, 235, 255)), (0.5, (90, 190, 255)), (1, (30, 130, 250)), (2, (20, 80, 220)),
        (5, (90, 40, 200)), (10, (170, 30, 170)), (20, (230, 30, 90))]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def url(model, run, pkg, group):
    iso = run.strftime("%Y-%m-%dT%H:00:00Z")
    return f"{BASE}/{iso}/{model['path']}/{pkg}/{model['name']}__{model['path'].split('/')[1]}__{pkg}__{group}__{iso}.grib2"


def exists(u):
    try:
        req = urllib.request.Request(u, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.URLError:
        return False


def latest_complete_run(model, now):
    """Newest run whose last package group is already published."""
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % model["runs_every"])
    for _ in range(8):
        if exists(url(model, t, "SP1", model["groups"][-1])) and exists(url(model, t, "HP1", model["groups"][-1])):
            return t
        t -= dt.timedelta(hours=model["runs_every"])
    return None


# ---------- download ----------

def wanted_messages(model, run, pkg, group, want):
    """Index one remote package and return [(url, off, len, meta)] matching want(meta)."""
    u = url(model, run, pkg, group)
    out = []
    for off, length, meta in grib_index.index(u):
        if want(meta):
            out.append((u, off, length, meta))
    return out


def decode_remote(item):
    u, off, length, meta = item
    raw = grib_index.fetch(u, off, length)
    path = f"/tmp/pgxc_{os.getpid()}_{off}.grib2" if os.name != "nt" else os.path.join(os.environ.get("TEMP", "."), f"pgxc_{off}.grib2")
    with open(path, "wb") as f:
        f.write(raw)
    try:
        m = next(grib2.messages(path))
        return meta, m["grid"], grib2.decode(m)
    finally:
        os.remove(path)


def regrid(field, grid):
    """Bilinear-sample a regular lat/lon field (rows north->south) onto the output grid.
    NaN outside the source domain."""
    lon_src = (LONS - grid["lo1"]) % 360 / grid["di"]
    lat_src = (grid["la1"] - LATS) / grid["dj"]
    nj, ni = field.shape
    x0 = np.floor(lon_src).astype(int)
    y0 = np.floor(lat_src).astype(int)
    fx = (lon_src - x0)[None, :]
    fy = (lat_src - y0)[:, None]
    valid = (x0 >= 0) & (x0 + 1 < ni)
    validy = (y0 >= 0) & (y0 + 1 < nj)
    xc = np.clip(x0, 0, ni - 2)
    yc = np.clip(y0, 0, nj - 2)
    a = field[yc][:, xc]
    b = field[yc][:, xc + 1]
    c = field[yc + 1][:, xc]
    d = field[yc + 1][:, xc + 1]
    out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    out[~validy, :] = np.nan
    out[:, ~valid] = np.nan
    return out


def collect(model, run, levels, quick=False):
    """{valid_time: {"u": {lvl: grid}, "v": {...}, "acc": grid}} for this model run."""
    run_hours = {}
    groups = model["groups"][:2] if quick else model["groups"]

    def valid(fc):
        return run + dt.timedelta(hours=fc)

    def want_hp1(m):
        return (
            m.get("category") == 2 and m.get("number") in (2, 3) and m.get("level_type") == 103
            and m.get("level") in levels and valid(m["fc"]).hour in UTC_HOURS
        )

    def want_sp1(m):
        if m.get("category") == 1 and m.get("number") == 52:
            return True  # accumulated rain: keep every hour, the hourly diff needs the previous one
        return (
            model is ARPEGE and 10 in levels and m.get("category") == 2 and m.get("number") in (2, 3)
            and m.get("level") == 10 and valid(m["fc"]).hour in UTC_HOURS
        )

    jobs = [("HP1", g, want_hp1) for g in groups] + [("SP1", g, want_sp1) for g in groups]
    with ThreadPoolExecutor(len(jobs)) as pool:
        lists = list(pool.map(lambda j: wanted_messages(model, run, j[0], j[1], j[2]), jobs))
    items = [it for lst in lists for it in lst]
    log(f"{model['name']} {run:%Y-%m-%d %HZ}: {len(items)} messages to fetch")

    acc = {}
    with ThreadPoolExecutor(16) as pool:
        for meta, grid, field in pool.map(decode_remote, items):
            if meta["category"] == 1:
                acc[meta["end"]] = regrid(field, grid)  # rain accumulated since the run start
                continue
            t = valid(meta["fc"])
            slot = run_hours.setdefault(t, {"u": {}, "v": {}})
            slot["u" if meta["number"] == 2 else "v"][meta["level"]] = regrid(field, grid)
    return run_hours, acc


def hourly_rain(acc, run):
    """{end_time: rain accumulated since the run start} -> {valid_time: mm in the hour ending then}."""
    rain = {}
    for t, total in acc.items():
        if t.hour not in UTC_HOURS:
            continue
        prev = acc.get(t - dt.timedelta(hours=1))
        if prev is None and t - dt.timedelta(hours=1) == run:
            prev = np.zeros_like(total)  # nothing accumulated yet at the run start
        if prev is not None:
            rain[t] = np.clip(total - prev, 0, None)
    return rain


# ---------- rendering ----------

def ramp(values, stops):
    xs = np.array([s for s, _ in stops], dtype=float)
    cols = np.array([c for _, c in stops], dtype=float)
    v = np.clip(values, xs[0], xs[-1])
    return np.stack([np.interp(v, xs, cols[:, k]) for k in range(3)], axis=-1)


def merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


MERC_ROWS = None


def to_mercator_rows(img):
    """Leaflet's imageOverlay stretches linearly in Web Mercator, not latitude."""
    global MERC_ROWS
    if MERC_ROWS is None:
        ys = merc_y(LATS)
        target = np.linspace(ys[0], ys[-1], NLAT)
        MERC_ROWS = np.clip(np.searchsorted(-ys, -target), 0, NLAT - 1)
    return img[MERC_ROWS]


def save_png(rgba, path):
    im = Image.fromarray(to_mercator_rows(rgba).astype(np.uint8), "RGBA")
    im = im.quantize(colors=96, method=Image.Quantize.FASTOCTREE)
    im.save(path, optimize=True)


def write_wind(t, lvl, u, v):
    spd = np.hypot(u, v) * 3.6
    rgb = ramp(np.nan_to_num(spd), WIND)
    alpha = np.where(np.isnan(spd), 0, 255)
    key = t.strftime("%Y%m%d%H")
    save_png(np.dstack([rgb, alpha]), f"{OUT}/wind/{key}_{lvl}.png")
    step = round(ARROW_STEP / RES)
    uu = np.nan_to_num(u[::step, ::step]) * 2  # int8 in 0.5 m/s steps
    vv = np.nan_to_num(v[::step, ::step]) * 2
    np.stack([np.clip(uu, -127, 127), np.clip(vv, -127, 127)]).astype(np.int8).tofile(f"{OUT}/wind/{key}_{lvl}.bin")


def write_rain(t, mm):
    rgb = ramp(np.nan_to_num(mm), RAIN)
    alpha = np.where(np.nan_to_num(mm) >= RAIN[0][0], 230, 0)
    save_png(np.dstack([rgb, alpha]), f"{OUT}/rain/{t:%Y%m%d%H}.png")


def main():
    quick = "--quick" in sys.argv
    now = dt.datetime.now(dt.timezone.utc)
    os.makedirs(f"{OUT}/wind", exist_ok=True)
    os.makedirs(f"{OUT}/rain", exist_ok=True)
    levels = [10] if quick else LEVELS

    arome_run = latest_complete_run(AROME, now)
    arpege_run = None if quick else latest_complete_run(ARPEGE, now)
    log("runs:", arome_run, arpege_run)

    layers = {}  # valid_time -> {"model", "run", "u", "v", "rain"}
    for model, run in ((ARPEGE, arpege_run), (AROME, arome_run)):  # AROME last: it wins
        if not run:
            continue
        winds, acc = collect(model, run, levels, quick)
        rain = hourly_rain(acc, run)
        for t, w in winds.items():
            slot = layers.setdefault(t, {"u": {}, "v": {}, "rain": None, "model": {}})
            for lvl in levels:
                if lvl in w["u"] and lvl in w["v"]:
                    base_u, base_v = slot["u"].get(lvl), slot["v"].get(lvl)
                    u, v = w["u"][lvl], w["v"][lvl]
                    if base_u is not None:  # keep ARPEGE where AROME has no data
                        u = np.where(np.isnan(u), base_u, u)
                        v = np.where(np.isnan(v), base_v, v)
                    slot["u"][lvl], slot["v"][lvl] = u, v
                    slot["model"][lvl] = model["name"]
        for t, mm in rain.items():
            slot = layers.setdefault(t, {"u": {}, "v": {}, "rain": None, "model": {}})
            slot["rain"] = mm if slot["rain"] is None else np.where(np.isnan(mm), slot["rain"], mm)

    first = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=1)
    hours = []
    for t in sorted(layers):
        if t < first:
            continue
        s = layers[t]
        for lvl in s["u"]:
            write_wind(t, lvl, s["u"][lvl], s["v"][lvl])
        if s["rain"] is not None:
            write_rain(t, s["rain"])
        hours.append(
            {
                "t": t.strftime("%Y%m%d%H"),
                "levels": sorted(s["u"]),
                "rain": s["rain"] is not None,
                "model": sorted(set(s["model"].values())),
            }
        )
    step = round(ARROW_STEP / RES)
    manifest = {
        "generated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "runs": {"arome": arome_run and arome_run.strftime("%Y%m%d%H"), "arpege": arpege_run and arpege_run.strftime("%Y%m%d%H")},
        "bounds": BOUNDS,
        "arrows": {"step": ARROW_STEP, "rows": len(range(0, NLAT, step)), "cols": len(range(0, NLON, step)), "scale": 0.5},
        "hours": hours,
        "attribution": "Météo-France (AROME, ARPEGE), Licence Ouverte",
    }
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    log(f"done: {len(hours)} hours")


if __name__ == "__main__":
    main()
