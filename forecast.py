"""PGXC's own point forecast + thermal ceiling layer, from the same Météo-France
open GRIBs as the wind/rain overlays (no key, no quota).

Per valid hour, on a 0.05° node grid over BOUNDS (ARPEGE everywhere, AROME on
top where it has data):
  terrain : the model's own ground (geopotential of the lowest height level)
  ceiling : parcel on virtual potential temperature, starting 60 m above
            ground with +1.0 K — the rule that reproduces Meteo-Parapente's
            BL height from its own profile (r 0.996, see meteo-study)
  w*      : Deardorff convective velocity from the instantaneous sensible
            heat flux (accumulated flux over [t-1h, t+1h]); no scaling
  rain    : mm in the hour ending at t
and on a 0.1° node grid: wind + cloud at fixed altitudes (m AMSL), plus 10 m
wind. The app draws the windgram from these, with Allen's (2006) updraft
profile w(z) = w* (z/zi)^(1/3) (1 - 1.1 z/zi) — what Meteo-Parapente plots.

Outputs (under site/):
  thermal/<YYYYMMDDHH>.png        thermal ceiling (m AMSL), transparent without thermals
  wind/<YYYYMMDDHH>_<3500..5000>.png/.bin   wind above 3000 m AGL (from pressure levels)
  fc/<YYYYMMDD>/<tile>.bin.gz     1°x1° tiles for the windgram, layout in write_tiles()
                                  (cloud = densest cloud within ±125 m of each 250 m level)
"""

import datetime as dt
import gzip
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

np.seterr(all="ignore")  # NaN-heavy columns; results are masked explicitly

import build
from build import AROME, ARPEGE, BOUNDS, OUT, log

G, RD, CP = 9.80665, 287.05, 1005.0
START_AGL, EXCESS = 60.0, 1.0
MIN_WSTAR = 0.3  # below this there's no usable thermal

STEP_T = 0.05  # thermal grid (nodes), degrees
STEP_P = 0.1  # profile grid (nodes), degrees
TILE = 1.0  # tile size, degrees
NT = round(TILE / STEP_T)  # nodes per tile side
NP = round(TILE / STEP_P)
PROFILE_ALTS = list(range(250, 7001, 250))  # m AMSL
CLOUD_BAND = 125  # m: each level keeps the densest cloud within ± this
EXTRA_WIND_AGL = [3500, 4000, 4500, 5000]
HOURS = range(4, 22)  # UTC hours kept in the point tiles
UPPER_PA = [85000, 80000, 75000, 70000, 65000, 60000, 55000, 50000, 45000, 40000, 35000]

LAT_T = BOUNDS["south"] + np.arange(round((BOUNDS["north"] - BOUNDS["south"]) / STEP_T) + 1) * STEP_T
LON_T = BOUNDS["west"] + np.arange(round((BOUNDS["east"] - BOUNDS["west"]) / STEP_T) + 1) * STEP_T
LAT_P = BOUNDS["south"] + np.arange(round((BOUNDS["north"] - BOUNDS["south"]) / STEP_P) + 1) * STEP_P
LON_P = BOUNDS["west"] + np.arange(round((BOUNDS["east"] - BOUNDS["west"]) / STEP_P) + 1) * STEP_P

# Thermal ceiling palette (m AMSL) — same stops as the app's legend bar
# (.thermix-weather-legend-bar, 0..4000 m)
CEILING = [(0, (10, 61, 143)), (480, (26, 158, 219)), (960, (53, 196, 192)), (1600, (47, 174, 98)),
           (2200, (122, 214, 84)), (2720, (211, 232, 74)), (3120, (246, 225, 58)), (3480, (242, 168, 58)),
           (3760, (232, 99, 47)), (4000, (179, 18, 18))]

# Where each variable lives: (package, category, number, level type)
VARS = {
    "arome": {
        "t": ("HP1", 0, 0, 103), "p": ("HP1", 3, 0, 103), "u": ("HP1", 2, 2, 103), "v": ("HP1", 2, 3, 103),
        "td": ("HP2", 0, 6, 103), "cl": ("HP2", 6, 32, 103), "z": ("HP2", 3, 4, 103),
        "T": ("IP1", 0, 0, 100), "Z": ("IP1", 3, 4, 100), "U": ("IP1", 2, 2, 100), "V": ("IP1", 2, 3, 100),
        "TD": ("IP3", 0, 6, 100), "CL": ("IP2", 6, 32, 100),
        "t2": ("SP1", 0, 0, 103), "td2": ("SP2", 0, 6, 103), "ps": ("SP2", 3, 0, 1),
        "rain": ("SP1", 1, 52, 1), "shf": ("SP3", 0, 11, 1),
    },
    "arpege": {
        "t": ("HP1", 0, 0, 103), "p": ("HP1", 3, 0, 103), "u": ("HP1", 2, 2, 103), "v": ("HP1", 2, 3, 103),
        "td": ("HP2", 0, 6, 103), "cl": ("HP2", 6, 32, 103), "z": ("HP2", 3, 4, 103),
        "T": ("IP1", 0, 0, 100), "Z": ("IP1", 3, 4, 100), "U": ("IP1", 2, 2, 100), "V": ("IP1", 2, 3, 100),
        "TD": ("IP2", 0, 6, 100), "CL": ("IP3", 6, 32, 100),
        "t2": ("SP1", 0, 0, 103), "td2": ("SP2", 0, 6, 103), "ps": ("SP2", 3, 0, 1),
        "rain": ("SP1", 1, 52, 1), "shf": ("SP2", 0, 11, 1),
        "u10": ("SP1", 2, 2, 103), "v10": ("SP1", 2, 3, 103),
    },
}
ACCUMULATED = {"rain", "shf"}


# ---------- download ----------

def group_of(model, fc):
    for g in model["groups"]:
        a, b = (int(x) for x in g.replace("H", " ").split())
        if a <= fc <= b:
            return g
    return None


def match(var_map, pkg, meta):
    """Name of the variable a GRIB message of package `pkg` is, or None."""
    key = (pkg, meta.get("category"), meta.get("number"), meta.get("level_type"))
    for name, spec in var_map.items():
        lt = spec[3]
        if spec == key:
            if name in ("t2", "td2") and meta.get("level") != 2:
                continue
            if name in ("u10", "v10") and meta.get("level") != 10:
                continue
            if lt == 100 and meta.get("level") not in UPPER_PA:
                continue
            if name in ACCUMULATED and meta.get("pdt") != 8:
                continue
            if name not in ACCUMULATED and meta.get("pdt") == 8:
                continue
            return name
    return None


class Sampler:
    """Nearest native-grid index for every node of the thermal and profile grids."""

    def __init__(self, grid):
        self.grid = grid

    def idx(self, lats, lons):
        g = self.grid
        la, lo = np.meshgrid(lats, lons, indexing="ij")
        j = np.rint((g["la1"] - la) / g["dj"]).astype(int)
        i = np.rint(((lo - g["lo1"]) % 360) / g["di"]).astype(int)
        ok = (j >= 0) & (j < g["nj"]) & (i >= 0) & (i < g["ni"])
        return np.clip(j, 0, g["nj"] - 1), np.clip(i, 0, g["ni"] - 1), ok


def fetch_model(model, run, fcs):
    """{fc: {var: {level: (thermal-grid values, profile-grid values)}}} for the wanted
    forecast hours, plus accumulated fields at every hour needed for the differences."""
    name = model["name"]
    var_map = VARS[name]
    # accumulations are hourly early on but only 3-hourly later in ARPEGE's run
    acc_fcs = sorted({f for fc in fcs for f in range(fc - 3, fc + 4) if f > 0})
    groups = sorted({group_of(model, f) for f in set(fcs) | set(acc_fcs) if group_of(model, f)})
    pkgs = sorted({v[0] for v in var_map.values()})

    def index(job):
        pkg, g = job
        u = build.url(model, run, pkg, g)
        out = []
        for off, length, meta in build.grib_index.index(u):  # cached per build
            var = match(var_map, pkg, meta)
            if var is None:
                continue
            fc = int((meta["end"] - run).total_seconds() // 3600) if var in ACCUMULATED else meta.get("fc")
            if (var in ACCUMULATED and fc in acc_fcs) or (var not in ACCUMULATED and fc in fcs):
                out.append((u, off, length, meta, var, fc))
        return out

    with ThreadPoolExecutor(12) as pool:
        items = [it for lst in pool.map(index, [(p, g) for g in groups for p in pkgs]) for it in lst]
    log(f"forecast {name} {run:%Y-%m-%d %HZ}: {len(items)} messages for {len(fcs)} hours "
        f"(~{sum(i[2] for i in items) / 1e9:.1f} GB)")

    samplers = {}
    out = {}

    def decode(it):
        u, off, length, meta, var, fc = it
        meta2, grid, field = build.decode_remote((u, off, length, meta))
        return var, fc, meta.get("level", 0), grid, field

    with ThreadPoolExecutor(8) as pool:
        for var, fc, level, grid, field in pool.map(decode, items):
            key = (grid["ni"], grid["nj"], grid["la1"], grid["lo1"])
            if key not in samplers:
                s = Sampler(grid)
                samplers[key] = (s.idx(LAT_T, LON_T), s.idx(LAT_P, LON_P))
            (jt, it_, okt), (jp, ip, okp) = samplers[key]
            vt = np.where(okt, field[jt, it_], np.nan).astype(np.float32)
            vp = np.where(okp, field[jp, ip], np.nan).astype(np.float32)
            out.setdefault(fc, {}).setdefault(var, {})[level] = (vt, vp)
    return out


# ---------- physics (vectorised over grid nodes) ----------

def theta_v(tk, tdk, p_pa):
    """Virtual potential temperature (K). tdk may contain NaN (dry air assumed)."""
    th = tk * (100000.0 / p_pa) ** 0.2857
    tdc = np.nan_to_num(tdk - 273.15, nan=-80.0)
    e = 6.112 * np.exp(17.67 * tdc / (tdc + 243.5))
    q = 0.622 * e / (p_pa / 100 - 0.378 * e)
    return th * (1 + 0.61 * q)


def column(d, which):
    """Stack the height-above-ground levels (which=0 thermal grid, 1 profile grid)
    and the pressure levels above them into (agl[L,N], T, Td, p, u, v, cl) arrays."""
    hp = sorted(d["t"])
    ter = None
    lowest = hp[0]
    if lowest in d.get("z", {}):
        ter = d["z"][lowest][which] / G - lowest
    rows = []
    for a in hp:
        t = d["t"][a][which]
        p = d["p"][a][which] if a in d.get("p", {}) else None
        if p is None:
            continue
        td = d["td"][a][which] if a in d.get("td", {}) else np.full_like(t, np.nan)
        u = d["u"][a][which] if a in d.get("u", {}) else np.full_like(t, np.nan)
        v = d["v"][a][which] if a in d.get("v", {}) else np.full_like(t, np.nan)
        cl = d["cl"][a][which] if a in d.get("cl", {}) else np.zeros_like(t)
        rows.append((np.full_like(t, a), t, td, p, u, v, cl))
    top = hp[-1]
    for pa in sorted(d.get("T", {}), reverse=True):  # high pressure (low) first
        if pa not in d.get("Z", {}):
            continue
        agl = d["Z"][pa][which] / G - ter
        t = d["T"][pa][which]
        td = d["TD"][pa][which] if pa in d.get("TD", {}) else np.full_like(t, np.nan)
        u = d["U"][pa][which] if pa in d.get("U", {}) else np.full_like(t, np.nan)
        v = d["V"][pa][which] if pa in d.get("V", {}) else np.full_like(t, np.nan)
        cl = d["CL"][pa][which] if pa in d.get("CL", {}) else np.zeros_like(t)
        # only where this pressure level sits above the height levels' top
        agl = np.where(agl > top + 50, agl, np.nan)
        rows.append((agl, t, td, np.full_like(t, float(pa)), u, v, cl))
    arr = [np.stack(x) for x in zip(*rows)]
    return ter, arr


def interp_levels(agl, val, target):
    """Linear interpolation of val[L,N] at target[N] (m AGL); agl rows ascending
    except NaN gaps. NaN where the target is outside the column."""
    out = np.full(target.shape, np.nan, dtype=np.float32)
    prev_a, prev_v = agl[0], val[0]
    for k in range(1, agl.shape[0]):
        a, v = agl[k], val[k]
        ok = ~np.isnan(a)
        seg = ok & ~np.isnan(prev_a) & (target >= prev_a) & (target <= a) & np.isnan(out)
        f = np.where(seg, (target - prev_a) / np.where(a != prev_a, a - prev_a, 1), 0)
        out = np.where(seg, prev_v + (v - prev_v) * f, out)
        prev_a = np.where(ok, a, prev_a)
        prev_v = np.where(ok, v, prev_v)
    return out


def parcel_zi(agl, th):
    """Ceiling (m AGL): first crossing of theta_v(60 m) + 1 K going up."""
    base = interp_levels(agl, th, np.full(agl.shape[1:], START_AGL, dtype=np.float32)) + EXCESS
    zi = np.full(agl.shape[1:], np.nan, dtype=np.float32)
    prev_a = np.full(agl.shape[1:], START_AGL, dtype=np.float32)
    prev_t = base - EXCESS
    last = np.full(agl.shape[1:], np.nan, dtype=np.float32)
    for k in range(agl.shape[0]):
        a, t = agl[k], th[k]
        ok = ~np.isnan(a) & ~np.isnan(t) & (a > START_AGL)
        cross = ok & np.isnan(zi) & (t > base)
        f = np.where(cross, (base - prev_t) / np.where(t != prev_t, t - prev_t, 1), 0)
        zi = np.where(cross, prev_a + (a - prev_a) * f, zi)
        prev_a = np.where(ok, a, prev_a)
        prev_t = np.where(ok, t, prev_t)
        last = np.where(ok, a, last)
    zi = np.where(np.isnan(zi), last, zi)  # never capped inside the column: its top
    return np.where(np.isnan(base), np.nan, np.maximum(zi, 0))


def hour_fields(fc_data, acc, fc, run_fc0):
    """One model, one forecast hour -> dict of thermal-grid + profile-grid arrays."""
    d = fc_data
    if not all(k in d for k in ("t", "p", "t2", "ps")):
        return None
    ter, (agl, t, td, p, u, v, cl) = column(d, 0)
    if ter is None:
        return None
    th = theta_v(t, td, p)
    zi = parcel_zi(agl, th)

    def acc_at(f, which):
        if f == run_fc0:
            return 0.0
        a = acc.get(("shf", f))
        return None if a is None else a[which]

    # Instantaneous flux ~ centred difference of the accumulation (J/m²,
    # negative = upward) over the narrowest interval the model publishes:
    # ±1 h, or wider where ARPEGE only has it every 3 hours.
    shf = None
    for k in (1, 2, 3):
        a0, a1 = acc_at(fc - k, 0), acc_at(fc + k, 0)
        if a0 is not None and a1 is not None:
            shf = -(a1 - a0) / (2 * k * 3600.0)
            break
    if shf is None:  # end of the run: one-sided
        for k in (1, 2, 3):
            a0, ac = acc_at(fc - k, 0), acc_at(fc, 0)
            if a0 is not None and ac is not None:
                shf = -(ac - a0) / (k * 3600.0)
                break
    if shf is None:
        shf = np.zeros_like(zi)
    t2 = d["t2"][2][0]
    ps = d["ps"][0][0]
    rho = ps / (RD * t2)
    buoy = G / t2 * np.clip(shf, 0, None) / (rho * CP) * zi
    wstar = np.cbrt(np.nan_to_num(buoy))
    r1, r0 = acc.get(("rain", fc)), acc.get(("rain", fc - 1))
    if r1 is not None:
        rain = np.clip(r1[0] - (r0[0] if r0 is not None else 0.0), 0, None) if (r0 is not None or fc - 1 == run_fc0) else None
    else:
        rain = None

    # extra map wind levels above the 3000 m height levels (m AGL)
    extra = {}
    for lvl in EXTRA_WIND_AGL:
        tgt = np.full(zi.shape, float(lvl), dtype=np.float32)
        extra[lvl] = (interp_levels(agl, u, tgt), interp_levels(agl, v, tgt))

    # profile grid: wind + cloud at fixed altitudes AMSL, and 10 m wind
    pter, (pagl, _, _, _, pu, pv, pcl) = column(d, 1)
    prof_u, prof_v, prof_c = [], [], []
    for alt in PROFILE_ALTS:
        tgt = (alt - pter).astype(np.float32)
        below = tgt < pagl[0]
        prof_u.append(np.where(below, np.nan, interp_levels(pagl, pu, tgt)))
        prof_v.append(np.where(below, np.nan, interp_levels(pagl, pv, tgt)))
        # densest cloud near this altitude, so thin layers between two
        # output levels aren't lost; AROME/ARPEGE give it as a 0..1 fraction
        near = np.abs(pagl - tgt[None]) <= CLOUD_BAND
        dense = np.where(near, np.nan_to_num(pcl, nan=-1.0), -1.0).max(axis=0)
        c = np.fmax(interp_levels(pagl, pcl, tgt), np.where(dense < 0, np.nan, dense))
        prof_c.append(np.where(below, np.nan, c))
    if "u10" in d and 10 in d["u10"]:
        u10, v10 = d["u10"][10][1], d["v10"][10][1]
    elif 10 in d.get("u", {}):
        u10, v10 = d["u"][10][1], d["v"][10][1]
    else:
        u10, v10 = pu[0], pv[0]
    return {
        "ter": ter, "zi": zi, "wstar": wstar, "rain": rain, "extra": extra,
        "pter": pter, "u10": u10, "v10": v10,
        "pu": np.stack(prof_u), "pv": np.stack(prof_v), "pc": np.stack(prof_c),
    }


# ---------- merge + encode ----------

class Hour:
    """Quantised output for one valid hour (ARPEGE first, AROME overwrites)."""

    def __init__(self):
        nt, npf = (len(LAT_T), len(LON_T)), (len(LAT_P), len(LON_P))
        self.ter = np.full(nt, -32768, np.int16)
        self.zi = np.full(nt, 65535, np.uint16)
        self.ws = np.zeros(nt, np.uint8)
        self.rain = np.full(nt, 255, np.uint8)
        self.xu = {l: np.full(nt, np.nan, np.float32) for l in EXTRA_WIND_AGL}
        self.xv = {l: np.full(nt, np.nan, np.float32) for l in EXTRA_WIND_AGL}
        self.pter = np.full(npf, -32768, np.int16)
        self.puv = np.full((len(PROFILE_ALTS) + 1, 2) + npf, -128, np.int8)  # [0] = 10 m wind
        self.pc = np.zeros((len(PROFILE_ALTS),) + npf, np.uint8)
        self.models = set()

    def put(self, f, name):
        shape_t = self.ter.shape
        ok = ~np.isnan(f["ter"].reshape(shape_t)) & ~np.isnan(f["zi"].reshape(shape_t))
        self.ter = np.where(ok, np.clip(np.rint(f["ter"].reshape(shape_t)), -500, 9000), self.ter).astype(np.int16)
        self.zi = np.where(ok, np.clip(np.rint(np.nan_to_num(f["zi"].reshape(shape_t))), 0, 9000), self.zi).astype(np.uint16)
        self.ws = np.where(ok, np.clip(np.rint(f["wstar"].reshape(shape_t) / 0.02), 0, 254), self.ws).astype(np.uint8)
        if f["rain"] is not None:
            r = f["rain"].reshape(shape_t)
            self.rain = np.where(ok & ~np.isnan(r), np.clip(np.rint(r * 10), 0, 254), self.rain).astype(np.uint8)
        for l in EXTRA_WIND_AGL:
            u, v = f["extra"][l][0].reshape(shape_t), f["extra"][l][1].reshape(shape_t)
            good = ok & ~np.isnan(u)
            self.xu[l] = np.where(good, u, self.xu[l])
            self.xv[l] = np.where(good, v, self.xv[l])
        shape_p = self.pter.shape
        okp = ~np.isnan(f["pter"].reshape(shape_p))
        self.pter = np.where(okp, np.clip(np.rint(f["pter"].reshape(shape_p)), -500, 9000), self.pter).astype(np.int16)

        def q(x):
            x = x.reshape(shape_p)
            return np.where(np.isnan(x), -128, np.clip(np.rint(x * 2), -127, 127)).astype(np.int8)

        self.puv[0, 0] = np.where(okp, q(f["u10"]), self.puv[0, 0])
        self.puv[0, 1] = np.where(okp, q(f["v10"]), self.puv[0, 1])
        for k in range(len(PROFILE_ALTS)):
            self.puv[k + 1, 0] = np.where(okp, q(f["pu"][k]), self.puv[k + 1, 0])
            self.puv[k + 1, 1] = np.where(okp, q(f["pv"][k]), self.puv[k + 1, 1])
            c = f["pc"][k].reshape(shape_p)
            self.pc[k] = np.where(okp, np.clip(np.rint(np.nan_to_num(c) * 100), 0, 100), self.pc[k]).astype(np.uint8)
        self.models.add(name)


def compute(model, run, hours_wanted, out):
    fcs = sorted(int((t - run).total_seconds() // 3600) for t in hours_wanted)
    last = int(model["groups"][-1].split("H")[1])
    fcs = [f for f in fcs if 0 < f <= last]
    if not fcs:
        return
    # fetch in chunks of hours so memory stays bounded
    for k in range(0, len(fcs), 6):
        chunk = fcs[k:k + 6]
        data = fetch_model(model, run, chunk)
        acc = {}
        for f, dd in data.items():
            for var in ACCUMULATED:
                if var in dd:
                    acc[(var, f)] = next(iter(dd[var].values()))
        for fc in chunk:
            if fc not in data:
                continue
            fields = hour_fields(data[fc], acc, fc, 0)
            if fields is None:
                continue
            t = run + dt.timedelta(hours=fc)
            out.setdefault(t, Hour()).put(fields, model["name"])
        log(f"  {model['name']}: {min(k + 6, len(fcs))}/{len(fcs)} hours")


# ---------- writers ----------

def merc_rows(lats_desc):
    ys = np.log(np.tan(np.pi / 4 + np.radians(lats_desc) / 2))
    target = np.linspace(ys[0], ys[-1], len(lats_desc))
    return np.clip(np.searchsorted(-ys, -target), 0, len(lats_desc) - 1)


ROWS_T = merc_rows(LAT_T[::-1])


def save_png(rgba_north_first, path):
    from PIL import Image

    im = Image.fromarray(rgba_north_first[ROWS_T].astype(np.uint8), "RGBA")
    im = im.quantize(colors=96, method=Image.Quantize.FASTOCTREE)
    im.save(path, optimize=True)


def write_thermal_png(t, h):
    top = h.ter.astype(np.float32) + h.zi.astype(np.float32)
    on = (h.ter > -32768) & (h.zi < 65535) & (h.ws >= MIN_WSTAR / 0.02)
    rgb = build.ramp(np.where(on, top, 0), CEILING)
    alpha = np.where(on, 255, 0)
    save_png(np.dstack([rgb, alpha])[::-1], f"{OUT}/thermal/{t:%Y%m%d%H}.png")


def write_extra_wind(t, h):
    step = round(build.ARROW_STEP / STEP_T)
    for l in EXTRA_WIND_AGL:
        u, v = h.xu[l], h.xv[l]
        spd = np.hypot(u, v) * 3.6
        rgb = build.ramp(np.nan_to_num(spd), build.WIND)
        alpha = np.where(np.isnan(spd), 0, 255)
        key = f"{t:%Y%m%d%H}_{l}"
        save_png(np.dstack([rgb, alpha])[::-1], f"{OUT}/wind/{key}.png")
        # arrows: same 0.2° grid as build.write_wind, rows north -> south
        uu = np.nan_to_num(u[::-1][::step, ::step]) * 2
        vv = np.nan_to_num(v[::-1][::step, ::step]) * 2
        np.stack([np.clip(uu, -127, 127), np.clip(vv, -127, 127)]).astype(np.int8).tofile(f"{OUT}/wind/{key}.bin")


def write_tiles(date, hours_by_utc):
    """One gzip file per 1°x1° tile with every kept UTC hour of `date`.

    Little-endian, all nodes south->north then west->east (row-major):
      header  4 s "PGXF", u8 version=1, u8 first UTC hour, u8 hour count,
              u8 thermal nodes per side (NT), u8 profile nodes per side (NP),
              u8 profile level count (L, incl. the 10 m level), 6 bytes padding
      thermal hours x NT*NT x [int16 terrain m, uint16 ceiling m AGL,
              uint8 w* (0.02 m/s), uint8 rain (0.1 mm; 255 none)]  (terrain -32768 = no data)
      profile hours x NP*NP x [int16 terrain m, L x (int8 u, int8 v) (0.5 m/s; -128 none),
              (L-1) x uint8 cloud %]
    Levels: 10 m above ground, then PROFILE_ALTS m AMSL.
    """
    first, count = HOURS.start, len(HOURS)
    L = len(PROFILE_ALTS) + 1
    n_lat = round((BOUNDS["north"] - BOUNDS["south"]) / TILE)
    n_lon = round((BOUNDS["east"] - BOUNDS["west"]) / TILE)
    os.makedirs(f"{OUT}/fc/{date}", exist_ok=True)
    empty = Hour()
    hs = [hours_by_utc.get(h, empty) for h in HOURS]
    tdt = np.dtype([("ter", "<i2"), ("zi", "<u2"), ("ws", "u1"), ("rain", "u1")])
    pdt = np.dtype([("ter", "<i2"), ("uv", "i1", (L, 2)), ("cl", "u1", (L - 1,))])
    written = 0
    for ti in range(n_lat):
        for tj in range(n_lon):
            rt, ct = slice(ti * NT, ti * NT + NT), slice(tj * NT, tj * NT + NT)
            rp, cp = slice(ti * NP, ti * NP + NP), slice(tj * NP, tj * NP + NP)
            if all((h.ter[rt, ct] == -32768).all() for h in hs):
                continue
            th = np.zeros((count, NT, NT), tdt)
            pr = np.zeros((count, NP, NP), pdt)
            for k, h in enumerate(hs):
                th["ter"][k], th["zi"][k] = h.ter[rt, ct], h.zi[rt, ct]
                th["ws"][k], th["rain"][k] = h.ws[rt, ct], h.rain[rt, ct]
                pr["ter"][k] = h.pter[rp, cp]
                pr["uv"][k] = np.moveaxis(h.puv[:, :, rp, cp], (0, 1), (2, 3))
                pr["cl"][k] = np.moveaxis(h.pc[:, rp, cp], 0, 2)
            head = b"PGXF" + bytes([1, first, count, NT, NP, L]) + bytes(6)
            with gzip.open(f"{OUT}/fc/{date}/{ti}_{tj}.bin.gz", "wb", compresslevel=6) as f:
                f.write(head + th.tobytes() + pr.tobytes())
            written += 1
    return written


class BadForecast(RuntimeError):
    """The computed forecast fails a sanity check: don't publish it."""


def validate(by_date):
    """Refuse to publish an obviously broken forecast (the previous good one
    stays online because the deploy step never runs). Thresholds are loose on
    purpose: they catch "a whole day with no thermals / no clouds / no hours",
    which real weather over Western Europe never produces."""
    problems = []
    daytime = {d: hs for d, hs in by_date.items() if any(h in hs for h in range(10, 16))}
    if len(daytime) < 3:
        problems.append(f"only {len(daytime)} days with daytime hours")
    for date, hs in sorted(daytime.items()):
        mid = [hs[h] for h in range(9, 16) if h in hs]
        land = (mid[0].ter > 100) & (mid[0].ter != -32768)
        if land.mean() < 0.2:
            problems.append(f"{date}: model terrain covers {land.mean():.0%} of the grid")
            continue
        thermal = np.zeros(land.shape, bool)
        for h in mid:
            thermal |= h.ws >= MIN_WSTAR / 0.02
        frac = thermal[land].mean()
        if frac < 0.01:
            problems.append(f"{date}: thermals on {frac:.1%} of land")
        cloud = max(float((h.pc > 0).mean()) for h in hs.values())
        if cloud < 0.001:
            problems.append(f"{date}: no cloud anywhere")
        zi = np.median(np.concatenate([h.zi[land & (h.zi < 65535)] for h in mid]))
        if not 50 <= zi <= 5000:
            problems.append(f"{date}: median ceiling {zi:.0f} m AGL")
        log(f"check {date}: thermals on {frac:.0%} of land, cloud {cloud:.1%}, median ceiling {zi:.0f} m")
    if problems:
        raise BadForecast("; ".join(problems))


def run_all(now, arome_run, arpege_run):
    """Compute and write everything; returns the manifest section."""
    os.makedirs(f"{OUT}/thermal", exist_ok=True)
    first = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=1)
    first_day = first.replace(hour=0)
    out = {}
    today = [first_day + dt.timedelta(hours=h) for h in HOURS]
    for model, run in ((ARPEGE, arpege_run), (AROME, arome_run)):  # AROME last: it wins
        if not run:
            continue
        # A run only forecasts from its own start on, so today's earlier hours
        # come from the newest run older than all of them (one fetch), before
        # this run's hours so those win where both exist.
        before = [t for t in today if t <= run]
        if before:
            prev = build.latest_complete_run(model, before[0] - dt.timedelta(hours=1))
            if prev:
                compute(model, prev, before, out)
        last = run + dt.timedelta(hours=int(model["groups"][-1].split("H")[1]))
        wanted = []
        t = first_day
        while t <= last:
            if t.hour in HOURS and t > run:
                wanted.append(t)
            t += dt.timedelta(hours=1)
        compute(model, run, wanted, out)

    by_date = {}
    for t in sorted(out):
        by_date.setdefault(t.strftime("%Y%m%d"), {})[t.hour] = out[t]
    validate(by_date)

    hours = []
    for t in sorted(out):
        h = out[t]
        if t >= first:
            write_thermal_png(t, h)
            write_extra_wind(t, h)
            hours.append({"t": t.strftime("%Y%m%d%H"), "model": sorted(h.models)})
    dates = []
    for date, hs in sorted(by_date.items()):
        n = write_tiles(date, hs)
        dates.append({"date": date, "hours": sorted(hs)})
        log(f"forecast tiles {date}: {n} tiles, hours {sorted(hs)}")
    return {
        "version": 1,
        "tile": TILE,
        "thermalStep": STEP_T,
        "profileStep": STEP_P,
        "profileAlts": PROFILE_ALTS,
        "firstHour": HOURS.start,
        "hourCount": len(HOURS),
        "dates": dates,
        "thermalHours": hours,
        "extraWindLevels": EXTRA_WIND_AGL,
    }
