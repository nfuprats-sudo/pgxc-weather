"""Minimal pure-Python GRIB2 reader — just enough for Météo-France AROME packages.

eccodes has no Windows wheel for this Python, so: walk the messages, read the
product definition (discipline/category/number, level, forecast hour) and
decode simple-packed (template 5.0) fields on a regular lat/lon grid.

  python grib2.py file.grib2          -> list the messages
"""

import struct
import sys

import_numpy = None


def _np():
    global import_numpy
    if import_numpy is None:
        import numpy

        import_numpy = numpy
    return import_numpy


def _signed(v, nbytes):
    """GRIB2 sign-magnitude integers (top bit = sign)."""
    top = 1 << (8 * nbytes - 1)
    return -(v & (top - 1)) if v & top else v


def messages(path):
    """Yield dicts with metadata + raw sections for each GRIB2 message."""
    with open(path, "rb") as f:
        buf = f.read()
    pos = 0
    while True:
        pos = buf.find(b"GRIB", pos)
        if pos < 0:
            return
        total = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
        msg = buf[pos : pos + total]
        discipline = msg[6]
        p = 16
        meta = {"discipline": discipline}
        grid = None
        while p < total - 4:
            length = struct.unpack(">I", msg[p : p + 4])[0]
            num = msg[p + 4]
            sec = msg[p : p + length]
            if num == 3:
                tmpl = struct.unpack(">H", sec[12:14])[0]
                if tmpl == 0:
                    ni, nj = struct.unpack(">II", sec[30:38])
                    la1, lo1 = _signed(struct.unpack(">I", sec[46:50])[0], 4), _signed(struct.unpack(">I", sec[50:54])[0], 4)
                    la2, lo2 = _signed(struct.unpack(">I", sec[55:59])[0], 4), _signed(struct.unpack(">I", sec[59:63])[0], 4)
                    di, dj = struct.unpack(">II", sec[63:71])
                    grid = {
                        "ni": ni, "nj": nj, "la1": la1 / 1e6, "lo1": lo1 / 1e6, "la2": la2 / 1e6,
                        "lo2": lo2 / 1e6, "di": di / 1e6, "dj": dj / 1e6, "scan": sec[71],
                    }
                meta["grid_template"] = tmpl
            elif num == 4:
                tmpl = struct.unpack(">H", sec[7:9])[0]
                meta.update(
                    pdt=tmpl, category=sec[9], number=sec[10],
                    time_unit=sec[17], fc=struct.unpack(">I", sec[18:22])[0],
                    level_type=sec[22], level_scale=sec[23], level=struct.unpack(">I", sec[24:28])[0],
                )
            elif num == 5:
                meta["npoints"] = struct.unpack(">I", sec[5:9])[0]
                meta["drt"] = struct.unpack(">H", sec[9:11])[0]
                meta["sec5"] = sec
            elif num == 6:
                meta["bitmap"] = sec[5]
                meta["sec6"] = sec
            elif num == 7:
                meta["sec7"] = sec
            p += length
        meta["grid"] = grid
        yield meta
        pos += total


def decode(m):
    """Values as a (nj, ni) float array, NaN where the bitmap masks a point.
    Supports simple packing (DRT 5.0)."""
    np = _np()
    if m["drt"] not in (0, 42):
        raise NotImplementedError(f"data representation template 5.{m['drt']}")
    s5 = m["sec5"]
    ref = struct.unpack(">f", s5[11:15])[0]
    e = _signed(struct.unpack(">H", s5[15:17])[0], 2)
    d = _signed(struct.unpack(">H", s5[17:19])[0], 2)
    nbits = s5[19]
    npacked = m["npoints"]
    raw = m["sec7"][5:]
    if nbits == 0:
        packed = np.zeros(npacked)
    elif m["drt"] == 42:
        # CCSDS / libaec (Météo-France's default), via imagecodecs
        import imagecodecs

        flags, block, rsi = s5[21], s5[22], struct.unpack(">H", s5[23:25])[0]
        nbytes = (nbits + 7) // 8
        if nbytes == 3 and not flags & 2:  # AEC_DATA_3BYTE unset -> 4-byte samples
            nbytes = 4
        # libaec decodes whole reference-sample intervals: leave room for padding
        out = imagecodecs.aec_decode(
            raw, bitspersample=nbits, flags=flags, blocksize=block, rsi=rsi,
            out=(npacked + block * rsi) * nbytes,
        )
        b = np.frombuffer(out, dtype=np.uint8)[: npacked * nbytes].reshape(npacked, nbytes).astype(np.uint64)
        packed = np.zeros(npacked, dtype=np.uint64)
        for k in range(nbytes):  # MSB first (flag AEC_DATA_MSB)
            packed = (packed << np.uint64(8)) | b[:, k]
    else:
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[: npacked * nbits].reshape(npacked, nbits)
        weights = 1 << np.arange(nbits - 1, -1, -1, dtype=np.uint64)
        packed = bits.astype(np.uint64) @ weights
    vals = (ref + packed.astype(np.float64) * 2.0**e) / 10.0**d
    g = m["grid"]
    n = g["ni"] * g["nj"]
    if m.get("bitmap") == 0:
        mask = np.unpackbits(np.frombuffer(m["sec6"][6:], dtype=np.uint8))[:n].astype(bool)
        full = np.full(n, np.nan)
        full[mask] = vals
        vals = full
    return vals.reshape(g["nj"], g["ni"])


def value_at(m, field, lat, lon):
    """Nearest grid point value (grid assumed regular lat/lon, rows north->south
    when scan flag bit 0x40 is 0)."""
    g = m["grid"]
    j = round((g["la1"] - lat) / g["dj"]) if not g["scan"] & 0x40 else round((lat - g["la1"]) / g["dj"])
    i = round(((lon - g["lo1"]) % 360) / g["di"])
    if not (0 <= i < g["ni"] and 0 <= j < g["nj"]):
        return None
    v = field[j, i]
    return None if v != v else float(v)


if __name__ == "__main__":
    for m in messages(sys.argv[1]):
        print(
            f"d{m['discipline']} c{m.get('category')} n{m.get('number')} lvltype {m.get('level_type')} "
            f"lvl {m.get('level')} fc {m.get('fc')} drt 5.{m.get('drt')} grid {m['grid'] and (m['grid']['ni'], m['grid']['nj'])}"
        )
