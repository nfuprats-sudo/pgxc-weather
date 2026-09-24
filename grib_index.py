"""Index a remote GRIB2 file with HTTP range requests: read only each message's
header (sections 0-4), so big packages (HP1 ~700 MB) can be filtered down to the
few fields we need and fetched message by message."""

import datetime
import struct
import urllib.request

HEAD_BYTES = 1024


def _get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def index(url):
    """Yield (offset, length, meta) for every message, meta from sections 3/4."""
    off = 0
    while True:
        try:
            head = _get(url, off, off + HEAD_BYTES - 1)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # past the end
                return
            raise
        if len(head) < 16 or head[:4] != b"GRIB":
            return
        total = struct.unpack(">Q", head[8:16])[0]
        meta = {"discipline": head[6]}
        p = 16
        while p + 5 <= len(head):
            length = struct.unpack(">I", head[p : p + 4])[0]
            num = head[p + 4]
            if num == 4 and p + 28 <= len(head):
                sec = head[p : p + length]
                meta.update(
                    category=sec[9], number=sec[10], fc=struct.unpack(">I", sec[18:22])[0],
                    level_type=sec[22], level=struct.unpack(">I", sec[24:28])[0], pdt=struct.unpack(">H", sec[7:9])[0],
                )
                if meta["pdt"] == 8 and len(sec) >= 41:
                    # accumulation: end of the overall time interval (octets 35-40)
                    y = struct.unpack(">H", sec[34:36])[0]
                    meta["end"] = datetime.datetime(y, sec[36], sec[37], sec[38], tzinfo=datetime.timezone.utc)
                break
            if length == 0:
                break
            p += length
        yield off, total, meta
        off += total


def fetch(url, off, length):
    return _get(url, off, off + length - 1)
