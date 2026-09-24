"""Index a remote GRIB2 file with HTTP range requests: read only each message's
header (sections 0-4), so big packages (HP1 ~700 MB) can be filtered down to the
few fields we need and fetched message by message."""

import datetime
import http.client
import struct
import threading
import urllib.error
import urllib.parse
import urllib.request

HEAD_BYTES = 1024


_local = threading.local()


def _get(url, start, end):
    """Range GET over a kept-alive connection per thread and host: thousands of
    small header reads, so a fresh TLS handshake each time would dominate."""
    parts = urllib.parse.urlsplit(url)
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    for attempt in range(3):
        conn = conns.get(parts.netloc)
        if conn is None:
            conn = conns[parts.netloc] = http.client.HTTPSConnection(parts.netloc, timeout=60)
        try:
            conn.request("GET", parts.path, headers={"Range": f"bytes={start}-{end}"})
            r = conn.getresponse()
            body = r.read()
        except (http.client.HTTPException, OSError):
            conn.close()
            conns.pop(parts.netloc, None)
            if attempt == 2:
                raise
            continue
        if r.status == 416:
            raise urllib.error.HTTPError(url, 416, "range not satisfiable", r.headers, None)
        if r.status not in (200, 206):
            raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)
        return body


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
