#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal HTTP-range-backed file object so that ``zipfile.ZipFile`` can list and
extract individual members of a large remote ZIP archive (e.g. the 3.7 GB
Abdelaal et al. 2019 benchmark archive on Zenodo) without downloading the
whole archive. Only the Python standard library is used.
"""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
import zipfile
from typing import Optional


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only file object that fetches bytes with HTTP Range requests."""

    def __init__(self, url: str, block_size: int = 1 << 20, timeout: int = 120, retries: int = 10):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self._cache_start = -1
        self._cache = b""
        self.size = self._fetch_size()

    # --- HTTP helpers -------------------------------------------------------
    def _request(self, start: int, end: int) -> bytes:
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                if resp.status not in (200, 206):
                    raise IOError(f"HTTP {resp.status}")
                if resp.status == 200:  # server ignored Range
                    data = data[start : end + 1]
                return data
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise IOError(f"Range request {start}-{end} failed: {last_err}")

    def _fetch_size(self) -> int:
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr:
                return int(cr.split("/")[-1])
            cl = resp.headers.get("Content-Length")
            if cl is None:
                raise IOError("Server reports neither Content-Range nor Content-Length")
            return int(cl)

    # --- io.RawIOBase API ---------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            if self._cache_start <= self.pos < self._cache_start + len(self._cache):
                off = self.pos - self._cache_start
                chunk = self._cache[off : off + n]
            else:
                want = max(n, self.block_size)
                end = min(self.size - 1, self.pos + want - 1)
                self._cache = self._request(self.pos, end)
                self._cache_start = self.pos
                chunk = self._cache[:n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str, block_size: int = 4 << 20) -> zipfile.ZipFile:
    return zipfile.ZipFile(HTTPRangeFile(url, block_size=block_size))


def extract_member(zf: zipfile.ZipFile, member: str, out_path: str, chunk: int = 4 << 20) -> str:
    """Stream-extract a single ZIP member to ``out_path`` (atomic rename on success)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    info = zf.getinfo(member)
    if os.path.exists(out_path) and os.path.getsize(out_path) == info.file_size:
        return out_path
    tmp = out_path + ".part"
    done = 0
    t0 = time.time()
    with zf.open(member) as src, open(tmp, "wb") as dst:
        while True:
            buf = src.read(chunk)
            if not buf:
                break
            dst.write(buf)
            done += len(buf)
            rate = done / max(1e-6, time.time() - t0)
            print(
                f"\r  {member}: {done / 1e6:8.1f} / {info.file_size / 1e6:8.1f} MB (uncompressed) "
                f"{rate / 1e3:7.1f} kB/s",
                end="",
                file=sys.stderr,
                flush=True,
            )
    print("", file=sys.stderr)
    os.replace(tmp, out_path)
    return out_path


def parallel_extract_member(
    url: str,
    zinfo: zipfile.ZipInfo,
    out_path: str,
    n_threads: int = 16,
    chunk_bytes: int = 2 << 20,
) -> str:
    """
    Download the compressed bytes of one ZIP member with many parallel HTTP Range
    requests (useful when per-connection bandwidth is throttled), then inflate
    them locally. Verifies the CRC-32 recorded in the archive.
    """
    import struct
    import zlib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) == zinfo.file_size:
        return out_path

    reader = HTTPRangeFile(url, block_size=64 * 1024)
    lh = reader._request(zinfo.header_offset, zinfo.header_offset + 29)
    sig, = struct.unpack("<I", lh[:4])
    if sig != 0x04034B50:
        raise IOError("Bad local file header signature")
    n_len, x_len = struct.unpack("<HH", lh[26:30])
    data_start = zinfo.header_offset + 30 + n_len + x_len
    total = zinfo.compress_size

    comp_path = out_path + ".compressed.part"
    ranges = [(s, min(total, s + chunk_bytes) - 1) for s in range(0, total, chunk_bytes)]
    done_flags = comp_path + ".done"
    finished = set()
    if os.path.exists(done_flags):
        with open(done_flags) as fh:
            finished = {int(x) for x in fh.read().split()}
    mode = "r+b" if os.path.exists(comp_path) else "wb"
    with open(comp_path, mode) as fh:
        fh.truncate(total)
    t0 = time.time()
    got = sum(ranges[i][1] - ranges[i][0] + 1 for i in finished if i < len(ranges))

    def _fetch(i: int):
        s, e = ranges[i]
        return i, reader._request(data_start + s, data_start + e)

    todo = [i for i in range(len(ranges)) if i not in finished]
    with ThreadPoolExecutor(max_workers=n_threads) as ex, open(comp_path, "r+b") as fh, open(done_flags, "a") as fd:
        futs = [ex.submit(_fetch, i) for i in todo]
        for fut in as_completed(futs):
            i, data = fut.result()
            fh.seek(ranges[i][0])
            fh.write(data)
            fd.write(f"{i}\n")
            fd.flush()
            got += len(data)
            rate = got / max(1e-6, time.time() - t0)
            print(
                f"\r  {os.path.basename(out_path)}: {got / 1e6:7.1f}/{total / 1e6:7.1f} MB compressed, {rate / 1e3:6.1f} kB/s",
                end="",
                file=sys.stderr,
                flush=True,
            )
    print("", file=sys.stderr)

    tmp = out_path + ".part"
    crc = 0
    with open(comp_path, "rb") as src, open(tmp, "wb") as dst:
        if zinfo.compress_type == zipfile.ZIP_STORED:
            while True:
                buf = src.read(8 << 20)
                if not buf:
                    break
                crc = zlib.crc32(buf, crc)
                dst.write(buf)
        elif zinfo.compress_type == zipfile.ZIP_DEFLATED:
            d = zlib.decompressobj(-15)
            while True:
                buf = src.read(8 << 20)
                if not buf:
                    break
                out = d.decompress(buf)
                crc = zlib.crc32(out, crc)
                dst.write(out)
            out = d.flush()
            crc = zlib.crc32(out, crc)
            dst.write(out)
        else:
            raise IOError(f"Unsupported compression type {zinfo.compress_type}")
    if (crc & 0xFFFFFFFF) != zinfo.CRC:
        raise IOError(f"CRC mismatch for {zinfo.filename}")
    os.replace(tmp, out_path)
    os.remove(comp_path)
    os.remove(done_flags)
    return out_path


def parallel_download(url: str, out_path: str, n_threads: int = 8, chunk_bytes: int = 8 << 20) -> str:
    """Download a whole remote file with parallel HTTP Range requests (resumable)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    reader = HTTPRangeFile(url, block_size=64 * 1024)
    total = reader.size
    if os.path.exists(out_path) and os.path.getsize(out_path) == total:
        return out_path
    part, flags = out_path + ".part", out_path + ".part.done"
    ranges = [(s, min(total, s + chunk_bytes) - 1) for s in range(0, total, chunk_bytes)]
    finished = set()
    if os.path.exists(flags):
        with open(flags) as fh:
            finished = {int(x) for x in fh.read().split()}
    with open(part, "r+b" if os.path.exists(part) else "wb") as fh:
        fh.truncate(total)
    todo = [i for i in range(len(ranges)) if i not in finished]
    got = total - sum(ranges[i][1] - ranges[i][0] + 1 for i in todo)
    t0 = time.time()

    def _fetch(i):
        s, e = ranges[i]
        data = reader._request(s, e)
        if len(data) != e - s + 1:
            raise IOError(f"short read for range {s}-{e}")
        return i, data

    with ThreadPoolExecutor(max_workers=n_threads) as ex, open(part, "r+b") as fh, open(flags, "a") as fd:
        for fut in as_completed([ex.submit(_fetch, i) for i in todo]):
            i, data = fut.result()
            fh.seek(ranges[i][0])
            fh.write(data)
            fd.write(f"{i}\n")
            fd.flush()
            got += len(data)
            print(f"\r  {os.path.basename(out_path)}: {got / 1e6:8.1f}/{total / 1e6:8.1f} MB "
                  f"{(got / max(1e-6, time.time() - t0)) / 1e6:5.2f} MB/s", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    os.replace(part, out_path)
    os.remove(flags)
    return out_path


if __name__ == "__main__":
    url = sys.argv[1]
    zf = open_remote_zip(url)
    for zi in zf.infolist():
        print(f"{zi.compress_size / 1e6:10.1f} MB  {zi.file_size / 1e6:10.1f} MB  {zi.filename}")
