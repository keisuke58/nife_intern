#!/usr/bin/env python3
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


class ClassifierDownloadError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_qza(path: Path) -> None:
    if not path.exists():
        raise ClassifierDownloadError(f"classifier not found: {path}")
    if path.stat().st_size < 1024:
        raise ClassifierDownloadError(f"classifier too small to be valid: {path}")
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "metadata.yaml" not in names:
                raise ClassifierDownloadError("qza is missing metadata.yaml (not a valid QIIME2 artifact?)")
    except zipfile.BadZipFile as e:
        raise ClassifierDownloadError("classifier is not a valid zip/qza") from e


def download(url: str, out: Path, expected_sha256: str | None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=out.name + ".", dir=str(out.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        h = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": "nife-qza-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                h.update(chunk)
        tmp.flush()
        os.fsync(tmp.fileno())

    sha = h.hexdigest()
    if expected_sha256 and sha.lower() != expected_sha256.lower():
        tmp_path.unlink(missing_ok=True)
        raise ClassifierDownloadError(f"sha256 mismatch: expected={expected_sha256} got={sha}")

    tmp_path.replace(out)
    _validate_qza(out)
    return {"url": url, "out": str(out.resolve()), "sha256": sha, "size_bytes": out.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "classifiers" / "classifier.qza")
    ap.add_argument("--sha256", type=str, default=None)
    args = ap.parse_args()

    info = download(args.url, args.out.resolve(), args.sha256)
    report_path = args.out.resolve().with_suffix(".download.json")
    report_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

