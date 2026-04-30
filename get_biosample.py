#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BioSampleFetchError(RuntimeError):
    pass


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub("", s)
    return s


def _to_https(url: str) -> str:
    url = url.strip()
    if url.startswith("ftp.sra.ebi.ac.uk/"):
        return "https://" + url
    if url.startswith("ftp."):
        return "https://" + url
    if url.startswith("ftp://"):
        return "https://" + url[len("ftp://") :]
    return url


def _http_get(url: str, timeout_s: float = 60.0, max_tries: int = 5, backoff_s: float = 1.0) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return r.read()
        except Exception as e:
            last_err = e
            if attempt == max_tries:
                break
            time.sleep(backoff_s * (2 ** (attempt - 1)))
    raise BioSampleFetchError(f"GET failed after {max_tries} tries: {url} ({type(last_err).__name__}: {last_err})")


def fetch_run_to_biosample_map(study_accession: str) -> dict[str, str]:
    fields = "run_accession,sample_accession"
    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport?"
        + urllib.parse.urlencode(
            {
                "accession": study_accession,
                "result": "read_run",
                "format": "tsv",
                "download": "true",
                "fields": fields,
            }
        )
    )
    raw = _http_get(url, timeout_s=60.0)
    text = raw.decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise BioSampleFetchError(f"ENA filereport returned no rows for: {study_accession}")
    hdr = lines[0].split("\t")
    run_i = hdr.index("run_accession")
    sample_i = hdr.index("sample_accession")
    out: dict[str, str] = {}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(run_i, sample_i):
            continue
        run = parts[run_i].strip()
        bs = parts[sample_i].strip()
        if run and bs:
            out[run] = bs
    return out


def fetch_biosample_xml(biosample_accession: str) -> bytes:
    url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{biosample_accession}"
    return _http_get(url, timeout_s=60.0)


@dataclass(frozen=True)
class ParsedBioSample:
    biosample_accession: str
    sample_alias: str
    in_vivo_in_vitro: str
    material: str
    day: str
    donor: str
    title: str
    collection_date: str
    env_local_scale: str
    env_medium: str
    geo_loc_name: str


def _extract_title_fields(title: str) -> tuple[str, str, str, str]:
    t = title.strip()
    tl = t.lower()

    if "in vivo" in tl:
        viv = "in_vivo"
    elif "in vitro" in tl:
        viv = "in_vitro"
    elif "saliva" in tl:
        viv = "saliva"
    else:
        viv = "not_provided"

    day = "not_provided"
    m = re.match(r"^\s*(\d+)\s*d\b", tl)
    if m:
        day = m.group(1)

    material = "not_provided"
    if re.search(r"\bti6al4v\b", tl):
        material = "Ti6Al4V"
    elif re.search(r"\by[- ]?tzp\b", tl):
        material = "Y-TZP"
    elif re.search(r"\btitanium\b|\bti\b", tl):
        material = "titanium"

    donor = "not_provided"
    if "saliva" in tl:
        m2 = re.search(r"\bsaliva\s+(\d+)\b", tl)
        if m2:
            donor = m2.group(1)

    return viv, material, day, donor


def parse_biosample_xml(xml_bytes: bytes) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    s = root.find(".//SAMPLE")
    if s is None:
        raise ValueError("No SAMPLE element found")
    accession = (s.attrib.get("accession") or "").strip()
    alias = (s.attrib.get("alias") or "").strip()
    title = (s.findtext("TITLE") or "").strip()

    attrs: dict[str, str] = {}
    for a in s.findall(".//SAMPLE_ATTRIBUTE"):
        tag = (a.findtext("TAG") or "").strip()
        val = (a.findtext("VALUE") or "").strip()
        if tag:
            attrs[tag] = val

    return {
        "biosample_accession": accession,
        "sample_alias": alias,
        "title": title,
        "attributes": attrs,
    }


def extract_required(parsed: dict[str, Any]) -> ParsedBioSample:
    accession = _sanitize_text(str(parsed.get("biosample_accession") or "").strip())
    alias = _sanitize_text(str(parsed.get("sample_alias") or "").strip())
    title = _sanitize_text(str(parsed.get("title") or "").strip())
    attrs = parsed.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    viv, material, day, donor = _extract_title_fields(title)

    def get_attr(key: str) -> str:
        v = attrs.get(key, "")
        v = _sanitize_text(str(v).strip())
        if not v or v.lower() in {"not provided", "not collected", "missing"}:
            return "not_provided"
        return v

    return ParsedBioSample(
        biosample_accession=accession or "not_provided",
        sample_alias=alias or "not_provided",
        in_vivo_in_vitro=viv,
        material=material,
        day=day,
        donor=donor,
        title=title or "not_provided",
        collection_date=get_attr("collection_date"),
        env_local_scale=get_attr("env_local_scale"),
        env_medium=get_attr("env_medium"),
        geo_loc_name=get_attr("geo_loc_name"),
    )


def load_filereport_runs(tsv_path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run = (row.get("run_accession") or "").strip()
            alias = (row.get("sample_alias") or "").strip()
            if run and alias:
                out.append((run, alias))
    if not out:
        raise ValueError(f"No runs found in: {tsv_path}")
    return out


def write_tsv(path: Path, rows: list[ParsedBioSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "biosample_accession",
                "sample_alias",
                "in_vivo_in_vitro",
                "material",
                "day",
                "donor",
                "title",
                "collection_date",
                "env_local_scale",
                "env_medium",
                "geo_loc_name",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.biosample_accession,
                    r.sample_alias,
                    r.in_vivo_in_vitro,
                    r.material,
                    r.day,
                    r.donor,
                    r.title,
                    r.collection_date,
                    r.env_local_scale,
                    r.env_medium,
                    r.geo_loc_name,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--filereport",
        type=Path,
        default=Path(__file__).parent / "data" / "PRJNA1159109_filereport.tsv",
    )
    ap.add_argument("--study", type=str, default="PRJNA1159109")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "data" / "biosample_attributes.tsv",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).parent / "data" / "run.log",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    runs = load_filereport_runs(args.filereport)
    if args.limit and args.limit > 0:
        runs = runs[: args.limit]

    run_to_bs = fetch_run_to_biosample_map(args.study)
    missing_run = []
    alias_mismatch = []
    biosamples_by_acc: dict[str, str] = {}
    for run, alias in runs:
        bs = run_to_bs.get(run)
        if not bs:
            missing_run.append(run)
            continue
        biosamples_by_acc[bs] = alias

    log_obj: dict[str, Any] = {
        "study": args.study,
        "filereport": str(args.filereport),
        "n_runs_in_filereport": len(runs),
        "n_run_to_biosample_map": len(run_to_bs),
        "missing_run_accessions": missing_run,
        "warnings": [],
    }

    rows: list[ParsedBioSample] = []
    missing_biosample = []
    for i, (bs, alias_expected) in enumerate(sorted(biosamples_by_acc.items()), start=1):
        try:
            xml_bytes = fetch_biosample_xml(bs)
            parsed = parse_biosample_xml(xml_bytes)
            row = extract_required(parsed)
            if row.sample_alias != alias_expected:
                alias_mismatch.append({"biosample": bs, "expected": alias_expected, "got": row.sample_alias})
            rows.append(row)
        except Exception as e:
            missing_biosample.append({"biosample": bs, "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0:
            time.sleep(0.2)

    log_obj["alias_mismatch"] = alias_mismatch
    log_obj["missing_biosample"] = missing_biosample
    log_obj["n_biosamples_requested"] = len(biosamples_by_acc)
    log_obj["n_biosamples_parsed"] = len(rows)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_tsv(args.out, rows)

    if missing_run or missing_biosample:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
