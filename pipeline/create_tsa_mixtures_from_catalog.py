#!/usr/bin/env python3
"""
Create TSA mixtures from mixture_mfa_catalog*.csv (fixed utterances + MFA transcripts).

Uses s1/s2/ref source paths and overlap from the catalog. Segment start times come from:
  - CSV columns start_sec_s1, start_sec_s2, start_sec_ref (if present), or
  - matching JSON under --timing-json-root (e.g. tsa_mix TSA_MIXED_V2 or TSE-beta 01_mixtures_v2).

Output layout matches create_tsa_mixtures.py (01_mixtures_v2/{AA,CA,CC}/ov*/).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore

from config import L_SEC, MIX_PEAK, SNR_DB, SR
from create_tsa_mixtures import (
    extract_segment,
    normalize_rms,
    peak_scale,
    place_segment,
    to_stored_path,
)
from path_utils import resolve_bundle_path


def resolve_catalog_audio(path_str: str, alpha: Path) -> Path:
    """Map catalog paths (legacy /app/.../00_source_data/ or data/) to bundle files."""
    s = (path_str or "").strip().replace("\\", "/")
    for marker in ("00_source_data/", "data/"):
        if marker in s:
            s = s[s.index(marker) :]
            break
    return resolve_bundle_path(s, alpha)


def load_timing_json(
    timing_root: Optional[Path],
    subset: str,
    overlap_dir: str,
    item_id: str,
) -> Dict[str, Any]:
    if not timing_root:
        return {}
    jpath = timing_root / subset / overlap_dir / f"{item_id}.json"
    if not jpath.is_file():
        return {}
    return json.loads(jpath.read_text(encoding="utf-8"))


def float_or_none(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def create_mixture_from_row(
    row: Dict[str, str],
    *,
    alpha: Path,
    out_dir: Path,
    timing_root: Optional[Path],
    L_sec: float = L_SEC,
    sr: int = SR,
    snr_db: float = SNR_DB,
    mix_peak: float = MIX_PEAK,
) -> Path:
    subset = row["subset"].strip()
    overlap_dir = row["overlap_dir"].strip()
    item_id = row["item_id"].strip()
    if row.get("overlap_ratio") not in (None, ""):
        overlap = float(row["overlap_ratio"])
    else:
        overlap = float(row.get("overlap_pct") or 0) / 100.0

    utt_s1 = resolve_catalog_audio(row["s1_source_wav_path"], alpha)
    utt_s2 = resolve_catalog_audio(row["s2_source_wav_path"], alpha)
    utt_ref = resolve_catalog_audio(row["ref_source_wav_path"], alpha)
    for label, p in (("s1", utt_s1), ("s2", utt_s2), ("ref", utt_ref)):
        if not p.is_file():
            raise FileNotFoundError(f"{label} source missing for {item_id}: {p}")

    meta = load_timing_json(timing_root, subset, overlap_dir, item_id)
    start_s1 = float_or_none(row.get("start_sec_s1"))
    start_s2 = float_or_none(row.get("start_sec_s2"))
    start_ref = float_or_none(row.get("start_sec_ref"))
    if start_s1 is None:
        start_s1 = float(meta.get("start_sec_s1", 0.0))
    if start_s2 is None:
        start_s2 = float(meta.get("start_sec_s2", 0.0))
    if start_ref is None:
        start_ref = float(meta.get("start_sec_ref", 0.0))

    shift_sec = (1.0 - overlap) * L_sec
    total_sec = L_sec + shift_sec
    n_total = int(round(total_sec * sr))
    start_s2_sample = int(round(shift_sec * sr))

    seg_s1 = normalize_rms(extract_segment(utt_s1, start_s1, L_sec, sr))
    seg_s2 = normalize_rms(extract_segment(utt_s2, start_s2, L_sec, sr))
    seg_ref = extract_segment(utt_ref, start_ref, L_sec, sr)

    target = place_segment(seg_s1, 0, n_total)
    nontarget = place_segment(seg_s2, start_s2_sample, n_total)
    mix_raw = target + nontarget
    mix, target, nontarget = peak_scale(mix_raw, target, nontarget, peak=mix_peak)
    ref, = peak_scale(seg_ref, peak=mix_peak)

    sub_out = out_dir / subset / overlap_dir
    sub_out.mkdir(parents=True, exist_ok=True)
    stem = item_id

    out_meta = {
        "overlap": overlap,
        "L_sec": L_sec,
        "shift_sec": shift_sec,
        "sr": sr,
        "utt_s1_mix_path": to_stored_path(utt_s1, alpha),
        "utt_s2_mix_path": to_stored_path(utt_s2, alpha),
        "utt_s1_ref_path": to_stored_path(utt_ref, alpha),
        "snr_db": snr_db,
        "start_sec_s1": round(start_s1, 4),
        "start_sec_s2": round(start_s2, 4),
        "start_sec_ref": round(start_ref, 4),
        "subset": subset,
        "s1_id": row.get("s1_speaker_id", ""),
        "s2_id": row.get("s2_speaker_id", ""),
        "mfa_transcript_s1": (row.get("mfa_transcript_s1") or "").strip(),
        "mfa_transcript_s2": (row.get("mfa_transcript_s2") or "").strip(),
        "catalog_item_id": item_id,
    }

    if sf is None:
        raise RuntimeError("soundfile is required")

    sf.write(str(sub_out / f"{stem}.mix.wav"), mix, sr)
    sf.write(str(sub_out / f"{stem}.target.wav"), target, sr)
    sf.write(str(sub_out / f"{stem}.nontarget.wav"), nontarget, sr)
    sf.write(str(sub_out / f"{stem}.ref.wav"), ref, sr)
    json_path = sub_out / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)
        f.write("\n")
    return json_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Create TSA mixtures from mixture_mfa_catalog CSV (fixed speakers/utterances)."
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="CSV e.g. 00_source_data/manifests/mixture_mfa_catalog_docker.csv",
    )
    ap.add_argument("--base-dir", type=Path, default=None, help="TSE-alpha root")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output e.g. 01_mixtures_v2")
    ap.add_argument(
        "--timing-json-root",
        type=Path,
        default=None,
        help="Optional tree {subset}/{ov*}/{item_id}.json for start_sec_* (reference mixes)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max rows (debug)")
    ap.add_argument("--dry-run", action="store_true", help="Validate paths only")
    args = ap.parse_args(argv)

    alpha = (args.base_dir or Path(__file__).resolve().parent.parent).resolve()
    out_dir = args.out_dir.resolve()
    catalog = args.catalog.resolve()
    timing_root = args.timing_json_root.resolve() if args.timing_json_root else None

    if not catalog.is_file():
        print(f"Catalog not found: {catalog}", file=sys.stderr)
        return 1

    rows: List[Dict[str, str]] = []
    with catalog.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if args.limit:
        rows = rows[: args.limit]

    print(f"base_dir={alpha}", file=sys.stderr)
    print(f"catalog={catalog} rows={len(rows)}", file=sys.stderr)
    print(f"out_dir={out_dir}", file=sys.stderr)
    if timing_root:
        print(f"timing_json_root={timing_root}", file=sys.stderr)

    missing_timing = 0
    ok = 0
    for i, row in enumerate(rows):
        item_id = row.get("item_id", f"row{i}")
        has_csv_timing = all(
            float_or_none(row.get(k)) is not None
            for k in ("start_sec_s1", "start_sec_s2", "start_sec_ref")
        )
        if not has_csv_timing:
            meta = load_timing_json(
                timing_root,
                row.get("subset", ""),
                row.get("overlap_dir", ""),
                item_id,
            )
            if not meta and timing_root:
                missing_timing += 1
            elif not meta and not timing_root:
                missing_timing += 1

        if args.dry_run:
            resolve_catalog_audio(row["s1_source_wav_path"], alpha)
            ok += 1
            continue

        try:
            create_mixture_from_row(
                row,
                alpha=alpha,
                out_dir=out_dir,
                timing_root=timing_root,
            )
            ok += 1
            if (ok % 100) == 0:
                print(f"  ... {ok}/{len(rows)}", file=sys.stderr)
        except Exception as e:
            print(f"FAIL {item_id}: {e}", file=sys.stderr)
            return 1

    if missing_timing and not args.dry_run:
        print(
            f"Warning: {missing_timing} rows used start_sec=0 (no CSV timing and no JSON). "
            "Run scripts/augment_mixture_catalog_timing.py or pass --timing-json-root.",
            file=sys.stderr,
        )

    print(f"Done. Wrote {ok} mixtures under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
