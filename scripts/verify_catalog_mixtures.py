#!/usr/bin/env python3
"""Verify mixtures/ matches mixture_mfa_catalog CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "pipeline"))
from path_utils import resolve_bundle_path  # noqa: E402


def _source_tail(path_str: str) -> str:
    s = (path_str or "").strip().replace("\\", "/")
    for marker in ("00_source_data/", "data/"):
        if marker in s:
            return s[s.index(marker) + len(marker) :]
    return s


def resolve_catalog_audio(path_str: str, root: Path) -> Path:
    s = (path_str or "").strip().replace("\\", "/")
    for marker in ("00_source_data/", "data/"):
        if marker in s:
            s = s[s.index(marker) :]
            break
    return resolve_bundle_path(s, root)


def load_catalog(catalog: Path) -> List[Dict[str, str]]:
    with catalog.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_row(row: Dict[str, str], *, root: Path, mix_root: Path) -> List[str]:
    errs: List[str] = []
    subset = row["subset"].strip()
    overlap_dir = row["overlap_dir"].strip()
    item_id = row["item_id"].strip()
    stem_dir = mix_root / subset / overlap_dir
    jpath = stem_dir / f"{item_id}.json"

    for suffix in (".mix.wav", ".target.wav", ".nontarget.wav", ".ref.wav"):
        if not (stem_dir / f"{item_id}{suffix}").is_file():
            errs.append(f"missing {item_id}{suffix}")
    if not jpath.is_file():
        return errs + ["missing json"]

    meta = json.loads(jpath.read_text(encoding="utf-8"))
    for json_key, csv_key in (
        ("utt_s1_mix_path", "s1_source_wav_path"),
        ("utt_s2_mix_path", "s2_source_wav_path"),
        ("utt_s1_ref_path", "ref_source_wav_path"),
    ):
        if _source_tail(meta.get(json_key, "")) != _source_tail(row.get(csv_key, "")):
            errs.append(f"{item_id}: {json_key} mismatch catalog")
        elif not resolve_catalog_audio(row.get(csv_key, ""), root).is_file():
            errs.append(f"{item_id}: source missing for {csv_key}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--mix-root", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--list-errors", type=int, default=20)
    args = ap.parse_args()

    root = (args.base_dir or _SCRIPT_DIR.parent).resolve()
    mix_root = args.mix_root.resolve()
    rows = load_catalog(args.catalog.resolve())
    errors: List[str] = []
    expected: Set[Tuple[str, str, str]] = set()

    for row in rows:
        expected.add((row["subset"].strip(), row["overlap_dir"].strip(), row["item_id"].strip()))
        errors.extend(verify_row(row, root=root, mix_root=mix_root))

    for jpath in mix_root.rglob("*.json"):
        parts = jpath.relative_to(mix_root).parts
        if len(parts) >= 3 and (parts[0], parts[1], jpath.stem) not in expected:
            errors.append(f"extra json: {jpath.relative_to(mix_root)}")

    n_mix = len(list(mix_root.rglob("*.mix.wav")))
    print(f"Catalog rows: {len(rows)}  mix.wav: {n_mix}  errors: {len(errors)}")
    if errors:
        for e in errors[: args.list_errors]:
            print(f"  {e}")
        return 1
    print("OK — mixtures match catalog.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
