#!/usr/bin/env python3
"""
Create TSA two-speaker mixtures (AA, CC, CA) with controlled overlap.
See README.md for spec (L=5s, overlaps 0–100%, seed 42 default).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore

from config import (
    DEFAULT_SEED,
    L_SEC,
    MIN_UTT_LENGTH_SEC,
    MIX_PEAK,
    N_MIXTURES_PER_OVERLAP,
    OVERLAPS,
    SNR_DB,
    SR,
    SUBSETS,
    get_base_dir,
    get_libri_root,
    get_myst_csv,
    get_myst_root,
    get_output_root,
)
from path_utils import resolve_bundle_path, to_bundle_path

# Random-mix path only (full pipeline uses create_tsa_mixtures_from_catalog.py).
# Stored paths use path_utils.to_bundle_path() → data/... under bundle ROOT, not /app.


def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile is required")
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _duration_sec(path: Path) -> float:
    if sf is None:
        y, sr = _load_wav(path)
        return len(y) / sr
    info = sf.info(str(path))
    return float(info.duration)


def to_stored_path(wav_path: Path, base_dir: Path) -> str:
    """Bundle-relative paths: 00_source_data/libri|myst/..."""
    return to_bundle_path(wav_path, base_dir)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    return resolve_bundle_path(path_str, base_dir)


def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.is_file() or p.is_dir():
            return p
    return None


def load_libri_pool(base_dir: Path) -> Dict[str, List[Path]]:
    manifest = _first_existing(
        base_dir / "data" / "manifests" / "librispeech_test_clean_manifest.jsonl",
        base_dir / "LibriSpeech" / "test-clean" / "librispeech_test_clean_manifest.jsonl",
    )
    root = _first_existing(
        get_libri_root(base_dir),
        base_dir / "data" / "libri" / "test-clean-normalised",
    )
    if manifest is None or root is None:
        return {}
    spk2paths: Dict[str, List[Path]] = defaultdict(list)
    if not manifest.is_file():
        return {}
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = (row.get("utterance_id") or "").strip()
            if not uid or "-" not in uid:
                continue
            spk = uid.split("-")[0]
            rel = row.get("normalised_audio_path") or row.get("normalized_audio_path")
            if not rel:
                continue
            path = resolve_path(rel, base_dir)
            if not path.is_file() and "test-clean-normalised" in str(rel):
                parts = Path(str(rel).replace("\\", "/")).parts
                idx = parts.index("test-clean-normalised")
                path = (root / Path(*parts[idx + 1 :])).resolve()
            if path.is_file() and _duration_sec(path) >= MIN_UTT_LENGTH_SEC:
                spk2paths[f"libri_{spk}"].append(path.resolve())
    return {k: v for k, v in spk2paths.items() if len(v) >= 2}


def load_myst_pool(base_dir: Path) -> Dict[str, List[Path]]:
    csv_path = _first_existing(
        get_myst_csv(base_dir),
        base_dir / "data" / "myst" / "myST_test_filtered" / "filtered_list_has_trn.csv",
    )
    root = _first_existing(
        get_myst_root(base_dir),
        base_dir / "data" / "myst" / "myST_test_filtered",
    )
    if csv_path is None or root is None:
        return {}
    spk2paths: Dict[str, List[Path]] = defaultdict(list)
    if not csv_path.is_file():
        return {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = (row.get("utt_id") or row.get("utterance_id") or "").strip()
            if not uid.startswith("myst_"):
                continue
            parts = uid.split("_")
            if len(parts) < 2:
                continue
            spk = f"myst_{parts[1]}"
            ap = (row.get("audio_path") or "").strip()
            if not ap:
                continue
            path = resolve_path(ap, base_dir)
            if not path.is_file():
                # relative to myst root
                tail = ap.split("myST_test_filtered/")[-1] if "myST_test_filtered/" in ap else Path(ap).name
                path = (root / tail).resolve()
            if path.is_file() and _duration_sec(path) >= MIN_UTT_LENGTH_SEC:
                spk2paths[spk].append(path.resolve())
    return {k: v for k, v in spk2paths.items() if len(v) >= 2}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def extract_segment(
    wav_path: Path, start_sec: float, length_sec: float, sr: int
) -> np.ndarray:
    y, file_sr = _load_wav(wav_path)
    if file_sr != sr:
        raise ValueError(f"sample rate {file_sr} != {sr} in {wav_path}")
    n = int(round(length_sec * sr))
    start = int(round(start_sec * sr))
    end = start + n
    if end > len(y):
        raise ValueError(f"segment past end: {wav_path} start={start_sec}")
    seg = y[start:end].copy()
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg


def place_segment(
    seg: np.ndarray, start_sample: int, total_samples: int
) -> np.ndarray:
    buf = np.zeros(total_samples, dtype=np.float32)
    n = len(seg)
    end = start_sample + n
    if end > total_samples:
        seg = seg[: total_samples - start_sample]
        end = total_samples
    buf[start_sample:end] = seg
    return buf


def normalize_rms(seg: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    r = rms(seg)
    if r < 1e-9:
        return seg
    return (seg * (target_rms / r)).astype(np.float32)


def peak_scale(*arrays: np.ndarray, peak: float = MIX_PEAK) -> Tuple[np.ndarray, ...]:
    m = max(float(np.max(np.abs(a))) for a in arrays if a.size)
    if m < 1e-9:
        return arrays
    scale = peak / m
    return tuple((a * scale).astype(np.float32) for a in arrays)


def overlap_dir_name(r: float) -> str:
    return f"ov{int(round(r * 100))}"


def pick_two_speakers(
    rng: random.Random, pool: Dict[str, List[Path]]
) -> Tuple[str, str, List[Path], List[Path]]:
    keys = list(pool.keys())
    s1_id = rng.choice(keys)
    s2_id = rng.choice([k for k in keys if k != s1_id])
    return s1_id, s2_id, pool[s1_id], pool[s2_id]


def pick_utterances(
    rng: random.Random, utts: List[Path], need: int
) -> List[Path]:
    if len(utts) < need:
        raise RuntimeError("not enough utterances")
    return rng.sample(utts, need)


def random_start(rng: random.Random, path: Path, length_sec: float) -> float:
    dur = _duration_sec(path)
    if dur <= length_sec:
        return 0.0
    return rng.uniform(0.0, dur - length_sec)


def create_one_mixture(
    rng: random.Random,
    *,
    subset: str,
    overlap: float,
    mix_idx: int,
    libri_pool: Dict[str, List[Path]],
    myst_pool: Dict[str, List[Path]],
    base_dir: Path,
    out_dir: Path,
    L_sec: float = L_SEC,
    sr: int = SR,
    snr_db: float = SNR_DB,
    mix_peak: float = MIX_PEAK,
) -> Path:
    shift_sec = (1.0 - overlap) * L_sec
    total_sec = L_sec + shift_sec
    n_total = int(round(total_sec * sr))
    start_s2 = int(round(shift_sec * sr))

    if subset == "AA":
        s1_id, s2_id, u1, u2 = pick_two_speakers(rng, libri_pool)
    elif subset == "CC":
        s1_id, s2_id, u1, u2 = pick_two_speakers(rng, myst_pool)
    elif subset == "CA":
        s1_id, _, u1, _ = pick_two_speakers(rng, myst_pool)
        s2_id, _, u2, _ = pick_two_speakers(rng, libri_pool)
        if s1_id == s2_id:
            # myst vs libri ids never collide; keep for type check
            pass
    else:
        raise ValueError(subset)

    utt_s1_mix, utt_s1_ref = pick_utterances(rng, u1, 2)
    utt_s2_mix = pick_utterances(rng, u2, 1)[0]

    start_s1 = random_start(rng, utt_s1_mix, L_sec)
    start_s2_sec = random_start(rng, utt_s2_mix, L_sec)
    start_ref = random_start(rng, utt_s1_ref, L_sec)

    seg_s1 = normalize_rms(extract_segment(utt_s1_mix, start_s1, L_sec, sr))
    seg_s2 = normalize_rms(extract_segment(utt_s2_mix, start_s2_sec, L_sec, sr))
    seg_ref = extract_segment(utt_s1_ref, start_ref, L_sec, sr)

    target = place_segment(seg_s1, 0, n_total)
    nontarget = place_segment(seg_s2, start_s2, n_total)
    mix_raw = target + nontarget
    mix, target, nontarget = peak_scale(mix_raw, target, nontarget, peak=mix_peak)
    ref, = peak_scale(seg_ref, peak=mix_peak)

    ov_name = overlap_dir_name(overlap)
    stem = f"{mix_idx:06d}_s1-{s1_id}_s2-{s2_id}_{ov_name}"
    sub_out = out_dir / subset / ov_name
    sub_out.mkdir(parents=True, exist_ok=True)

    meta = {
        "overlap": overlap,
        "L_sec": L_sec,
        "shift_sec": shift_sec,
        "sr": sr,
        "utt_s1_mix_path": to_stored_path(utt_s1_mix, base_dir),
        "utt_s2_mix_path": to_stored_path(utt_s2_mix, base_dir),
        "utt_s1_ref_path": to_stored_path(utt_s1_ref, base_dir),
        "snr_db": snr_db,
        "start_sec_s1": round(start_s1, 4),
        "start_sec_s2": round(start_s2_sec, 4),
        "start_sec_ref": round(start_ref, 4),
        "subset": subset,
        "s1_id": s1_id,
        "s2_id": s2_id,
    }

    if sf is None:
        raise RuntimeError("soundfile is required to write WAV")

    sf.write(str(sub_out / f"{stem}.mix.wav"), mix, sr)
    sf.write(str(sub_out / f"{stem}.target.wav"), target, sr)
    sf.write(str(sub_out / f"{stem}.nontarget.wav"), nontarget, sr)
    sf.write(str(sub_out / f"{stem}.ref.wav"), ref, sr)
    json_path = sub_out / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return json_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create TSA mixture dataset (AA, CC, CA).")
    ap.add_argument("--base-dir", type=Path, default=None, help="Repo root with LibriSpeech/ and MyST/")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output root (default: tse/tsa_mix/data/TSA_MIXED)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-per-overlap", type=int, default=N_MIXTURES_PER_OVERLAP)
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=list(SUBSETS),
        choices=list(SUBSETS),
    )
    ap.add_argument(
        "--overlaps",
        nargs="+",
        type=float,
        default=None,
        help="Overlap ratios (default: 0 0.2 ... 1.0)",
    )
    args = ap.parse_args(argv)

    base_s = get_base_dir()
    base_dir = args.base_dir or (Path(base_s) if base_s else Path.cwd())
    base_dir = base_dir.resolve()

    if args.out_dir:
        out_dir = args.out_dir.resolve()
    else:
        out_dir = get_output_root(base_dir).resolve()

    overlaps = args.overlaps if args.overlaps is not None else list(OVERLAPS)
    rng = random.Random(args.seed)

    print(f"base_dir={base_dir}", file=sys.stderr)
    print(f"out_dir={out_dir}", file=sys.stderr)
    print("Loading Libri pool...", file=sys.stderr)
    libri_pool = load_libri_pool(base_dir)
    print(f"  Libri speakers: {len(libri_pool)}", file=sys.stderr)
    print("Loading MyST pool...", file=sys.stderr)
    myst_pool = load_myst_pool(base_dir)
    print(f"  MyST speakers: {len(myst_pool)}", file=sys.stderr)

    if not libri_pool or not myst_pool:
        print("ERROR: empty speaker pool (check BASE_DIR and symlinks)", file=sys.stderr)
        return 1

    total = 0
    for subset in args.subsets:
        for overlap in overlaps:
            ov_name = overlap_dir_name(overlap)
            print(f"Creating {subset}/{ov_name} ({args.n_per_overlap} mixtures)...", file=sys.stderr)
            for i in range(args.n_per_overlap):
                create_one_mixture(
                    rng,
                    subset=subset,
                    overlap=overlap,
                    mix_idx=i,
                    libri_pool=libri_pool,
                    myst_pool=myst_pool,
                    base_dir=base_dir,
                    out_dir=out_dir,
                )
                total += 1

    print(f"Done. Wrote {total} mixtures under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
