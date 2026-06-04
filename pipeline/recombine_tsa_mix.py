#!/usr/bin/env python3
"""
Recombine anonymized target (target_anon.wav) with non-target (nontarget.wav) so the
anonymized target is at the exact same positions as in the original mix.

Alignment: target_anon and nontarget are both time-aligned to the mix (see
RECOMBINATION_ALIGNMENT.md). We require same length and do sample-wise add + normalize.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

SR = 16000
MIX_PEAK = 0.9


def load_wav(path: Path):
    if sf is None:
        from scipy.io import wavfile
        sr, y = wavfile.read(str(path))
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        return y.astype(np.float32), sr
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def recombine_one(
    nontarget_path: Path,
    target_anon_path: Path,
    out_path: Path,
    mix_path: Optional[Path],
    sr: int = SR,
    mix_peak: float = MIX_PEAK,
    strict_length: bool = True,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """
    Combine nontarget + target_anon → final mix. Require same length as mix (or nontarget).
    Returns (success, message).
    """
    if not nontarget_path.exists():
        return False, f"nontarget missing: {nontarget_path}"
    if not target_anon_path.exists():
        return False, f"target_anon missing: {target_anon_path}"

    nontarget, sr_nt = load_wav(nontarget_path)
    target_anon, sr_ta = load_wav(target_anon_path)
    if sr_nt != sr or sr_ta != sr:
        return False, f"sample rate not {sr}: nontarget={sr_nt}, target_anon={sr_ta}"

    n_nt = len(nontarget)
    n_ta = len(target_anon)
    if n_nt != n_ta:
        if strict_length:
            return False, f"length mismatch: nontarget={n_nt}, target_anon={n_ta}"
        # Optional: pad or trim to nontarget length (misalignment risk)
        if n_ta > n_nt:
            target_anon = target_anon[:n_nt]
        else:
            target_anon = np.pad(target_anon, (0, n_nt - n_ta), mode="constant", constant_values=0)

    # Reference length from mix if provided
    if mix_path and mix_path.exists():
        mix, _ = load_wav(mix_path)
        if len(mix) != n_nt and strict_length:
            return False, f"nontarget len={n_nt} != mix len={len(mix)}"

    final = nontarget + target_anon
    peak = np.max(np.abs(final)) + 1e-8
    final = (mix_peak * final / peak).astype(np.float32)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if sf is not None:
            sf.write(str(out_path), final, sr, subtype="PCM_16")
        else:
            from scipy.io import wavfile
            wavfile.write(str(out_path), sr, (np.clip(final, -1, 1) * 32767).astype(np.int16))

    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description="Recombine target_anon + nontarget → final TSA mix")
    ap.add_argument("--manifest", type=Path, default=None, help="tsa_items.jsonl path")
    ap.add_argument("--mode", choices=("selection", "ohnn", "both"), default="both",
                    help="Which anonymized outputs to recombine")
    ap.add_argument("--strict-length", action="store_true", default=True,
                    help="Fail if target_anon length != nontarget (default: True)")
    ap.add_argument("--no-strict-length", action="store_false", dest="strict_length",
                    help="Pad/trim target_anon to nontarget length on mismatch (not recommended)")
    ap.add_argument("--limit", type=int, default=None, help="Max items (debug)")
    ap.add_argument("--dry-run", action="store_true", help="Only check paths and lengths, do not write")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    manifest_path = args.manifest or (script_dir / "manifests" / "tsa_items.jsonl")
    if not manifest_path.exists():
        print("Manifest not found:", manifest_path, file=sys.stderr)
        sys.exit(1)

    ok = 0
    fail = 0
    errors = []

    with open(manifest_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        if args.limit is not None and i >= args.limit:
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            errors.append((i, "invalid JSON"))
            fail += 1
            continue

        nontarget_path = Path(item["nontarget_path"])
        mix_path = Path(item["mix_path"])

        for typ in (["selection", "ohnn"] if args.mode == "both" else [args.mode]):
            if typ == "selection":
                target_anon_path = Path(item["anon_out_path_selection"])
                out_path = Path(item["final_mix_out_path_selection"])
            else:
                target_anon_path = Path(item["anon_out_path_ohnn"])
                out_path = Path(item["final_mix_out_path_ohnn"])

            success, msg = recombine_one(
                nontarget_path,
                target_anon_path,
                out_path,
                mix_path,
                sr=item.get("sr", SR),
                strict_length=args.strict_length,
                dry_run=args.dry_run,
            )
            if success:
                ok += 1
            else:
                fail += 1
                errors.append((item.get("item_id", i), f"[{typ}] {msg}"))

    print(f"Recombine: ok={ok}, fail={fail}")
    for e in errors[:20]:
        print("  ", e)
    if len(errors) > 20:
        print("  ... and", len(errors) - 20, "more")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
