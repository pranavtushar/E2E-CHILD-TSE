#!/usr/bin/env python3
"""Pick mixtures per subset spread across overlap folders (for smoke tests)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SUBSETS_ORDER = ("CA", "CC", "AA")
OVERLAP_DIRS = ("ov0", "ov20", "ov40", "ov60", "ov80", "ov100")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument(
        "--per-subset",
        type=int,
        default=10,
        help="Total items per subset, spread across ov0..ov100 (not all from ov0)",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    n_ov = len(OVERLAP_DIRS)
    base = args.per_subset // n_ov
    extra = args.per_subset % n_ov
    # First `extra` overlap bands get base+1 items each
    take_per_ov = [base + (1 if i < extra else 0) for i in range(n_ov)]

    out_rows = []
    for subset in SUBSETS_ORDER:
        picked = 0
        for ov_idx, ov in enumerate(OVERLAP_DIRS):
            need = take_per_ov[ov_idx]
            if need <= 0:
                continue
            folder = data_root / subset / ov
            if not folder.is_dir():
                continue
            count = 0
            for mix_path in sorted(folder.glob("*.mix.wav")):
                if count >= need:
                    break
                base_name = mix_path.stem.replace(".mix", "")
                ov_pct = int(ov.replace("ov", ""))
                out_rows.append(
                    {
                        "item_id": base_name,
                        "subset": subset,
                        "overlap_dir": ov,
                        "overlap_pct": ov_pct,
                    }
                )
                count += 1
                picked += 1
        print(f"  {subset}: {picked} (across overlaps)", file=sys.stderr)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(out_rows)} smoke item_ids -> {args.out_manifest}", file=sys.stderr)
    return 0 if out_rows else 1


if __name__ == "__main__":
    sys.exit(main())
