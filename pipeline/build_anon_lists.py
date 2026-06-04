#!/usr/bin/env python3
"""
From manifests/tsa_items.jsonl, write per-domain list files of tse_hat_path (one per line).
Used by Selection and OHNN anonymization runners.
--smoke: write 2 child + 1 adult (3 total) for smoke test.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Build child/adult tse_hat list files from manifest")
    ap.add_argument("--manifest", type=Path, default=None, help="tsa_items.jsonl path")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write list files")
    ap.add_argument("--smoke", action="store_true", help="Write only 3 items: 2 child + 1 adult")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    manifest = args.manifest or (script_dir / "manifests" / "tsa_items.jsonl")
    out_dir = args.out_dir or (script_dir / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        print("Manifest not found:", manifest)
        sys.exit(1)

    child_paths = []
    adult_paths = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tse = item.get("tse_hat_path")
            domain = item.get("target_domain")
            if not tse:
                continue
            if domain == "child":
                child_paths.append(tse)
            elif domain == "adult":
                adult_paths.append(tse)

    if args.smoke:
        child_paths = child_paths[:2]
        adult_paths = adult_paths[:1]
        child_list = out_dir / "tsa_anon_child_smoke.txt"
        adult_list = out_dir / "tsa_anon_adult_smoke.txt"
    else:
        child_list = out_dir / "tsa_anon_child_list.txt"
        adult_list = out_dir / "tsa_anon_adult_list.txt"

    with open(child_list, "w") as f:
        f.write("\n".join(child_paths) + ("\n" if child_paths else ""))
    with open(adult_list, "w") as f:
        f.write("\n".join(adult_paths) + ("\n" if adult_paths else ""))

    print(f"Child list: {child_list} ({len(child_paths)} paths)")
    print(f"Adult list: {adult_list} ({len(adult_paths)} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
