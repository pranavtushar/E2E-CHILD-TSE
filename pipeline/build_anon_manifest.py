#!/usr/bin/env python3
"""
Build tsa_items.jsonl after TSE: paths for anonymization and recombination.
Run after run_tsa_tse.py; consumed by build_anon_lists, anon runners, recombine_tsa_mix.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

SUBSETS_ORDER = ("CA", "CC", "AA")
OVERLAP_DIRS = ("ov0", "ov20", "ov40", "ov60", "ov80", "ov100")
OUTPUT_SUFFIX = "tse_hat"


def collect_items(tsa_root: Path) -> Iterator[Tuple[str, str, Path, Path, str]]:
    for subset in SUBSETS_ORDER:
        for ov in OVERLAP_DIRS:
            folder = tsa_root / subset / ov
            if not folder.is_dir():
                continue
            for mix_path in sorted(folder.glob("*.mix.wav")):
                base_name = mix_path.stem.replace(".mix", "")
                ref_path = mix_path.parent / (base_name + ".ref.wav")
                yield subset, ov, mix_path, ref_path, base_name


def domain_for_subset(subset: str) -> Tuple[str, str]:
    if subset == "AA":
        return "adult", "adult_model"
    return "child", "child_model"


def overlap_int(ov_dir: str) -> int:
    return int(ov_dir.replace("ov", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build tsa_items.jsonl for anon + recombine")
    ap.add_argument("--data-root", type=Path, required=True, help="TSA mixtures root")
    ap.add_argument("--tse-root", type=Path, required=True, help="TSE_OUT root (*.tse_hat.wav)")
    ap.add_argument("--anon-root-selection", type=Path, required=True)
    ap.add_argument("--anon-root-ohnn", type=Path, default=None)
    ap.add_argument("--recombined-root-selection", type=Path, required=True)
    ap.add_argument("--recombined-root-ohnn", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for tsa_items.jsonl")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument(
        "--only-items-manifest",
        type=Path,
        default=None,
        help="jsonl with item_id; only include these mixtures",
    )
    args = ap.parse_args()

    allowed_ids = None
    if args.only_items_manifest is not None:
        allowed_ids = set()
        with open(args.only_items_manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = (row.get("item_id") or "").strip()
                if iid:
                    allowed_ids.add(iid)
        print(f"Filter: {len(allowed_ids)} item_ids from {args.only_items_manifest}", file=sys.stderr)

    data_root = args.data_root.resolve()
    tse_root = args.tse_root.resolve()
    anon_sel = args.anon_root_selection.resolve()
    anon_ohnn = (args.anon_root_ohnn or args.anon_root_selection.parent / "anon_ohnn").resolve()
    recom_sel = args.recombined_root_selection.resolve()
    recom_ohnn = (args.recombined_root_ohnn or args.recombined_root_selection.parent / "recombined_ohnn").resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "tsa_items.jsonl"

    if not data_root.is_dir():
        print(f"data-root not found: {data_root}", file=sys.stderr)
        return 1
    if not tse_root.is_dir():
        print(f"tse-root not found: {tse_root}", file=sys.stderr)
        return 1

    rows = []
    skipped_no_tse = []
    counts_subset_ov: Dict[str, int] = defaultdict(int)
    counts_domain: Dict[str, int] = defaultdict(int)

    for subset, ov, mix_path, ref_path, base_name in collect_items(data_root):
        if allowed_ids is not None and base_name not in allowed_ids:
            continue
        target_path = mix_path.parent / (base_name + ".target.wav")
        nontarget_path = mix_path.parent / (base_name + ".nontarget.wav")
        tse_hat = tse_root / subset / ov / (base_name + f".{OUTPUT_SUFFIX}.wav")
        if not tse_hat.is_file():
            skipped_no_tse.append(str(tse_hat))
            continue

        target_domain, anonymizer_profile = domain_for_subset(subset)
        anon_out_sel = anon_sel / subset / ov / (base_name + ".target_anon.wav")
        anon_out_ohnn = anon_ohnn / subset / ov / (base_name + ".target_anon.wav")
        final_sel = recom_sel / subset / ov / (base_name + ".mix_anon.wav")
        final_ohnn = recom_ohnn / subset / ov / (base_name + ".mix_anon.wav")

        row = {
            "item_id": base_name,
            "subset": subset,
            "overlap": overlap_int(ov),
            "mix_path": str(mix_path.resolve()),
            "ref_path": str(ref_path.resolve()),
            "target_path": str(target_path.resolve()),
            "nontarget_path": str(nontarget_path.resolve()),
            "tse_hat_path": str(tse_hat.resolve()),
            "target_domain": target_domain,
            "anonymizer_profile": anonymizer_profile,
            "anon_out_path_selection": str(anon_out_sel.resolve()),
            "anon_out_path_ohnn": str(anon_out_ohnn.resolve()),
            "final_mix_out_path_selection": str(final_sel.resolve()),
            "final_mix_out_path_ohnn": str(final_ohnn.resolve()),
            "sr": args.sr,
        }
        rows.append(row)
        counts_subset_ov[f"{subset}/{ov}"] += 1
        counts_domain[target_domain] += 1

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "total_items": len(rows),
        "skipped": len(skipped_no_tse),
        "counts_by_subset_overlap": dict(sorted(counts_subset_ov.items())),
        "counts_by_target_domain": dict(counts_domain),
        "paths": {
            "data_root": str(data_root),
            "tse_root": str(tse_root),
            "anon_root_selection": str(anon_sel),
            "anon_root_ohnn": str(anon_ohnn),
            "recombined_root_selection": str(recom_sel),
            "recombined_root_ohnn": str(recom_ohnn),
            "manifest": str(out_jsonl),
        },
    }
    with open(out_dir / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(rows)} items -> {out_jsonl}", file=sys.stderr)
    if skipped_no_tse:
        print(f"Skipped {len(skipped_no_tse)} (missing tse_hat)", file=sys.stderr)
        for p in skipped_no_tse[:5]:
            print(f"  {p}", file=sys.stderr)
    if not rows:
        print("ERROR: no manifest rows written (run TSE first?)", file=sys.stderr)
        return 1
    if skipped_no_tse:
        print(f"Note: {len(skipped_no_tse)} mixtures skipped (no tse_hat yet)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
