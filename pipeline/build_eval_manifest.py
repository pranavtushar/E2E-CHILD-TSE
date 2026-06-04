#!/usr/bin/env python3
"""
Build evaluation manifest: one row per recombined file (900 × 2 = 1800 rows).
Transcripts and utterance IDs come from MyST and LibriSpeech test manifests
(myst_test_manifest.jsonl, librispeech_test_clean_manifest.jsonl) keyed by
the source utterance paths in each mixture JSON.
Output: evaluations/eval_manifest.jsonl + manifest_summary.json;
subdirs evaluations/eer, evaluations/wer, evaluations/der.
"""
import argparse
import json
import sys
from pathlib import Path

# Import transcript_utils from evaluations/ (same repo)
SCRIPT_DIR = Path(__file__).resolve().parent
_evals_dir = SCRIPT_DIR / "evaluations"
if _evals_dir.is_dir() and str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
try:
    from evaluations.transcript_utils import (
        load_libri_manifest,
        load_myst_manifest,
        lookup_utterance,
    )
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("transcript_utils", _evals_dir / "transcript_utils.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_myst_manifest = _mod.load_myst_manifest
    load_libri_manifest = _mod.load_libri_manifest
    lookup_utterance = _mod.lookup_utterance


def parse_speaker_ids(item_id: str):
    """From item_id like 000000_s1-myst_002116_s2-libri_3575_ov0 get target_spk_id, nontarget_spk_id."""
    target_spk_id = ""
    nontarget_spk_id = ""
    if "_s2-" in item_id and "s1-" in item_id:
        i = item_id.index("s1-")
        j = item_id.index("_s2-")
        target_spk_id = item_id[i + 3 : j]
    if "_s2-" in item_id and "s2-" in item_id:
        i = item_id.index("s2-")
        j = item_id.index("_ov", i)
        nontarget_spk_id = item_id[i + 3 : j]
    return target_spk_id, nontarget_spk_id


def main():
    ap = argparse.ArgumentParser(description="Build evaluation manifest (1800 rows: 900 items × 2 systems)")
    ap.add_argument("--manifest", type=Path, default=None, help="tsa_items.jsonl")
    ap.add_argument("--myst-manifest", type=Path, default=None, help="myst_test_manifest.jsonl (utterance_id + transcript)")
    ap.add_argument("--libri-manifest", type=Path, default=None, help="librispeech_test_clean_manifest.jsonl")
    ap.add_argument("--base-dir", type=Path, default=None, help="Base dir for path resolution (/app vs host)")
    ap.add_argument("--out-dir", type=Path, default=None, help="evaluations/")
    ap.add_argument("--limit", type=int, default=None, help="Max items (debug)")
    ap.add_argument(
        "--systems",
        type=str,
        default="SELECTION,OHNN",
        help="Comma-separated systems to emit (default: SELECTION,OHNN)",
    )
    args = ap.parse_args()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    script_dir = Path(__file__).resolve().parent
    manifest_path = args.manifest or (script_dir / "manifests" / "tsa_items.jsonl")
    base_dir = args.base_dir
    if not base_dir and script_dir.parent.parent.exists():
        base_dir = script_dir.parent.parent  # workspace root
    out_dir = args.out_dir or (script_dir / "evaluations")

    # MyST and Libri manifests: TSE-alpha/00_source_data/manifests by default
    if args.myst_manifest:
        myst_manifest_path = args.myst_manifest
    elif base_dir:
        from path_utils import myst_manifest as _myst_manifest

        myst_manifest_path = _myst_manifest(Path(base_dir))
    else:
        myst_manifest_path = script_dir.parent / "data" / "manifests" / "myst_test_manifest.jsonl"
    if args.libri_manifest:
        libri_manifest_path = args.libri_manifest
    elif base_dir:
        from path_utils import libri_manifest as _libri_manifest

        libri_manifest_path = _libri_manifest(Path(base_dir))
    else:
        libri_manifest_path = script_dir.parent / "data" / "manifests" / "librispeech_test_clean_manifest.jsonl"

    if not manifest_path.exists():
        print("Manifest not found:", manifest_path, file=sys.stderr)
        sys.exit(1)

    print("Loading MyST manifest:", myst_manifest_path, file=sys.stderr)
    myst_lookup = load_myst_manifest(myst_manifest_path, base_dir)
    print("Loading Libri manifest:", libri_manifest_path, file=sys.stderr)
    libri_lookup = load_libri_manifest(libri_manifest_path, base_dir)
    print("MyST entries (path keys):", len(myst_lookup), "Libri entries:", len(libri_lookup), file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eer").mkdir(parents=True, exist_ok=True)
    (out_dir / "wer").mkdir(parents=True, exist_ok=True)
    (out_dir / "der").mkdir(parents=True, exist_ok=True)

    eval_jsonl = out_dir / "eval_manifest.jsonl"
    summary_path = out_dir / "manifest_summary.json"

    rows = []
    missing_json = 0
    missing_s1 = 0
    missing_s2 = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = item["item_id"]
            subset = item["subset"]
            overlap = item["overlap"]
            target_domain = item["target_domain"]
            mix_org_path = item["mix_path"]
            target_spk_id, nontarget_spk_id = parse_speaker_ids(item_id)

            # Resolve mixture JSON path: same dir as mix, stem without .mix.wav -> .json
            mix_path_p = Path(mix_org_path)
            if mix_path_p.name.endswith(".mix.wav"):
                json_name = mix_path_p.name[:- len(".mix.wav")] + ".json"
            else:
                json_name = mix_path_p.stem + ".json"
            json_path = mix_path_p.parent / json_name
            if not json_path.exists() and base_dir:
                # Try under base_dir (e.g. mix_org_path is /app/... and we're on host)
                rel = mix_org_path.replace("/app/", "").replace("\\", "/").lstrip("/")
                json_path = (base_dir / rel).parent / json_name
            if not json_path.exists():
                missing_json += 1
                utt_id_s1, utt_id_s2 = "", ""
                trans_x, trans_y = "", ""
            else:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                utt_s1 = (meta.get("utt_s1_mix_path") or meta.get("utt_s1_path") or "").strip()
                utt_s2 = (meta.get("utt_s2_mix_path") or meta.get("utt_s2_path") or "").strip()
                if not utt_s1:
                    missing_s1 += 1
                if not utt_s2:
                    missing_s2 += 1
                utt_id_s1, trans_x = lookup_utterance(utt_s1, myst_lookup, libri_lookup, base_dir) if utt_s1 else ("", "")
                utt_id_s2, trans_y = lookup_utterance(utt_s2, myst_lookup, libri_lookup, base_dir) if utt_s2 else ("", "")

            for system in systems:
                if system == "SELECTION":
                    mix_anon_path = item["final_mix_out_path_selection"]
                    target_anon_path = item["anon_out_path_selection"]
                else:
                    mix_anon_path = item["final_mix_out_path_ohnn"]
                    target_anon_path = item["anon_out_path_ohnn"]

                eval_id = "{}_{}".format(item_id, system)
                row = {
                    "eval_id": eval_id,
                    "item_id": item_id,
                    "subset": subset,
                    "overlap": overlap,
                    "system": system,
                    "target_domain": target_domain,
                    "mix_org_path": mix_org_path,
                    "mix_anon_path": mix_anon_path,
                    "target_org_path": item.get("target_path") or "",
                    "nontarget_path": item.get("nontarget_path") or "",
                    "tse_hat_path": item["tse_hat_path"],
                    "target_anon_path": target_anon_path,
                    "ref_path": item["ref_path"],
                    "target_spk_id": target_spk_id,
                    "nontarget_spk_id": nontarget_spk_id,
                    "utt_id_target": utt_id_s1,
                    "utt_id_nontarget": utt_id_s2,
                    "transcript_target": trans_x,
                    "transcript_nontarget": trans_y,
                    "sr": item.get("sr", 16000),
                }
                rows.append(row)

    with open(eval_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_system = {"SELECTION": 0, "OHNN": 0}
    by_subset = {}
    by_domain = {}
    for r in rows:
        by_system[r["system"]] = by_system.get(r["system"], 0) + 1
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
        by_domain[r["target_domain"]] = by_domain.get(r["target_domain"], 0) + 1

    summary = {
        "total_rows": len(rows),
        "total_items": len(rows) // 2,
        "by_system": by_system,
        "by_subset": by_subset,
        "by_target_domain": by_domain,
        "eval_manifest_path": str(eval_jsonl),
        "myst_manifest": str(myst_manifest_path),
        "libri_manifest": str(libri_manifest_path),
        "missing_mixture_json": missing_json,
        "missing_utt_s1": missing_s1,
        "missing_utt_s2": missing_s2,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Wrote {} rows to {}".format(len(rows), eval_jsonl))
    print("Summary: {}".format(summary_path))
    print("  By system: {}".format(by_system))
    print("  By subset: {}".format(by_subset))
    print("  By target_domain: {}".format(by_domain))
    if missing_json or missing_s1 or missing_s2:
        print("  Warnings: missing_mixture_json={} missing_utt_s1={} missing_utt_s2={}".format(missing_json, missing_s1, missing_s2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
