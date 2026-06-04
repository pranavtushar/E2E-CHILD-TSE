#!/usr/bin/env python3
"""
Channel WER only (Whisper large-v3):

  s1: MFA ground truth (5s)  vs  Whisper(target_anon)
  s2: MFA ground truth (5s)  vs  Whisper(nontarget)  [unchanged in recombine]

Exports: detail CSV, Excel (Detail + WER%% matrices by subset x overlap), JSON summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ALPHA_ROOT = SCRIPT_DIR.parents[2]
PIPE_DIR = ALPHA_ROOT / "02_pipeline"
if str(PIPE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import resolve_bundle_path  # noqa: E402
from tsa_wer_utils import compute_wer_single  # noqa: E402

SUBSETS = ("AA", "CA", "CC")
OVERLAP_PCTS = (0, 20, 40, 60, 80, 100)


def resolve_audio(path_str: str, alpha: Path) -> Path:
    s = (path_str or "").strip().replace("\\", "/")
    if "00_source_data/" in s:
        s = s[s.index("00_source_data/") :]
    p = Path(s)
    if p.is_file():
        return p.resolve()
    if "/TSE-alpha/" in s:
        tail = s.split("/TSE-alpha/", 1)[1]
        cand = alpha / tail
        if cand.is_file():
            return cand.resolve()
    return resolve_bundle_path(s, alpha)


def load_catalog(catalog_path: Path) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    if not catalog_path.is_file():
        return out
    with catalog_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            iid = (row.get("item_id") or "").strip()
            if iid:
                out[iid] = (
                    (row.get("mfa_transcript_s1") or "").strip(),
                    (row.get("mfa_transcript_s2") or "").strip(),
                )
    return out


def overlap_pct_from_row(row: Dict[str, Any]) -> int:
    if row.get("overlap_pct") is not None and str(row.get("overlap_pct")).strip() != "":
        return int(row["overlap_pct"])
    ov_dir = row.get("overlap_dir") or ""
    if str(ov_dir).startswith("ov"):
        return int(str(ov_dir).replace("ov", ""))
    ov = row.get("overlap")
    if ov is not None and str(ov).strip() != "":
        try:
            v = float(ov)
            return int(round(v * 100)) if v <= 1.0 else int(round(v))
        except ValueError:
            pass
    iid = row.get("item_id", "")
    m = re.search(r"_ov(\d+)(?:_|$)", iid)
    return int(m.group(1)) if m else 0


def load_whisper(backend: str, model_name: str, device: str, compute_type: str):
    dev = device.split(":")[0] if ":" in device else device
    if backend == "faster":
        from faster_whisper import WhisperModel

        return ("faster", WhisperModel(model_name, device=dev, compute_type=compute_type))
    import whisper

    return ("openai", whisper.load_model(model_name, device=dev))


def transcribe(pack: Tuple[str, Any], wav: Path, language: str) -> str:
    b, model = pack
    if not wav.is_file():
        return ""
    if b == "faster":
        segs, _ = model.transcribe(str(wav), language=language, beam_size=5)
        return " ".join(s.text.strip() for s in segs).strip()
    return (model.transcribe(str(wav), language=language, verbose=False).get("text") or "").strip()


def paths_for_item(alpha: Path, item: Dict[str, Any]) -> Dict[str, Path]:
    mix = item.get("mix_org_path") or item.get("mix_path") or ""
    mp = resolve_audio(mix, alpha)
    parent = mp.parent
    stem = mp.stem.replace(".mix.wav", "") if mp.name.endswith(".mix.wav") else mp.stem
    if stem.endswith(".mix"):
        stem = stem[: -len(".mix")]

    target_anon = item.get("target_anon_path") or item.get("anon_out_path_selection")
    if target_anon:
        s1_regen = resolve_audio(target_anon, alpha)
    else:
        s1_regen = parent / f"{stem}.target_anon.wav"

    nontarget = item.get("nontarget_path") or str(parent / f"{stem}.nontarget.wav")
    s2_regen = resolve_audio(nontarget, alpha)

    return {"s1_regen": s1_regen, "s2_regen": s2_regen}


def aggregate_matrix(
    acc: Dict[Tuple[str, int], List[float]],
) -> Dict[str, Dict[int, Optional[float]]]:
    """subset -> overlap_pct -> mean WER fraction."""
    out: Dict[str, Dict[int, Optional[float]]] = {s: {} for s in SUBSETS}
    for sub in SUBSETS:
        for ov in OVERLAP_PCTS:
            vals = acc.get((sub, ov), [])
            out[sub][ov] = sum(vals) / len(vals) if vals else None
    return out


def matrix_to_rows(mat: Dict[str, Dict[int, Optional[float]]], as_percent: bool = True) -> List[List[Any]]:
    header = ["overlap_pct"] + list(SUBSETS)
    rows = [header]
    for ov in OVERLAP_PCTS:
        row: List[Any] = [ov]
        for sub in SUBSETS:
            v = mat.get(sub, {}).get(ov)
            if v is None:
                row.append("")
            else:
                row.append(round(v * 100.0, 2) if as_percent else round(v, 4))
        rows.append(row)
    return rows


def write_xlsx(
    detail_rows: List[Dict[str, Any]],
    out_path: Path,
    mat_s1: Dict[str, Dict[int, Optional[float]]],
    mat_s2: Dict[str, Dict[int, Optional[float]]],
) -> bool:
    try:
        from openpyxl import Workbook
    except ImportError:
        print("Skip XLSX (pip install openpyxl)", file=sys.stderr)
        return False

    wb = Workbook()
    ws_d = wb.active
    ws_d.title = "Detail"
    if detail_rows:
        headers = list(detail_rows[0].keys())
        ws_d.append(headers)
        for r in detail_rows:
            ws_d.append([r.get(h, "") for h in headers])

    for title, mat in (("WER_s1_pct", mat_s1), ("WER_s2_pct", mat_s2)):
        ws = wb.create_sheet(title)
        for row in matrix_to_rows(mat, as_percent=True):
            ws.append(row)

    wb.save(out_path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="s1/s2 channel WER vs MFA (Whisper large-v3)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--catalog", type=Path, default=None)
    ap.add_argument("--backend", default="auto", choices=("auto", "openai", "faster"))
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="en")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--smoke-by-subset",
        action="store_true",
        help="Only 3 items (one per AA/CA/CC). Default smoke uses full smoke manifest (e.g. 30 rows).",
    )
    ap.add_argument("--write-xlsx", action="store_true", default=True)
    ap.add_argument("--no-write-xlsx", action="store_false", dest="write_xlsx")
    args = ap.parse_args()

    alpha = (args.base_dir or ALPHA_ROOT).resolve()
    catalog = load_catalog(
        args.catalog
        or (alpha / "data/manifests/mixture_mfa_catalog_docker.csv")
        or (alpha / "00_source_data/manifests/mixture_mfa_catalog_docker.csv")
    )

    backend = args.backend
    if backend == "auto":
        try:
            import whisper  # noqa: F401

            backend = "openai"
        except ImportError:
            backend = "faster"
    pack = load_whisper(backend, args.model, args.device, args.compute_type)

    rows_in: List[Dict[str, Any]] = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows_in.append(json.loads(line))

    if args.smoke_by_subset:
        seen = set()
        filtered = []
        for r in rows_in:
            key = r.get("subset", "")
            if key in seen:
                continue
            seen.add(key)
            filtered.append(r)
        rows_in = filtered
    elif args.limit:
        rows_in = rows_in[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "s1_s2_wer_detail.csv"
    summary_path = args.out_dir / "s1_s2_wer_summary.json"
    xlsx_path = args.out_dir / "s1_s2_wer_pack.xlsx"

    cache: Dict[str, str] = {}
    detail_rows: List[Dict[str, Any]] = []
    acc_s1: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    acc_s2: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    for item in rows_in:
        item_id = item.get("item_id", "")
        subset = (item.get("subset") or "").strip().upper()
        ov_pct = overlap_pct_from_row(item)
        mfa_s1, mfa_s2 = catalog.get(item_id, ("", ""))
        if not mfa_s1 and not mfa_s2:
            mix = item.get("mix_org_path") or item.get("mix_path", "")
            if mix:
                mp = resolve_audio(mix, alpha)
                jp = mp.parent / (mp.stem.replace(".mix.wav", "") + ".json")
                if jp.is_file():
                    meta = json.loads(jp.read_text(encoding="utf-8"))
                    mfa_s1 = (meta.get("mfa_transcript_s1") or "").strip()
                    mfa_s2 = (meta.get("mfa_transcript_s2") or "").strip()

        paths = paths_for_item(alpha, item)
        err = ""
        hyp_s1 = hyp_s2 = ""
        w1 = w2 = None
        try:
            k1, k2 = str(paths["s1_regen"]), str(paths["s2_regen"])
            if k1 not in cache:
                cache[k1] = transcribe(pack, paths["s1_regen"], args.language)
            if k2 not in cache:
                cache[k2] = transcribe(pack, paths["s2_regen"], args.language)
            hyp_s1, hyp_s2 = cache[k1], cache[k2]
            w1, _, _, _, _ = compute_wer_single(mfa_s1, hyp_s1)
            w2, _, _, _, _ = compute_wer_single(mfa_s2, hyp_s2)
            if w1 is not None and subset in SUBSETS:
                acc_s1[(subset, ov_pct)].append(w1)
            if w2 is not None and subset in SUBSETS:
                acc_s2[(subset, ov_pct)].append(w2)
        except Exception as e:
            err = str(e)

        detail_rows.append(
            {
                "item_id": item_id,
                "subset": subset,
                "overlap_pct": ov_pct,
                "ground_truth_s1_mfa": mfa_s1,
                "extracted_s1_whisper_target_anon": hyp_s1,
                "wer_s1": f"{w1:.4f}" if w1 is not None else "",
                "wer_s1_pct": f"{100.0 * w1:.2f}" if w1 is not None else "",
                "audio_s1_regen": str(paths["s1_regen"]),
                "ground_truth_s2_mfa": mfa_s2,
                "extracted_s2_whisper_nontarget": hyp_s2,
                "wer_s2": f"{w2:.4f}" if w2 is not None else "",
                "wer_s2_pct": f"{100.0 * w2:.2f}" if w2 is not None else "",
                "audio_s2_regen": str(paths["s2_regen"]),
                "error": err,
            }
        )

    if detail_rows:
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    mat_s1 = aggregate_matrix(acc_s1)
    mat_s2 = aggregate_matrix(acc_s2)

    for name, mat in (("wer_s1_pct_by_subset_overlap", mat_s1), ("wer_s2_pct_by_subset_overlap", mat_s2)):
        md_path = args.out_dir / f"{name}.md"
        lines = [f"# {name} (mean WER %%, not pooled across overlaps)\n", "| overlap_pct | AA | CA | CC |", "|---|---|---|---|"]
        for row in matrix_to_rows(mat, as_percent=True)[1:]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "model": args.model,
        "backend": pack[0],
        "n_items": len(detail_rows),
        "note": "Matrices are per (subset, overlap_pct). Do not use pooled mean across overlaps.",
        "wer_s1_pct_by_subset_overlap": {
            sub: {str(ov): (None if mat_s1[sub][ov] is None else round(100 * mat_s1[sub][ov], 2)) for ov in OVERLAP_PCTS}
            for sub in SUBSETS
        },
        "wer_s2_pct_by_subset_overlap": {
            sub: {str(ov): (None if mat_s2[sub][ov] is None else round(100 * mat_s2[sub][ov], 2)) for ov in OVERLAP_PCTS}
            for sub in SUBSETS
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.write_xlsx:
        if write_xlsx(detail_rows, xlsx_path, mat_s1, mat_s2):
            print(f"Wrote {xlsx_path}", file=sys.stderr)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
