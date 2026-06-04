#!/usr/bin/env python3
"""
Build EER protocol: enrollment (original, clean, disjoint from mix/ref) and trials for Step2.
OO: trial = tse_hat.wav.  OA: trial = target_anon.wav.
Enrollment: K utterances per target speaker from MyST/Libri manifests, excluding any path used in mixtures.
Trials: one target trial (enroll_id=target_spk_id, test_id=eval_id) + N_IMPOSTOR impostor trials per eval row.
Output: Kaldi-style dirs (wav.scp, utt2spk, spk2gender) + trials file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def norm_path(p: str, base_dir: Path | None) -> str:
    """Resolve manifest/mix paths to files under base_dir (00_source_data, regenerated/, etc.)."""
    s = (p or "").strip().replace("\\", "/")
    if not base_dir or not s:
        return s
    bd = base_dir.resolve()
    bd_s = str(bd).replace("\\", "/").rstrip("/")
    if s.startswith(bd_s + "/") or s == bd_s:
        return str(Path(s).resolve())
    try:
        import sys

        pipe = bd / "02_pipeline"
        if str(pipe) not in sys.path:
            sys.path.insert(0, str(pipe))
        from path_utils import resolve_bundle_path, rewrite_legacy_path

        if s.startswith("/app/"):
            tail = s[len("/app/") :]
            # /app/TSE-alpha/... must not become /app/TSE-alpha/TSE-alpha/...
            if tail.startswith(bd.name + "/"):
                return str((Path("/app") / tail).resolve())
        resolved = resolve_bundle_path(s, bd)
        if resolved.is_file():
            return str(resolved)
        rewritten = rewrite_legacy_path(s, bd)
        if Path(rewritten).is_file():
            return rewritten
        return rewritten
    except Exception:
        return s


def load_used_paths(mixed_root: Path, base_dir: Path | None) -> set:
    """Collect all utt_s1_mix_path, utt_s2_mix_path, utt_s1_ref_path from mixture JSONs."""
    used = set()
    for jpath in mixed_root.rglob("*.json"):
        if "_ov" not in jpath.stem:
            continue
        try:
            meta = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("utt_s1_mix_path", "utt_s2_mix_path", "utt_s1_ref_path"):
            p = meta.get(key)
            if p:
                used.add(norm_path(p, base_dir))
                used.add(p.strip())
    return used


def load_myst_by_speaker(manifest_path: Path) -> dict:
    """speaker_id -> list of (utterance_id, audio_path) where path is normalized_audio_path or home_audio_path."""
    from collections import defaultdict
    spk2utts = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = (row.get("utterance_id") or "").strip()
            if not uid or not uid.startswith("myst_"):
                continue
            # speaker = myst_002116 from myst_002116_2014-...
            parts = uid.split("_")
            if len(parts) >= 2:
                spk = "myst_" + parts[1]
            else:
                continue
            path = row.get("normalized_audio_path") or row.get("home_audio_path")
            if path:
                spk2utts[spk].append((uid, path.strip()))
    return dict(spk2utts)


def load_libri_by_speaker(manifest_path: Path) -> dict:
    """speaker_id (numeric str) -> list of (utterance_id, audio_path)."""
    from collections import defaultdict
    spk2utts = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = (row.get("utterance_id") or "").strip()
            if not uid or "-" not in uid:
                continue
            spk = uid.split("-")[0]
            path = row.get("normalised_audio_path") or row.get("normalized_audio_path")
            if path:
                spk2utts[spk].append((uid, path.strip()))
    return dict(spk2utts)


def main():
    ap = argparse.ArgumentParser(description="Build EER enrollment + trials (Step2 OO/OA)")
    ap.add_argument("--eval-manifest", type=Path, default=None, help="eval_manifest.jsonl")
    ap.add_argument("--mixed-root", type=Path, default=None, help="TSA_MIXED root (mixture JSONs)")
    ap.add_argument("--myst-manifest", type=Path, default=None)
    ap.add_argument("--libri-manifest", type=Path, default=None)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="eer_data/")
    ap.add_argument("--K", type=int, default=5, help="Enrollment utterances per speaker")
    ap.add_argument("--n-impostors", type=int, default=50, help="Impostor trials per eval row")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Max eval rows (debug)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    # evaluations/eer -> parent = evaluations, parent.parent = tsa_mix
    tsa_mix_root = script_dir.parent.parent
    base = tsa_mix_root.parent.parent  # tse/tsa_mix -> base = repo root (e.g. /app or /mnt/hdd/pranav)
    eval_path = args.eval_manifest or (tsa_mix_root / "evaluations" / "eval_manifest.jsonl")
    mixed_root = args.mixed_root or (tsa_mix_root / "data" / "TSA_MIXED")
    base_dir = args.base_dir
    if base_dir is None:
        base_dir = base  # use repo root for path resolution
    data_dir = base / "data" if (base / "data").is_dir() else base / "00_source_data"
    myst_path = args.myst_manifest or (data_dir / "manifests" / "myst_test_manifest.jsonl")
    libri_path = args.libri_manifest or (data_dir / "manifests" / "librispeech_test_clean_manifest.jsonl")
    out_dir = args.out_dir or (script_dir / "eer_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        print("Eval manifest not found:", eval_path, file=sys.stderr)
        return 1

    print("Loading used paths from mixture JSONs...", file=sys.stderr)
    used_paths = load_used_paths(mixed_root, base_dir)
    # Also add /app variants for comparison
    if base_dir:
        for p in list(used_paths):
            if str(p).startswith(str(base_dir)):
                used_paths.add("/app/" + str(Path(p).relative_to(base_dir)))
    print("  Used paths:", len(used_paths), file=sys.stderr)

    print("Loading manifests by speaker...", file=sys.stderr)
    myst_spk2utts = load_myst_by_speaker(myst_path) if myst_path.exists() else {}
    libri_spk2utts = load_libri_by_speaker(libri_path) if libri_path.exists() else {}
    print("  MyST speakers:", len(myst_spk2utts), "Libri speakers:", len(libri_spk2utts), file=sys.stderr)

    def path_used(p: str) -> bool:
        pn = norm_path(p, base_dir)
        if p in used_paths or pn in used_paths:
            return True
        return False

    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print("Eval rows:", len(rows), file=sys.stderr)

    # Unique target speakers
    target_speakers = set(r.get("target_spk_id", "") for r in rows if r.get("target_spk_id"))
    all_speakers = target_speakers.copy()
    for r in rows:
        all_speakers.add(r.get("nontarget_spk_id", ""))
    all_speakers.discard("")

    # Enrollment: for each target speaker, pick K utterances not in used_paths
    enroll_wav = {}
    enroll_utt2spk = {}
    enroll_spk2gender = {}
    rng = random.Random(args.seed)
    for spk in sorted(target_speakers):
        if spk.startswith("myst_"):
            utts = myst_spk2utts.get(spk, [])
        else:
            # Eval manifest uses libri_3575; Libri manifest keys are numeric (3575)
            libri_key = spk.replace("libri_", "", 1) if spk.startswith("libri_") else spk
            utts = libri_spk2utts.get(libri_key, [])
        available = [(u, p) for u, p in utts if not path_used(p)]
        if len(available) < args.K:
            print("  Warning: speaker", spk, "has only", len(available), "disjoint utts (need", args.K, ")", file=sys.stderr)
        chosen = rng.sample(available, min(args.K, len(available)))
        for i, (uid, path) in enumerate(chosen):
            eid = f"{spk}_enroll_{i}"
            enroll_wav[eid] = norm_path(path, base_dir)
            enroll_utt2spk[eid] = spk
        enroll_spk2gender[spk] = "u"

    (out_dir / "enroll_orig").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "enroll_orig" / "wav.scp", "w", encoding="utf-8") as f:
        for k in sorted(enroll_wav.keys()):
            f.write(f"{k} {enroll_wav[k]}\n")
    with open(out_dir / "enroll_orig" / "utt2spk", "w", encoding="utf-8") as f:
        for k in sorted(enroll_utt2spk.keys()):
            f.write(f"{k} {enroll_utt2spk[k]}\n")
    with open(out_dir / "enroll_orig" / "spk2gender", "w", encoding="utf-8") as f:
        for k in sorted(enroll_spk2gender.keys()):
            f.write(f"{k} {enroll_spk2gender[k]}\n")

    # Test dirs: Step2_OO (tse_hat), Step2_OA (target_anon). Same trial IDs (eval_id), different wav paths.
    test_oo_wav = {}
    test_oa_wav = {}
    test_utt2spk = {}
    for r in rows:
        eid = r.get("eval_id", "")
        if not eid:
            continue
        tse_hat = r.get("tse_hat_path", "")
        target_anon = r.get("target_anon_path", "")
        spk = r.get("target_spk_id", "")
        if tse_hat:
            test_oo_wav[eid] = norm_path(tse_hat, base_dir)
        if target_anon:
            test_oa_wav[eid] = norm_path(target_anon, base_dir)
        test_utt2spk[eid] = spk

    for name, wav_dict in [("test_Step2_OO", test_oo_wav), ("test_Step2_OA", test_oa_wav)]:
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "wav.scp", "w", encoding="utf-8") as f:
            for k in sorted(wav_dict.keys()):
                f.write(f"{k} {wav_dict[k]}\n")
        with open(d / "utt2spk", "w", encoding="utf-8") as f:
            for k in sorted(wav_dict.keys()):
                f.write(f"{k} {test_utt2spk.get(k, '')}\n")

    # Trials: (enroll_spk_id, test_id, target|nontarget). One target + N_IMPOSTOR nontarget per eval_id.
    impostor_pool = [s for s in sorted(all_speakers)]
    trials = []
    eval_id_to_meta = {}
    for r in rows:
        eid = r.get("eval_id", "")
        if not eid:
            continue
        target_spk = r.get("target_spk_id", "")
        eval_id_to_meta[eid] = {"overlap": r.get("overlap"), "subset": r.get("subset"), "system": r.get("system")}
        trials.append((target_spk, eid, "target"))
        others = [s for s in impostor_pool if s != target_spk]
        if len(others) > args.n_impostors:
            others = rng.sample(others, args.n_impostors)
        for imp in others:
            trials.append((imp, eid, "nontarget"))

    with open(out_dir / "trials_Step2", "w", encoding="utf-8") as f:
        for en, te, label in trials:
            f.write(f"{en} {te} {label}\n")

    with open(out_dir / "eval_id_metadata.json", "w", encoding="utf-8") as f:
        json.dump(eval_id_to_meta, f, indent=0)

    print("Wrote", out_dir / "enroll_orig", "| test_Step2_OO | test_Step2_OA | trials_Step2 | eval_id_metadata.json", file=sys.stderr)
    print("Enrollment utts:", len(enroll_wav), "| Trial (OO/OA) utts:", len(test_oo_wav), "| Trials:", len(trials), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
