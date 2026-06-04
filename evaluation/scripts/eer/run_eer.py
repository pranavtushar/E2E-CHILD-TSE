#!/usr/bin/env python3
"""
Compute EER for TSA: OO (enroll orig vs trial tse_hat) and OA (enroll orig vs trial target_anon).
Uses ECAPA-TDNN (SpeechBrain pretrained or local). No retraining.
Reports EER overall and by overlap / subset / system.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def read_kaldi_scp(path: Path) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
            elif len(parts) == 1:
                out[parts[0]] = ""
    return out


def load_ecapa(device: str, model_path: Path | None = None):
    """Load ECAPA-TDNN from SpeechBrain. If model_path given and exists, load from there."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            raise ImportError("pip install speechbrain for EER (ECAPA)")
    if model_path and Path(model_path).exists():
        p = Path(model_path)
        savedir = p.parent if p.is_file() else p
        return EncoderClassifier.from_hparams(source=str(p), savedir=str(savedir), run_opts={"device": device})
    script_dir = Path(__file__).resolve().parent
    bundle_root = script_dir.parent.parent.parent  # evaluation/scripts/eer -> bundle root
    savedir = bundle_root / "vendor" / "speechbrain" / "spkrec-ecapa-voxceleb"
    if not savedir.is_dir():
        savedir = script_dir / "pretrained_models" / "spkrec-ecapa-voxceleb"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": device},
    )


def extract_embedding(model, wav_path: str, device: str, sr: int = 16000):
    """Load wav and return 1D embedding vector."""
    try:
        wav, fs = torchaudio.load(wav_path)
    except Exception:
        return None
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if fs != sr:
        wav = torchaudio.functional.resample(wav, fs, sr)
    wav = wav.squeeze(0)
    if wav.shape[0] < sr * 0.5:
        return None
    with torch.no_grad():
        emb = model.encode_batch(wav.unsqueeze(0).to(device))
    if emb is None:
        return None
    return emb.squeeze().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Compute EER (Step2 OO and OA)")
    ap.add_argument("--eer-data", type=Path, default=None, help="eer_data/ from build_eer_protocol")
    ap.add_argument("--condition", choices=["Step2_OO", "Step2_OA"], default=None, help="Run one or both if not set")
    ap.add_argument("--ecapa", type=Path, default=None, help="ECAPA model path (else SpeechBrain pretrained)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=None, help="results/")
    ap.add_argument("--limit", type=int, default=None, help="Max trial utts (debug)")
    ap.add_argument("--t-norm", action="store_true", help="Apply T-norm (cohort normalization per test); can raise OA EER toward 40s")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    eer_data = args.eer_data or (script_dir / "eer_data")
    out_dir = args.out_dir or (script_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (eer_data / "enroll_orig" / "wav.scp").exists():
        print("Run build_eer_protocol.py first. Missing", eer_data / "enroll_orig" / "wav.scp", file=sys.stderr)
        return 1

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU", file=sys.stderr)
    print("Loading ECAPA...", file=sys.stderr)
    model = load_ecapa(device, args.ecapa)

    enroll_wav = read_kaldi_scp(eer_data / "enroll_orig" / "wav.scp")
    enroll_utt2spk = read_kaldi_scp(eer_data / "enroll_orig" / "utt2spk")
    # Speaker-level enrollment: mean of utt embeddings per speaker
    from collections import defaultdict
    spk2embs = defaultdict(list)
    for utt_id, path in enroll_wav.items():
        if not path or not Path(path).exists():
            continue
        emb = extract_embedding(model, path, device)
        if emb is not None:
            spk = enroll_utt2spk.get(utt_id, "")
            if spk:
                spk2embs[spk].append(emb)
    enroll_emb = {spk: np.mean(embs, axis=0) for spk, embs in spk2embs.items() if embs}
    print("Enrollment speakers:", len(enroll_emb), file=sys.stderr)

    conditions = [args.condition] if args.condition else ["Step2_OO", "Step2_OA"]
    with open(eer_data / "eval_id_metadata.json", "r", encoding="utf-8") as f:
        eval_id_meta = json.load(f)

    for cond in conditions:
        # build_eer_protocol writes test_Step2_OO and test_Step2_OA
        test_dir = eer_data / f"test_{cond}"
        if not (test_dir / "wav.scp").exists():
            print("Skip", cond, "(no wav.scp)", file=sys.stderr)
            continue
        test_wav = read_kaldi_scp(test_dir / "wav.scp")
        if args.limit:
            test_wav = dict(list(test_wav.items())[: args.limit])
        print("Extracting test embeddings", cond, len(test_wav), file=sys.stderr)
        test_emb = {}
        for tid, path in test_wav.items():
            if not path or not Path(path).exists():
                continue
            emb = extract_embedding(model, path, device)
            if emb is not None:
                test_emb[tid] = emb
        print("  Got", len(test_emb), "test embeddings", file=sys.stderr)

        trials_path = eer_data / "trials_Step2"
        trials = []
        with open(trials_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] in test_emb:
                    trials.append((parts[0], parts[1], 1 if parts[2] == "target" else 0))

        enroll_ids = []
        test_ids = []
        labels = []
        for en_id, te_id, label in trials:
            if en_id not in enroll_emb or te_id not in test_emb:
                continue
            enroll_ids.append(en_id)
            test_ids.append(te_id)
            labels.append(label)
        enroll_vecs = np.stack([enroll_emb[e] for e in enroll_ids])
        test_vecs = np.stack([test_emb[t] for t in test_ids])
        scores = cosine_similarity(enroll_vecs, test_vecs).diagonal()
        labels = np.array(labels)
        if args.t_norm:
            # T-norm: per test_id, normalize scores using cohort of impostor scores (standard in ASV; can raise OA EER)
            scores = _apply_tnorm(scores, test_ids, labels)
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            print(cond, "EER: N/A (no pos or neg trials)", file=sys.stderr)
            continue
        eer, th = compute_eer(pos_scores, neg_scores)
        print(cond, "EER: {:.2f}% (thr={:.4f}){}".format(eer * 100, th, " [T-norm]" if args.t_norm else ""), file=sys.stderr)

        # EER by overlap, subset, system, overlap×subset, overlap×system, subset×system, overlap×subset×system
        meta = eval_id_meta
        by_overlap = {}
        by_subset = {}
        by_system = {}
        by_os = {}
        by_ov_sys = {}
        by_subset_system = {}
        by_ov_sub_sys = {}
        for i, (en_id, te_id, label) in enumerate(zip(enroll_ids, test_ids, labels)):
            m = meta.get(te_id, {})
            ov = m.get("overlap")
            sub = m.get("subset")
            sys_name = m.get("system")
            sc = float(scores[i])
            if ov is not None:
                by_overlap.setdefault(ov, []).append((sc, label))
            if sub:
                by_subset.setdefault(sub, []).append((sc, label))
            if sys_name:
                by_system.setdefault(sys_name, []).append((sc, label))
            if ov is not None and sub:
                by_os.setdefault((ov, sub), []).append((sc, label))
            if ov is not None and sys_name:
                by_ov_sys.setdefault((ov, sys_name), []).append((sc, label))
            if sub and sys_name:
                by_subset_system.setdefault((sub, sys_name), []).append((sc, label))
            if ov is not None and sub and sys_name:
                by_ov_sub_sys.setdefault((ov, sub, sys_name), []).append((sc, label))

        def eer_from_pairs(pairs):
            if not pairs:
                return None
            p = np.array([x[0] for x in pairs if x[1] == 1])
            n = np.array([x[0] for x in pairs if x[1] == 0])
            if len(p) == 0 or len(n) == 0:
                return None
            return float(compute_eer(p, n)[0] * 100)

        results = {
            "condition": cond,
            "t_norm": args.t_norm,
            "eer_overall_pct": float(eer * 100),
            "threshold": float(th),
            "n_target": int(labels.sum()),
            "n_nontarget": int((1 - labels).sum()),
            "by_overlap": {str(k): eer_from_pairs(v) for k, v in by_overlap.items()},
            "by_subset": {k: eer_from_pairs(v) for k, v in by_subset.items()},
            "by_system": {k: eer_from_pairs(v) for k, v in by_system.items()},
            "by_overlap_subset": {f"{k[0]}_{k[1]}": eer_from_pairs(v) for k, v in by_os.items()},
            "by_overlap_system": {f"{k[0]}_{k[1]}": eer_from_pairs(v) for k, v in by_ov_sys.items()},
            "by_subset_system": {f"{k[0]}_{k[1]}": eer_from_pairs(v) for k, v in by_subset_system.items()},
            "by_overlap_subset_system": {f"{k[0]}_{k[1]}_{k[2]}": eer_from_pairs(v) for k, v in by_ov_sub_sys.items()},
        }
        out_file = out_dir / f"eer_{cond}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Wrote", out_file, file=sys.stderr)

    # Summary CSV for comparison: scenario (OO/OA), subset (AA/CA/CC/all), system (SELECTION/OHNN/all), eer_pct, input_type
    _write_eer_summary_csv(out_dir, conditions)

    return 0


def _write_eer_summary_csv(out_dir: Path, conditions: list):
    """Write eer_summary.csv for comparison: scenario (OO/OA), subset (AA/CA/CC/all), system (SELECTION/OHNN/all)."""
    import csv
    seen = set()
    rows = []
    # Step2 = single-speaker trial (tse_hat or target_anon)
    input_type = "single_speaker"
    for cond in conditions:
        jpath = out_dir / f"eer_{cond}.json"
        if not jpath.exists():
            continue
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenario = "OO" if "OO" in cond else "OA"
        # Overall
        overall = data.get("eer_overall_pct")
        if overall is not None:
            key = (scenario, "all", "all")
            if key not in seen:
                seen.add(key)
                rows.append({"scenario": scenario, "subset": "all", "system": "all", "eer_pct": overall, "input_type": input_type})
        # By subset (AA, CA, CC)
        for sub in ("AA", "CA", "CC"):
            val = (data.get("by_subset") or {}).get(sub)
            if val is not None:
                key = (scenario, sub, "all")
                if key not in seen:
                    seen.add(key)
                    rows.append({"scenario": scenario, "subset": sub, "system": "all", "eer_pct": val, "input_type": input_type})
        # By system (SELECTION, OHNN)
        for sys_name in ("SELECTION", "OHNN"):
            val = (data.get("by_system") or {}).get(sys_name)
            if val is not None:
                key = (scenario, "all", sys_name)
                if key not in seen:
                    seen.add(key)
                    rows.append({"scenario": scenario, "subset": "all", "system": sys_name, "eer_pct": val, "input_type": input_type})
        # By subset x system (AA+SELECTION, AA+OHNN, ...)
        for key_str, val in (data.get("by_subset_system") or {}).items():
            if val is not None:
                parts = key_str.split("_", 1)
                sub = parts[0] if len(parts) >= 1 else "all"
                sys_name = parts[1] if len(parts) == 2 else "all"
                key = (scenario, sub, sys_name)
                if key not in seen:
                    seen.add(key)
                    rows.append({"scenario": scenario, "subset": sub, "system": sys_name, "eer_pct": val, "input_type": input_type})
    if not rows:
        return
    rows.sort(key=lambda r: (r["scenario"], (0 if r["subset"] == "all" else 1), r["subset"], r["system"]))
    csv_path = out_dir / "eer_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "subset", "system", "eer_pct", "input_type"])
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", csv_path, file=sys.stderr)


def _apply_tnorm(scores: np.ndarray, test_ids: list, labels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """T-norm: per test segment, normalize scores using cohort of impostor (nontarget) scores."""
    scores = np.array(scores, dtype=np.float64)
    test_ids = np.asarray(test_ids)
    out = scores.copy()
    for te_id in np.unique(test_ids):
        mask = test_ids == te_id
        cohort = scores[mask & (labels == 0)]
        if len(cohort) < 2:
            continue
        mean_c = np.mean(cohort)
        std_c = np.std(cohort) + eps
        out[mask] = (scores[mask] - mean_c) / std_c
    return out


def compute_eer(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """Equal Error Rate: threshold where FAR = FRR. Returns (eer, threshold).
    FAR = proportion of nontarget (impostor) trials accepted. FRR = proportion of target trials rejected.
    Accept when score >= threshold (higher = more same-speaker-like).
    """
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    threshs = np.sort(np.unique(all_scores))
    best_eer = 1.0
    best_th = 0.0
    for th in threshs:
        accept = all_scores >= th
        # FAR = P(accept | nontarget) = proportion of impostor trials incorrectly accepted
        far = np.mean(accept[labels == 0]) if (labels == 0).any() else 0.0
        # FRR = P(reject | target) = proportion of true speaker trials incorrectly rejected
        frr = 1.0 - np.mean(accept[labels == 1]) if (labels == 1).any() else 0.0
        eer = (far + frr) / 2.0
        if eer <= best_eer:
            best_eer = eer
            best_th = th
    return float(best_eer), float(best_th)


if __name__ == "__main__":
    sys.exit(main())
