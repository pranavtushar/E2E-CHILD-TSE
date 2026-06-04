#!/usr/bin/env python3
"""
Batch-run TSE (Target Speaker Extraction) on TSA_MIXED.
Input:  TSA_MIXED/{AA,CC,CA}/{ov0,...,ov100}/*.mix.wav + *.ref.wav
Output: TSE_OUT/{AA,CC,CA}/{ovX}/*.tse_hat.wav + *.tse_hat.json
        + summary.json in output root.

Order: CA → CC → AA. Duration from each mix file (or full length).
Multi-GPU: use --num-workers 2 --gpu-ids 0,1 to split work across GPUs.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

SUBSETS_ORDER = ("CA", "CC", "AA")
OVERLAP_DIRS = ("ov0", "ov20", "ov40", "ov60", "ov80", "ov100")
OUTPUT_SUFFIX = "tse_hat"  # *.tse_hat.wav, *.tse_hat.json


def get_mix_duration_sec(mix_path: Path) -> float:
    """Duration of mix wav in seconds."""
    if sf is not None:
        info = sf.info(str(mix_path))
        return float(info.duration)
    import wave
    with wave.open(str(mix_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def get_audio_duration_sec(wav_path: Path) -> float:
    """Duration of any wav in seconds (for validation)."""
    if sf is not None:
        info = sf.info(str(wav_path))
        return float(info.duration)
    import wave
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def collect_mix_items(tsa_root: Path):
    """Yield (subset, ov_dir, mix_path, ref_path, base_name) for each *.mix.wav with matching .ref.wav."""
    for subset in SUBSETS_ORDER:
        for ov in OVERLAP_DIRS:
            folder = tsa_root / subset / ov
            if not folder.is_dir():
                continue
            for mix_path in sorted(folder.glob("*.mix.wav")):
                # Match .mix.wav -> .ref.wav by replacing suffix
                base_name = mix_path.stem.replace(".mix", "")
                ref_path = mix_path.parent / (base_name + ".ref.wav")
                yield subset, ov, mix_path, ref_path, base_name


def write_item_log(log_path: Path, data: dict) -> None:
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Batch TSE on TSA_MIXED → TSE_OUT")
    ap.add_argument("--tsa-root", type=Path, default=None, help="Input TSA_MIXED root")
    ap.add_argument("--out-root", type=Path, default=None, help="Output root (default: tsa_mix/outputs/TSE_OUT)")
    ap.add_argument("--tse-dir", type=Path, default=None, help="tse/ dir with later/inference_server.py")
    ap.add_argument("--ecapa", "--ecapa-checkpoint", dest="ecapa", type=Path, default=None, help="ECAPA embedding_model.ckpt")
    ap.add_argument("--tse-checkpoint", type=Path, default=None, help="Conformer TSE checkpoint file")
    ap.add_argument("--num-workers", type=int, default=1, help="Split work across N GPUs (default 1)")
    ap.add_argument("--gpu-ids", type=str, default="0,1", help="Comma-separated GPU IDs for workers (e.g. 0,1)")
    ap.add_argument("--offset", type=int, default=None, help="Skip first N items (used by multi-GPU worker)")
    ap.add_argument("--worker-id", type=int, default=None, help="Internal: worker index when spawned by manager")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N mixtures (or N per worker)")
    ap.add_argument(
        "--items-manifest",
        type=Path,
        default=None,
        help="jsonl with item_id field; only process those mixtures (smoke subset)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if .tse_hat.wav exists")
    ap.add_argument("--dry-run", action="store_true", help="Only print counts and exit")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    tsa_root = args.tsa_root or (script_dir / "data" / "TSA_MIXED")
    out_root = args.out_root or (script_dir / "outputs" / "TSE_OUT")
    tse_dir = args.tse_dir or (script_dir.parent)

    if not tsa_root.is_dir():
        print(f"TSA root not found: {tsa_root}")
        sys.exit(1)

    # Collect all items (mix + ref); ref missing => skip and count
    total_found = 0
    items_to_process = []
    skipped_no_ref = []
    for subset, ov, mix_path, ref_path, base_name in collect_mix_items(tsa_root):
        total_found += 1
        if not ref_path.exists():
            skipped_no_ref.append(str(mix_path))
            continue
        items_to_process.append((subset, ov, mix_path, ref_path, base_name))

    if args.items_manifest is not None:
        allowed = set()
        with open(args.items_manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = (row.get("item_id") or "").strip()
                if iid:
                    allowed.add(iid)
        items_to_process = [it for it in items_to_process if it[4] in allowed]
        print(f"Filtered by manifest {args.items_manifest}: {len(items_to_process)} items", file=sys.stderr)

    if args.offset is not None:
        items_to_process = items_to_process[args.offset :]
    if args.limit is not None:
        items_to_process = items_to_process[: args.limit]

    print(f"Total .mix.wav found: {total_found}")
    print(f"Skipped (no ref):    {len(skipped_no_ref)}")
    print(f"To process:          {len(items_to_process)}")
    if skipped_no_ref and len(skipped_no_ref) <= 5:
        for p in skipped_no_ref:
            print(f"  no ref: {p}")
    elif skipped_no_ref:
        print(f"  (first no-ref: {skipped_no_ref[0]})")
    print()

    if args.dry_run:
        print("Dry run. Would write to:", out_root)
        return 0

    # Multi-GPU: manager spawns one process per GPU with --offset/--limit
    if args.num_workers >= 2 and args.worker_id is None:
        out_root.mkdir(parents=True, exist_ok=True)
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
        n_workers = min(args.num_workers, len(gpu_ids), len(items_to_process))
        if n_workers <= 0:
            print("No workers or no items.")
            return 0
        n = len(items_to_process)
        chunk_size = (n + n_workers - 1) // n_workers
        chunks = [items_to_process[i : i + chunk_size] for i in range(0, n, chunk_size)]
        cmd_base = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--tsa-root", str(tsa_root),
            "--out-root", str(out_root),
            "--tse-dir", str(tse_dir),
            "--num-workers", "1",
            "--overwrite" if args.overwrite else "",
        ]
        if args.ecapa:
            cmd_base.extend(["--ecapa", str(args.ecapa)])
        if args.tse_checkpoint:
            cmd_base.extend(["--tse-checkpoint", str(args.tse_checkpoint)])
        cmd_base = [x for x in cmd_base if x != ""]
        procs = []
        for i in range(len(chunks)):
            offset = i * chunk_size
            limit = len(chunks[i])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[i % len(gpu_ids)])
            cmd = cmd_base + ["--offset", str(offset), "--limit", str(limit), "--worker-id", str(i)]
            print(f"  Worker {i} (GPU {gpu_ids[i % len(gpu_ids)]}): offset={offset} limit={limit}")
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)
        for p in procs:
            p.wait()
            if p.returncode != 0:
                print("  Worker exit code:", p.returncode)
        # Merge summaries
        all_processed = 0
        all_skipped = 0
        all_failures = []
        for i in range(len(chunks)):
            wj = out_root / f"summary_worker_{i}.json"
            if wj.exists():
                with open(wj) as f:
                    d = json.load(f)
                all_processed += d.get("total_processed", 0)
                all_skipped += d.get("total_skipped_existing", 0)
                all_failures.extend(d.get("failures", []))
        summary = {
            "total_found": total_found,
            "total_processed": all_processed,
            "total_skipped_no_ref": len(skipped_no_ref),
            "total_skipped_existing": all_skipped,
            "total_failures": len(all_failures),
            "failures": all_failures,
            "num_workers": n_workers,
        }
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print()
        print("Summary (merged):")
        print("  Total found:     ", total_found)
        print("  Processed:       ", all_processed)
        print("  Skipped (no ref):", len(skipped_no_ref))
        print("  Skipped (exist): ", all_skipped)
        print("  Failures:        ", len(all_failures))
        return 0 if not all_failures else 1

    # Import TSE: inference_server lives in tse/later/, ecapa in tse/ — need both on path
    later_dir = tse_dir / "later"
    if later_dir.is_dir():
        sys.path.insert(0, str(later_dir))
    sys.path.insert(0, str(tse_dir))  # ecapa.py is in tse/, not tse/later/
    try:
        from inference_server import TSEInference
    except ImportError as e:
        print(f"Could not import TSEInference: {e}")
        sys.exit(1)

    ecapa_path = args.ecapa or (tse_dir / "embedding_model.ckpt")
    tse_ckpt = args.tse_checkpoint or os.environ.get("TSE_CHECKPOINT") or (tse_dir / "libri2talker_libri2vox")
    tse_ckpt = Path(tse_ckpt) if tse_ckpt else (tse_dir / "libri2talker_libri2vox")
    if not ecapa_path.is_file():
        print(f"ECAPA not found: {ecapa_path}")
        sys.exit(1)
    if not tse_ckpt.is_file():
        print(f"TSE checkpoint not found: {tse_ckpt}")
        sys.exit(1)

    gpu_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    if args.worker_id is not None:
        print(f"[Worker {args.worker_id}] GPU CUDA_VISIBLE_DEVICES={gpu_vis} | items {len(items_to_process)}")
    print("Loading TSE model...")
    tse = TSEInference(tse_model_path=str(tse_ckpt), ecapa_model_path=str(ecapa_path))
    print("Output root:", out_root)
    print()

    if sf is None:
        print("soundfile required for writing WAV and duration checks")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped_existing = 0
    failures = []

    for subset, ov, mix_path, ref_path, base_name in items_to_process:
        out_dir = out_root / subset / ov
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / (base_name + f".{OUTPUT_SUFFIX}.wav")
        out_json = out_dir / (base_name + f".{OUTPUT_SUFFIX}.json")

        if out_wav.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        mix_duration_sec = get_mix_duration_sec(mix_path)
        log_entry = {
            "mix_path": str(mix_path),
            "ref_path": str(ref_path),
            "out_path": str(out_wav),
            "subset": subset,
            "overlap": ov,
            "mix_duration_sec": round(mix_duration_sec, 4),
            "status": "ok",
            "error": None,
        }

        try:
            # Process full mix length: pass mix duration (or None if API allows)
            target_audio = tse.separate_speech(
                str(mix_path), str(ref_path), target_duration=mix_duration_sec
            )
            sf.write(str(out_wav), target_audio, 16000)

            # Correctness: output exists and nonzero duration
            if not out_wav.exists():
                raise RuntimeError("Output file was not created")
            out_duration = get_audio_duration_sec(out_wav)
            if out_duration <= 0:
                raise RuntimeError(f"Output duration is zero (got {out_duration})")
            log_entry["out_duration_sec"] = round(out_duration, 4)

            write_item_log(out_json, log_entry)
            processed += 1
            if processed % 50 == 0:
                print(f"  [{subset}/{ov}] {processed} processed")
        except Exception as e:
            log_entry["status"] = "fail"
            log_entry["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip()
            if out_json.parent.exists():
                write_item_log(out_json, log_entry)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                write_item_log(out_json, log_entry)
            failures.append({
                "mix_path": str(mix_path),
                "subset": subset,
                "overlap": ov,
                "error": str(e),
            })
            print("  FAIL:", mix_path.name, "->", e)

    # Summary JSON (per-worker when multi-GPU, else global)
    summary = {
        "total_found": total_found,
        "total_processed": processed,
        "total_skipped_no_ref": len(skipped_no_ref),
        "total_skipped_existing": skipped_existing,
        "total_failures": len(failures),
        "failures": failures,
    }
    if args.worker_id is not None:
        summary_path = out_root / f"summary_worker_{args.worker_id}.json"
    else:
        summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Summary:")
    print("  Total found:     ", total_found)
    print("  Processed:       ", processed)
    print("  Skipped (no ref):", len(skipped_no_ref))
    print("  Skipped (exist): ", skipped_existing)
    print("  Failures:        ", len(failures))
    print("  Summary written: ", summary_path)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
