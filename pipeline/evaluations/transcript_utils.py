"""
Load MyST and LibriSpeech test manifests and resolve utterance path -> (utterance_id, transcript).
Used by build_eval_manifest and validate_eval_transcripts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from path_utils import rewrite_legacy_path
except ImportError:
    import sys
    from pathlib import Path as _Path

    _pipe = _Path(__file__).resolve().parents[1]
    if str(_pipe) not in sys.path:
        sys.path.insert(0, str(_pipe))
    from path_utils import rewrite_legacy_path


def _add_path_variants(
    d: Dict[str, dict],
    path_str: Optional[str],
    value: dict,
    base_dir: Optional[Path],
) -> None:
    """Add value to dict under path_str and under base_dir variant for robust lookup."""
    if not path_str or not path_str.strip():
        return
    path_str = path_str.strip().replace("\\", "/")
    d[path_str] = value
    if base_dir:
        base_str = str(base_dir.resolve()).replace("\\", "/").rstrip("/")
        bundled = rewrite_legacy_path(path_str, base_dir.resolve())
        if bundled != path_str:
            d[bundled] = value
        if path_str.startswith("/app/"):
            alt = path_str.replace("/app/", base_str + "/", 1)
            d[alt] = value
        elif path_str.startswith("00_source_data/"):
            d[f"{base_str}/{path_str}"] = value
        elif path_str.startswith(base_str + "/"):
            d[path_str[len(base_str) + 1 :]] = value
    # Do not key by basename: different speakers can share the same filename and would collide.


def load_myst_manifest(manifest_path: Path, base_dir: Optional[Path] = None) -> Dict[str, dict]:
    """
    Load myst_test_manifest.jsonl. Returns dict path -> {utterance_id, transcript}.
    Keys: normalized_audio_path (when set), and path variants with base_dir.
    """
    out: Dict[str, dict] = {}
    if not manifest_path.exists():
        return out
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            utt_id = (row.get("utterance_id") or "").strip()
            transcript = (row.get("transcript") or "").strip().replace("\t", " ").replace("\n", " ")
            val = {"utterance_id": utt_id, "transcript": transcript}
            path_norm = row.get("normalized_audio_path")
            if path_norm:
                _add_path_variants(out, path_norm, val, base_dir)
            # Fallback: home_audio_path -> we can't match to myST_test_filtered directly; skip
    return out


def load_libri_manifest(manifest_path: Path, base_dir: Optional[Path] = None) -> Dict[str, dict]:
    """
    Load librispeech_test_clean_manifest.jsonl. Returns dict path -> {utterance_id, transcript}.
    Keys: normalised_audio_path and path variants with base_dir.
    """
    out: Dict[str, dict] = {}
    if not manifest_path.exists():
        return out
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            utt_id = (row.get("utterance_id") or "").strip()
            transcript = (row.get("transcript") or "").strip().replace("\t", " ").replace("\n", " ")
            val = {"utterance_id": utt_id, "transcript": transcript}
            path_norm = row.get("normalised_audio_path") or row.get("normalized_audio_path")
            if path_norm:
                _add_path_variants(out, path_norm, val, base_dir)
    return out


def is_myst_path(path_str: str) -> bool:
    return "myst" in path_str.lower() or "myST_test_filtered" in path_str or "myst_test" in path_str.lower()


def _try_lookup(lookup: Dict[str, dict], path_str: str, base_dir: Optional[Path]) -> Optional[dict]:
    path_str = path_str.strip().replace("\\", "/")
    entry = lookup.get(path_str)
    if entry:
        return entry
    if base_dir:
        base_str = str(base_dir).replace("\\", "/").rstrip("/")
        if path_str.startswith("/app/"):
            entry = lookup.get(path_str.replace("/app/", base_str + "/", 1))
        elif path_str.startswith(base_str + "/"):
            entry = lookup.get(path_str.replace(base_str + "/", "/app/", 1))
    return entry


def lookup_utterance(
    path_str: str,
    myst_lookup: Dict[str, dict],
    libri_lookup: Dict[str, dict],
    base_dir: Optional[Path],
) -> Tuple[str, str]:
    """Return (utterance_id, transcript) for the given utterance path. Empty strings if not found."""
    path_str = path_str.strip().replace("\\", "/")
    if is_myst_path(path_str):
        entry = _try_lookup(myst_lookup, path_str, base_dir)
    else:
        entry = _try_lookup(libri_lookup, path_str, base_dir)
    if entry:
        return (entry.get("utterance_id") or "", entry.get("transcript") or "")
    return ("", "")
