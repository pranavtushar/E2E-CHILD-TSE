"""Resolve audio/manifest paths inside the E2E bundle (data/ by default)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def _data_subdir() -> str:
    return os.environ.get("DATA_SUBDIR", "data").strip().rstrip("/") or "data"


def source_data_dir(root: Path) -> Path:
    return root.resolve() / _data_subdir()


def libri_audio_root(root: Path) -> Path:
    return source_data_dir(root) / "libri" / "test-clean-normalised"


def myst_audio_root(root: Path) -> Path:
    return source_data_dir(root) / "myst" / "myST_test_filtered"


def myst_manifest(root: Path) -> Path:
    return source_data_dir(root) / "manifests" / "myst_test_manifest.jsonl"


def libri_manifest(root: Path) -> Path:
    return source_data_dir(root) / "manifests" / "librispeech_test_clean_manifest.jsonl"


def child_reference_root(root: Path) -> Path:
    sd = source_data_dir(root)
    for name in ("development", "ai_kids_processed"):
        p = sd / "myst" / name
        if p.is_dir() and any(p.iterdir()):
            return p
    return myst_audio_root(root)


def adult_reference_root(root: Path) -> Path:
    return libri_audio_root(root)


def _data_prefix() -> str:
    return f"{_data_subdir()}/"


def to_bundle_path(path: Path, root: Path) -> str:
    p = path.resolve()
    root = root.resolve()
    sd = source_data_dir(root)
    prefix = _data_prefix()
    for anchor in (sd, root):
        try:
            rel = p.relative_to(anchor)
            rel_s = rel.as_posix()
            if anchor == sd:
                return prefix + rel_s
            if rel_s.startswith(prefix) or rel_s.startswith("00_source_data/"):
                return rel_s.replace("00_source_data/", prefix, 1)
            return rel_s
        except ValueError:
            continue
    return p.as_posix()


_LEGACY_E2E_MARKERS = (
    "/app/app/E2E_CHILD_TSE/",
    "/app/E2E_CHILD_TSE/",
    "/app/app/E2E_CHILD_TSE_2/",
    "/app/E2E_CHILD_TSE_2/",
)


def _rewrite_e2e_bundle_path(s: str, root: Optional[Path]) -> str:
    if not root:
        return s
    root_s = str(root.resolve()).replace("\\", "/").rstrip("/")
    for marker in _LEGACY_E2E_MARKERS:
        if marker in s:
            return f"{root_s}/{s.split(marker, 1)[1]}"
    return s


def _normalize_legacy_marker(s: str) -> str:
    """Accept paths stored as data/ or 00_source_data/."""
    for old, new in (
        ("00_source_data/", _data_prefix()),
        ("/app/TSE-beta/00_source_data/", _data_prefix()),
        ("/app/TSE-alpha/00_source_data/", _data_prefix()),
    ):
        if old in s:
            s = s.replace(old, new)
    return s


def rewrite_legacy_path(path_str: str, root: Optional[Path]) -> str:
    if not path_str or not root:
        return path_str
    s = _rewrite_e2e_bundle_path(
        _normalize_legacy_marker(path_str.strip().replace("\\", "/")), root
    )
    root_s = str(root.resolve()).replace("\\", "/").rstrip("/")
    if s.startswith(f"{root_s}/"):
        return s
    prefix = _data_prefix()
    if s.startswith(prefix):
        return f"{root_s}/{s}"
    if "test-clean-normalised/" in s:
        tail = s.split("test-clean-normalised/", 1)[1]
        return f"{root_s}/{prefix}libri/test-clean-normalised/{tail}"
    if "myST_test_filtered/" in s:
        tail = s.split("myST_test_filtered/", 1)[1]
        return f"{root_s}/{prefix}myst/myST_test_filtered/{tail}"
    if s.startswith("/app/") and not s.startswith(root_s):
        tail = s[5:]
        if tail.startswith("00_source_data/") or tail.startswith("data/"):
            return f"{root_s}/{tail.replace('00_source_data/', prefix)}"
    return s


def resolve_bundle_path(path_str: str, root: Path) -> Path:
    s = _rewrite_e2e_bundle_path(
        _normalize_legacy_marker((path_str or "").strip().replace("\\", "/")), root
    )
    root = root.resolve()
    prefix = _data_prefix()
    candidates: List[Path] = []
    if s.startswith(prefix) or s.startswith("00_source_data/"):
        candidates.append(root / s.replace("00_source_data/", prefix, 1))
    p = Path(s)
    if p.is_file():
        return p.resolve()
    rewritten = rewrite_legacy_path(s, root)
    candidates.append(Path(rewritten))
    if "test-clean-normalised/" in s:
        tail = s.split("test-clean-normalised/", 1)[1]
        candidates.append(source_data_dir(root) / "libri" / "test-clean-normalised" / tail)
    if "myST_test_filtered/" in s:
        tail = s.split("myST_test_filtered/", 1)[1]
        candidates.append(source_data_dir(root) / "myst" / "myST_test_filtered" / tail)
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return Path(rewritten)
