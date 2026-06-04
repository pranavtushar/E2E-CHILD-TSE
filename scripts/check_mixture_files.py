#!/usr/bin/env python3
"""Check mixture WAVs and resolvable source paths in JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "pipeline"))
from path_utils import resolve_bundle_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-root", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--check-ref", action="store_true")
    ap.add_argument("--list-missing", type=int, default=10)
    args = ap.parse_args()

    root = (args.base_dir or _SCRIPT_DIR.parent).resolve()
    mix_root = args.mix_root.resolve()
    json_files = sorted(mix_root.rglob("*.json"))
    n = len(json_files)
    ok_mix = ok_tgt = ok_nt = ok_ref = ok_s1 = ok_s2 = 0
    missing: list[str] = []

    for jpath in json_files:
        stem = jpath.stem
        parent = jpath.parent
        for name, cnt in (
            (f"{stem}.mix.wav", "mix"),
            (f"{stem}.target.wav", "tgt"),
            (f"{stem}.nontarget.wav", "nt"),
        ):
            if (parent / name).is_file():
                if cnt == "mix":
                    ok_mix += 1
                elif cnt == "tgt":
                    ok_tgt += 1
                else:
                    ok_nt += 1
            else:
                missing.append(str(parent / name))
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        for key, ok_attr in (("utt_s1_mix_path", "s1"), ("utt_s2_mix_path", "s2")):
            p = resolve_bundle_path(meta.get(key, ""), root)
            if p.is_file():
                if ok_attr == "s1":
                    ok_s1 += 1
                else:
                    ok_s2 += 1
            else:
                missing.append(f"{jpath.name}: {key}")
        if args.check_ref:
            if (parent / f"{stem}.ref.wav").is_file():
                ok_ref += 1
            p = resolve_bundle_path(meta.get("utt_s1_ref_path", ""), root)
            if p.is_file():
                pass

    print(f"JSON: {n}  mix: {ok_mix}/{n}  target: {ok_tgt}/{n}  nontarget: {ok_nt}/{n}")
    print(f"  s1_src: {ok_s1}/{n}  s2_src: {ok_s2}/{n}")
    all_ok = ok_mix == n == ok_tgt == ok_nt == ok_s1 == ok_s2
    if not all_ok and missing:
        for m in missing[: args.list_missing]:
            print(f"  missing: {m}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
