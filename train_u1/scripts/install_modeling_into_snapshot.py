"""Install pinned modeling .py from u1_src into the HF snapshot directory.

Why: HF model repo `sensenova/SenseNova-U1-8B-MoT` ships only weights /
tokenizer / config; the auto_map points to `modeling_neo_chat.py` etc.
which live in the GitHub repo `OpenSenseNova/SenseNova-U1`. Loading via
`trust_remote_code=True` requires those .py to sit alongside `config.json`.

Safety:
- Source files are taken from a single pinned commit `df86ca9` (matching
  the research report's evidence anchor).
- Each source file's sha256 is verified against `upstream_pinned_sha256.json`
  *before* it is placed into the snapshot — drift aborts the install.
- We `cp` (not symlink) so the artifact is self-contained and the snapshot
  is reproducible from this script + the pinned hash list.
- Each destination is checked to be inside the configured snapshot dir
  (no path traversal).

Usage:
    .venv/bin/python -m train_u1.scripts.install_modeling_into_snapshot \
        --src u1_src/src/sensenova_u1/models/neo_unify \
        --snapshot hf_cache/models--sensenova--SenseNova-U1-8B-MoT/snapshots/749fb605230f216d7a7cc0202bfb28369805466b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

DEFAULT_SRC = Path("u1_src/src/sensenova_u1/models/neo_unify")
DEFAULT_SNAP = Path(
    "hf_cache/models--sensenova--SenseNova-U1-8B-MoT/snapshots/"
    "749fb605230f216d7a7cc0202bfb28369805466b"
)
PINNED_HASHES = Path(__file__).resolve().parents[1] / "upstream_pinned_sha256.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAP)
    ap.add_argument(
        "--hashes", type=Path, default=PINNED_HASHES,
        help="JSON file with pinned commit + per-file sha256 expectations",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src.resolve()
    snap = args.snapshot.resolve()
    if not src.is_dir():
        print(f"[err] src dir not found: {src}", file=sys.stderr)
        return 2
    if not snap.is_dir():
        print(f"[err] snapshot dir not found: {snap}", file=sys.stderr)
        return 2
    if not (snap / "config.json").exists():
        print(f"[err] not a HF snapshot (no config.json): {snap}", file=sys.stderr)
        return 2

    with open(args.hashes) as f:
        pinned = json.load(f)
    expected_files: dict[str, str] = pinned["files"]

    print(f"[install] commit pin: {pinned['commit']}")
    print(f"[install] src      : {src}")
    print(f"[install] snapshot : {snap}")

    plan = []
    for fname, expected_sha in expected_files.items():
        src_path = (src / fname).resolve()
        if not src_path.is_file():
            print(f"[err] missing source file {src_path}", file=sys.stderr)
            return 2
        got_sha = _sha256(src_path)
        if got_sha != expected_sha:
            print(
                f"[err] sha256 mismatch for {fname}\n"
                f"      expected: {expected_sha}\n"
                f"      got     : {got_sha}",
                file=sys.stderr,
            )
            return 3
        # destination must be inside snapshot dir
        dst_path = (snap / fname).resolve()
        if snap not in dst_path.parents and dst_path.parent != snap:
            print(f"[err] destination escapes snapshot: {dst_path}", file=sys.stderr)
            return 4
        plan.append((src_path, dst_path, got_sha))

    print(f"\n[install] verified {len(plan)} files; planning to install:")
    for src_path, dst_path, sha in plan:
        marker = "[exists]" if dst_path.exists() else "[new]   "
        print(f"  {marker} {dst_path.name}  sha={sha[:12]}…")
    if args.dry_run:
        print("\n[install] --dry-run: no changes made")
        return 0

    for src_path, dst_path, _sha in plan:
        # If destination exists with same sha, skip; otherwise overwrite.
        if dst_path.exists() and _sha256(dst_path) == _sha:
            print(f"[skip] {dst_path.name}: identical")
            continue
        shutil.copy2(src_path, dst_path)
        # post-copy verify
        if _sha256(dst_path) != _sha:
            print(f"[err] post-copy sha mismatch for {dst_path}", file=sys.stderr)
            return 5
        print(f"[copy] {dst_path.name}")

    print("\n[install] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
