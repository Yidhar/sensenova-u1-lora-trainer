"""Dataset format conversion utilities.

Three subcommands:

    unify-txt    — fold legacy `<id>.txt` + `<id>.think.txt` into a single
                   `<id>.txt` with the `---think---` delimiter, in place.
                   Use after a successful training run when you want to
                   simplify the on-disk layout. Originals at `<id>.old.txt`
                   are left untouched.

    pack-arrow   — pack a folder dataset into a parquet shard for the
                   ArrowT2IDataset path. Schema written:
                       sample_id, caption, think (nullable), image (bytes)
                   Suitable for 1M-image scaling — one shard per ~10-50k images.

    inspect-arrow — print row count + first 3 rows of a parquet shard for
                   sanity checking.

Usage:
    python -m train_u1.scripts.dataset_tools unify-txt dataset/Hayateluc
    python -m train_u1.scripts.dataset_tools pack-arrow dataset/Hayateluc \
        --out artifacts/hayateluc.parquet
    python -m train_u1.scripts.dataset_tools inspect-arrow artifacts/hayateluc.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


THINK_MARKER = "---think---"


def _read(p: Path) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


def cmd_unify_txt(folder: Path, *, dry_run: bool = False) -> int:
    """Fold `<id>.txt` + `<id>.think.txt` → unified `<id>.txt` (in place).

    Skips ids whose `<id>.txt` already contains the `---think---` marker
    (idempotent). Skips ids without a corresponding `<id>.think.txt`.
    """
    if not folder.is_dir():
        print(f"folder not found: {folder}", file=sys.stderr); return 2

    n_unified = 0
    n_skipped_already = 0
    n_skipped_no_think = 0
    for txt in sorted(folder.glob("*.txt")):
        if txt.name.endswith(".old.txt") or txt.name.endswith(".think.txt"):
            continue
        body = _read(txt)
        if THINK_MARKER in body.lower():
            n_skipped_already += 1
            continue
        think_path = txt.with_suffix(".think.txt")
        if not think_path.is_file():
            n_skipped_no_think += 1
            continue
        think_body = _read(think_path)
        unified = f"{body}\n\n{THINK_MARKER}\n{think_body}\n"
        if dry_run:
            print(f"[dry-run] would unify {txt.name} ({len(body)} + {len(think_body)} chars)")
        else:
            txt.write_text(unified, encoding="utf-8")
            think_path.unlink()
        n_unified += 1
    print(f"unified={n_unified} already-unified={n_skipped_already} no-think={n_skipped_no_think}")
    return 0


def cmd_pack_arrow(
    folder: Path, out: Path, *,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    inline_bytes: bool = True,
) -> int:
    """Pack a paired-folder dataset into a single parquet shard."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow required (pip install pyarrow)", file=sys.stderr); return 2

    if not folder.is_dir():
        print(f"folder not found: {folder}", file=sys.stderr); return 2

    sample_ids: list[str] = []
    captions: list[str] = []
    thinks: list[str | None] = []
    image_blobs: list[bytes | None] = []
    image_paths: list[str | None] = []

    pairs: list[tuple[Path, Path]] = []
    for ext in image_extensions:
        for img in sorted(folder.glob(f"*{ext}")):
            txt = img.with_suffix(".txt")
            if txt.is_file():
                pairs.append((img, txt))
    if not pairs:
        print(f"no paired (image, .txt) pairs in {folder}", file=sys.stderr); return 2

    for img_path, txt_path in pairs:
        # Use the same parser used by PairedFolderT2IDataset
        from train_u1.data.datasets import parse_caption_and_think
        raw = _read(txt_path)
        caption, think = parse_caption_and_think(raw)
        # Legacy fallback for unmigrated data
        if think is None:
            tt = txt_path.with_suffix(".think.txt")
            if tt.is_file():
                think = _read(tt) or None
        sample_ids.append(img_path.stem)
        captions.append(caption)
        thinks.append(think)
        if inline_bytes:
            with open(img_path, "rb") as f:
                image_blobs.append(f.read())
            image_paths.append(None)
        else:
            image_blobs.append(None)
            image_paths.append(str(img_path.resolve()))

    arrays = {
        "sample_id": pa.array(sample_ids, type=pa.string()),
        "caption": pa.array(captions, type=pa.string()),
        "think": pa.array(thinks, type=pa.string()),
    }
    if inline_bytes:
        arrays["image"] = pa.array(image_blobs, type=pa.binary())
    else:
        arrays["image_path"] = pa.array(image_paths, type=pa.string())

    table = pa.table(arrays)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out), compression="zstd")
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {table.num_rows} rows → {out}  ({size_mb:.1f} MB, "
          f"inline_bytes={inline_bytes})")
    return 0


def cmd_inspect_arrow(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow required", file=sys.stderr); return 2

    table = pq.read_table(str(path))
    print(f"path: {path}")
    print(f"rows: {table.num_rows}")
    print(f"schema:")
    for f in table.schema:
        print(f"  {f.name:<14s}  {f.type}")
    print(f"---first 3 rows (truncated)---")
    for i in range(min(3, table.num_rows)):
        row = table.slice(i, 1).to_pydict()
        sid = row["sample_id"][0]
        cap = row["caption"][0][:80] + "..." if len(row["caption"][0]) > 80 else row["caption"][0]
        thk = row.get("think", [None])[0]
        thk_str = (thk[:80] + "...") if thk and len(thk) > 80 else (thk or "<none>")
        if "image" in table.column_names:
            sz = len(row["image"][0]) if row["image"][0] else 0
            print(f"  [{i}] id={sid}  cap={cap!r}  image={sz/1e3:.1f} KB")
        else:
            print(f"  [{i}] id={sid}  cap={cap!r}  image_path={row['image_path'][0]}")
        print(f"        think: {thk_str!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Dataset format conversion utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_unify = sub.add_parser("unify-txt", help="fold .txt + .think.txt into one .txt")
    p_unify.add_argument("folder", type=Path)
    p_unify.add_argument("--dry-run", action="store_true")

    p_pack = sub.add_parser("pack-arrow", help="pack folder → parquet shard")
    p_pack.add_argument("folder", type=Path)
    p_pack.add_argument("--out", type=Path, required=True)
    p_pack.add_argument("--paths-only", action="store_true",
                        help="store image_path instead of inline image bytes "
                             "(smaller parquet, but reads still hit the filesystem)")

    p_inspect = sub.add_parser("inspect-arrow", help="print row count + sample rows")
    p_inspect.add_argument("path", type=Path)

    args = ap.parse_args()
    if args.cmd == "unify-txt":
        return cmd_unify_txt(args.folder, dry_run=args.dry_run)
    if args.cmd == "pack-arrow":
        return cmd_pack_arrow(args.folder, args.out, inline_bytes=not args.paths_only)
    if args.cmd == "inspect-arrow":
        return cmd_inspect_arrow(args.path)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
