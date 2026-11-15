#!/usr/bin/env python3
"""Convert .cbr comic archives to .cbz, in place, verifying every file.

Most ".cbr" files are really ZIPs (the scripts in this repo used to write
them that way): those are only renamed after the ZIP passes an integrity
test. Real RAR-based ones are unpacked with `unrar` (or `7z`), stored into a
new ZIP (images are already compressed, so no recompression), and the new
page count must equal the RAR's before the original is moved aside.

  cbr_to_cbz.py <folder> --originals-to <dir>   # RARs are moved to <dir>/<same relative path>
  cbr_to_cbz.py <folder> --keep-originals       # RARs stay next to the new .cbz
  cbr_to_cbz.py <folder> --dry-run              # only report what would happen

Example:
  venv/bin/python comic-archive/scripts/cbr_to_cbz.py ~/Comics --originals-to ~/cbr-originals
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ZIP_MAGIC = b"PK\x03\x04"
RAR_MAGIC = b"Rar!"


@dataclass(frozen=True)
class Options:
    root: Path
    originals_to: Path | None
    dry_run: bool
    work_dir: Path | None


def find_cbr(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".cbr")


def archive_kind(path: Path) -> str:
    """Return 'zip', 'rar' or 'unknown' from the file's first bytes."""
    with path.open("rb") as handle:
        head = handle.read(4)
    if head == ZIP_MAGIC:
        return "zip"
    return "rar" if head == RAR_MAGIC else "unknown"


def find_extractor() -> list[str]:
    for tool, args in (("unrar", ["x", "-y", "-inul"]), ("7z", ["x", "-y", "-bso0", "-bsp0"])):
        if shutil.which(tool):
            return [tool, *args]
    raise SystemExit("ERROR: RAR-based .cbr files need `unrar` or `7z` (p7zip-full) on PATH.")


def target_for(source: Path) -> Path:
    return source.with_suffix(".cbz")


def convert_zip(source: Path, target: Path) -> str:
    with zipfile.ZipFile(source) as archive:
        bad_member = archive.testzip()
        pages = len(archive.namelist())
    if bad_member:
        raise ValueError(f"corrupt member {bad_member!r} in {source}")
    source.rename(target)
    return f"renamed ({pages} entries)"


def extract_rar(source: Path, destination: Path, extractor: list[str]) -> None:
    dest_arg = f"{destination}/" if extractor[0] == "unrar" else f"-o{destination}"
    command = [*extractor, str(source), dest_arg] if extractor[0] == "unrar" else [*extractor, dest_arg, str(source)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"{extractor[0]} failed on {source} (exit {result.returncode}): {result.stderr.strip()}")


def zip_folder(folder: Path, target: Path) -> int:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.relative_to(folder).as_posix())
    return len(files)


def convert_rar(source: Path, target: Path, options: Options, extractor: list[str]) -> str:
    part = target.with_name(target.name + ".part")
    with tempfile.TemporaryDirectory(dir=options.work_dir, prefix="cbr2cbz-") as scratch:
        extract_rar(source, Path(scratch), extractor)
        pages = zip_folder(Path(scratch), part)
    with zipfile.ZipFile(part) as check:
        if check.testzip() or len(check.namelist()) != pages or pages == 0:
            part.unlink()
            raise ValueError(f"verification failed for {source}: wrote {pages} pages")
    part.rename(target)
    return f"repacked ({pages} files)"


def set_original_aside(source: Path, options: Options) -> None:
    if options.originals_to is None:
        return
    destination = options.originals_to / source.relative_to(options.root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def convert_one(source: Path, options: Options, extractor: list[str] | None) -> str:
    target = target_for(source)
    kind = archive_kind(source)
    if target.exists():
        return f"SKIP {source}: {target.name} already exists"
    if kind == "unknown":
        return f"SKIP {source}: neither ZIP nor RAR"
    if options.dry_run:
        return f"would convert ({kind}) {source}"
    try:
        if kind == "zip":
            return f"{convert_zip(source, target)} {source}"
        message = convert_rar(source, target, options, extractor or find_extractor())
        set_original_aside(source, options)
        return f"{message} {source}"
    except (ValueError, zipfile.BadZipFile, OSError) as error:
        return f"FAIL {source}: {error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .cbr archives to .cbz in place.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--originals-to", type=Path, help="Move converted RAR originals here, keeping their relative paths.")
    parser.add_argument("--keep-originals", action="store_true", help="Leave converted RAR originals where they are.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, help="Scratch folder for unpacking (default: system temp).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.originals_to or args.keep_originals or args.dry_run):
        raise SystemExit("ERROR: pass --originals-to <dir>, --keep-originals or --dry-run.")
    options = Options(args.folder.resolve(), args.originals_to, args.dry_run, args.work_dir)
    sources = find_cbr(options.root)
    extractor = find_extractor() if any(archive_kind(p) == "rar" for p in sources) else None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda p: convert_one(p, options, extractor), sources))
    for line in results:
        print(line)
    failures = [r for r in results if r.startswith("FAIL")]
    print(f"{len(sources)} .cbr found, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
