#!/usr/bin/env python3
"""Install the official LIBERO-Plus assets ZIP, stripping its packaging prefix.

The published archive may contain inspire/hdd/.../LIBERO-plus-0/assets/ rather
than a top-level assets/. Only that asset subtree is installed, without changing
any XML/model contents. Interrupted installs can be resumed with the same command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = ROOT / "third_party/libero_plus/libero/libero/assets"


def asset_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Validate one unambiguous assets subtree before extracting any files."""
    prefixes = set()
    selected = []
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if "assets" not in parts:
            continue
        index = parts.index("assets")
        prefixes.add(parts[:index + 1])
        relative = parts[index + 1:]
        if ".." in parts or "\\" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Unsafe asset entry: {info.filename}")
        if relative:
            selected.append((info, Path(*relative)))
    if len(prefixes) != 1 or not selected:
        raise ValueError("Archive must contain exactly one non-empty assets/ subtree")
    if len({str(path) for _, path in selected}) != len(selected):
        raise ValueError("Archive has duplicate asset destinations")
    return selected


def install(archive_path: Path, destination: Path, *, inspect_only: bool = False) -> dict:
    destination = destination.expanduser().resolve()
    with zipfile.ZipFile(archive_path.expanduser()) as archive:
        members = asset_members(archive)
        files = [(info, path) for info, path in members if not info.is_dir()]
        metadata = {"archive": str(archive_path.resolve()), "destination": str(destination),
                    "files": len(files), "uncompressed_bytes": sum(info.file_size for info, _ in files)}
        print(json.dumps(metadata), flush=True)
        if inspect_only:
            return metadata
        destination.mkdir(parents=True, exist_ok=True)
        skipped = 0
        for count, (info, relative) in enumerate(files, start=1):
            target = destination / relative
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Asset path escapes destination: {relative}")
            if target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.install-{os.getpid()}.part")
                try:
                    with archive.open(info) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    # ZipExtFile checks the CRC before the atomic replacement.
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            if count % 5000 == 0 or count == len(files):
                print(f"Assets: {count}/{len(files)} files (already present: {skipped})", flush=True)
        for info, relative in members:
            if info.is_dir():
                target = destination / relative
                if not target.resolve().is_relative_to(destination):
                    raise ValueError(f"Asset directory escapes destination: {relative}")
                target.mkdir(parents=True, exist_ok=True)
        (destination / ".emerge-assets-install.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Assets installed: {destination}", flush=True)
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--inspect", action="store_true", help="Validate ZIP layout without extracting")
    args = parser.parse_args()
    install(args.archive, args.destination, inspect_only=args.inspect)


if __name__ == "__main__":
    main()
