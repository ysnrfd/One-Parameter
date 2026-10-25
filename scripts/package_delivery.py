#!/usr/bin/env python3
"""Build a portable source/evidence ZIP and verify every member against SHA-256.

No staging checkout is deleted or rewritten. Missing optional historical
evidence is simply absent; the manifest describes the files actually shipped.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".git",
    ".venv", "bundle", "hfviewer-source",
}
SOURCE_ITEMS = (
    "pyproject.toml", "MANIFEST.in", ".gitignore", "README.md", "LICENSE", "requirements.txt", "AUDIT.md",
    "REPAIR_REPORT.md", "STUDIO.md", "STUDIO_REPORT.md", "NODE_LAB.md", "PLUGIN_SDK.md", "NODE_LAB_REPORT.md",
    "TRANSFORMERS_STUDIO.md", "TRANSFORMERS_AUDIT.md", "TRANSFORMERS_REPORT.md",
    "LLAMA_STUDIO.md", "LLAMA_SOURCE_NOTES.md", "LLAMA_BLOCKS.md",
    "examples", "Launch Studio.cmd", ".github", "single_parameter_observatory", "evidence",
    "scripts", "SPOBS_Forensic_Audit.pdf",
)


def project_version(root: Path) -> str:
    tree = ast.parse((root / "single_parameter_observatory" / "__init__.py").read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets):
                value = ast.literal_eval(statement.value)
                if isinstance(value, str):
                    return value
    raise ValueError("package __version__ is missing or is not a literal string")


def source_files(root: Path) -> list[Path]:
    paths = []
    for name in SOURCE_ITEMS:
        candidate = root / name
        if candidate.is_symlink():
            raise ValueError(f"source roots must not be symlinks: {candidate}")
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            for directory, directories, filenames in os.walk(candidate, followlinks=False):
                directories[:] = sorted(
                    d for d in directories
                    if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")
                    and not (Path(directory) / d).is_symlink()
                )
                for filename in sorted(filenames):
                    path = Path(directory) / filename
                    if not path.is_symlink() and path.suffix not in (".pyc", ".pyo"):
                        paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def build_bundle(root: Path, output: Path) -> tuple[int, str]:
    root = root.resolve(strict=True)
    output = output.resolve()
    required = ("pyproject.toml", "README.md", "LICENSE", "single_parameter_observatory/__init__.py")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    files = source_files(root)
    if output in files:
        raise ValueError("the output ZIP must not overwrite a source or evidence file")
    for name in SOURCE_ITEMS:
        candidate = root / name
        if output == candidate or (candidate.is_dir() and output.is_relative_to(candidate)):
            raise ValueError("the output ZIP must be outside source and evidence directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str | int]] = {}
    with tempfile.TemporaryDirectory(prefix="spobs-package-", dir=output.parent) as temporary:
        temporary_zip = Path(temporary) / "bundle.zip"
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                data = path.read_bytes()
                archive.writestr(name, data)
                manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            archive.writestr("MANIFEST.json", json.dumps({
                "version": project_version(root),
                "algorithm": "sha256",
                "scope": "Every ZIP member except MANIFEST.json itself.",
                "files": manifest,
            }, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(temporary_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
            if set(archive.namelist()) != set(manifest) | {"MANIFEST.json"}:
                raise ValueError("ZIP inventory does not match the manifest")
            for name, expected in manifest.items():
                data = archive.read(name)
                if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    raise ValueError(f"ZIP member failed SHA-256 verification: {name}")
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected["sha256"]:
                    raise ValueError(f"source changed during packaging: {name}")
            final_inventory = {path.relative_to(root).as_posix() for path in source_files(root)}
            if final_inventory != set(manifest):
                raise ValueError("source inventory changed during packaging")
        os.replace(temporary_zip, output)
    return len(manifest), hashlib.sha256(output.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    output = args.output or root / "download" / f"single_parameter_observatory_{project_version(root)}.zip"
    count, digest = build_bundle(root, output)
    print(f"ZIP: {output.resolve()}")
    print(f"Verified source/evidence files: {count}")
    print(f"ZIP SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
