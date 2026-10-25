"""Verify delivery against actual archive bytes and source inventory changes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

_SPEC = importlib.util.spec_from_file_location(
    "package_delivery", Path(__file__).resolve().parents[1] / "package_delivery.py",
)
assert _SPEC is not None and _SPEC.loader is not None
delivery = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(delivery)


class TestPackageDelivery(unittest.TestCase):
    def fixture(self, directory: str) -> Path:
        root = Path(directory) / "source"
        package = root / "single_parameter_observatory"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('__version__ = "1.1.1"\n', encoding="utf-8")
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            (root / name).write_text(f"fixture {name}\n", encoding="utf-8")
        return root

    def test_every_manifest_member_matches_saved_zip_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            output = Path(directory) / "release.zip"
            count, digest = delivery.build_bundle(root, output)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("MANIFEST.json"))
                self.assertEqual(count, len(manifest["files"]))
                self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {"MANIFEST.json"})
                for name, item in manifest["files"].items():
                    self.assertEqual(archive.read(name), (root / name).read_bytes())
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), item["sha256"])

    def test_new_source_during_packaging_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            output = Path(directory) / "release.zip"
            output.write_bytes(b"previous release")

            def add_source(source_root: Path) -> str:
                (source_root / "single_parameter_observatory" / "new.py").write_text("added = True\n")
                return "1.1.1"

            with patch.object(delivery, "project_version", side_effect=add_source):
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    delivery.build_bundle(root, output)
            self.assertEqual(output.read_bytes(), b"previous release")

    def test_archive_cannot_be_created_inside_a_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            output = root / "single_parameter_observatory" / "release.zip"
            with self.assertRaisesRegex(ValueError, "outside source"):
                delivery.build_bundle(root, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
