"""Inventory a pinned official Transformers source distribution without importing it.

This is source acquisition, hash verification and static AST coverage. It does not
execute upstream code, download model weights, or certify every upstream feature.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


CRITICAL = (
    "src/transformers/__init__.py",
    "src/transformers/configuration_utils.py",
    "src/transformers/modeling_utils.py",
    "src/transformers/modeling_outputs.py",
    "src/transformers/models/auto/configuration_auto.py",
    "src/transformers/models/auto/auto_factory.py",
    "src/transformers/models/auto/auto_mappings.py",
    "src/transformers/models/auto/modeling_auto.py",
    "src/transformers/models/auto/tokenization_auto.py",
    "src/transformers/trainer.py",
    "src/transformers/training_args.py",
    "src/transformers/data/data_collator.py",
    "src/transformers/tokenization_utils_base.py",
    "src/transformers/tokenization_utils_tokenizers.py",
    "src/transformers/pipelines/__init__.py",
    "src/transformers/pipelines/base.py",
    "src/transformers/pipelines/text_classification.py",
    "src/transformers/pipelines/text_generation.py",
    "src/transformers/generation/configuration_utils.py",
    "src/transformers/generation/utils.py",
    "src/transformers/models/llama/configuration_llama.py",
    "src/transformers/models/llama/modeling_llama.py",
    "src/transformers/models/gpt2/configuration_gpt2.py",
    "src/transformers/models/gpt2/modeling_gpt2.py",
    "src/transformers/models/distilbert/configuration_distilbert.py",
    "src/transformers/models/distilbert/modeling_distilbert.py",
    "src/transformers/utils/hub.py",
    "src/transformers/safetensors_conversion.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, maximum: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ParameterStudio-source-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"Remote response exceeds {maximum} bytes: {url}")
    return data


def declarations(tree: ast.AST) -> list[dict[str, object]]:
    """Retain class/function signatures, not only an optimistic public count."""
    records: list[dict[str, object]] = []

    def visit(node: ast.AST, prefix: str = "") -> None:
        children_prefix = prefix
        if isinstance(node, ast.ClassDef):
            name = prefix + node.name
            records.append({"kind": "class", "name": name, "line": node.lineno,
                            "end_line": node.end_lineno,
                            "bases": [ast.unparse(base) for base in node.bases],
                            "public_name": not node.name.startswith("_")})
            children_prefix = name + "."
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            records.append({"kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                            "name": name, "line": node.lineno, "end_line": node.end_lineno,
                            "arguments": ast.unparse(node.args),
                            "returns": ast.unparse(node.returns) if node.returns else None,
                            "public_name": not node.name.startswith("_")})
            children_prefix = name + ".<locals>."
        for child in ast.iter_child_nodes(node):
            visit(child, children_prefix)

    visit(tree)
    expected = sum(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    if len(records) != expected:
        raise ValueError("Declaration traversal did not cover the complete AST")
    return records


def static_catalogs(tree: ast.AST) -> dict[str, object]:
    """Read literal catalogs only; dynamic expressions stay explicitly unresolved."""
    result: dict[str, object] = {}
    value: ast.expr | None
    literal: ast.expr | None
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
            value = node.value
        else:
            continue
        for name in names:
            if not (name.endswith("_MAPPING_NAMES") or name in {"SUPPORTED_TASKS", "TASK_ALIASES"}):
                continue
            literal = value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "OrderedDict":
                literal = value.args[0] if value.args else None
            try:
                decoded = ast.literal_eval(literal) if literal else None
                if isinstance(decoded, (list, tuple)):
                    decoded = dict(decoded)
                if isinstance(decoded, dict):
                    result[name] = {"count": len(decoded), "entries": decoded, "line": node.lineno}
                    continue
            except (ValueError, TypeError, SyntaxError):
                pass
            if isinstance(literal, ast.Dict):
                keys = [key.value for key in literal.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                result[name] = {"count": len(keys), "keys": keys, "line": node.lineno,
                                "scope": "literal string keys only; values can reference live classes"}
            else:
                result[name] = {"line": node.lineno, "scope": "dynamic expression not evaluated"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="5.20.2")
    parser.add_argument("--output", type=Path, default=Path("evidence/transformers_20261023"))
    args = parser.parse_args()
    if not all(part.isdecimal() for part in args.version.split(".")):
        parser.error("version must be an exact stable numeric release")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://pypi.org/pypi/transformers/{args.version}/json"
    metadata_bytes = fetch(metadata_url, 8 * 1024 * 1024)
    metadata = json.loads(metadata_bytes)
    if metadata["info"]["version"] != args.version:
        raise ValueError("PyPI returned a different version")
    distributions = [entry for entry in metadata["urls"] if entry["packagetype"] == "sdist"]
    if len(distributions) != 1:
        raise ValueError("Expected exactly one official source distribution")
    distribution = distributions[0]
    archive = output / distribution["filename"]
    if archive.exists():
        archive_bytes = archive.read_bytes()
    else:
        archive_bytes = fetch(distribution["url"], 100 * 1024 * 1024)
    actual_hash = sha256(archive_bytes)
    if actual_hash != distribution["digests"]["sha256"] or len(archive_bytes) != distribution["size"]:
        raise ValueError("Official source archive failed SHA-256/length verification")
    archive.write_bytes(archive_bytes)
    (output / "hf_audit_pypi.json").write_bytes(metadata_bytes)
    prefix = f"transformers-{args.version}/"
    inventory: list[dict[str, object]] = []
    api: dict[str, object] = {}
    failures: list[dict[str, str]] = []
    families: set[str] = set()
    categories: Counter[str] = Counter()
    source_bytes = source_lines = python_files = declarations_total = public_total = 0
    critical_found: dict[str, dict[str, object]] = {}
    catalogs: dict[str, object] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if member.isdir() and member.name.rstrip("/") == prefix.rstrip("/"):
                continue
            if path.is_absolute() or ".." in path.parts or not member.name.startswith(prefix):
                raise ValueError(f"Unsafe/unexpected archive path: {member.name}")
            relative = member.name[len(prefix):]
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unexpected link/special archive entry: {relative}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Could not read archive member: {relative}")
            data = stream.read()
            inventory.append({"path": relative, "bytes": len(data), "sha256": sha256(data)})
            categories[relative.split("/", 1)[0]] += 1
            if not (relative.startswith("src/transformers/") and relative.endswith(".py")):
                continue
            python_files += 1
            source_bytes += len(data)
            source_lines += len(data.splitlines())
            if relative.startswith("src/transformers/models/"):
                pieces = relative.split("/")
                if len(pieces) >= 5:
                    families.add(pieces[3])
            try:
                source = data.decode("utf-8-sig")
                tree = ast.parse(source, filename=relative)
                records = declarations(tree)
                api[relative] = records
                discovered = static_catalogs(tree)
                if discovered:
                    catalogs[relative] = discovered
                declarations_total += len(records)
                public_total += sum(bool(record["public_name"]) for record in records)
            except (SyntaxError, UnicodeError) as exc:
                failures.append({"path": relative, "error": str(exc)})
            if relative in CRITICAL:
                target = output / "hf_source" / relative.removeprefix("src/transformers/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                critical_found[relative] = {"sha256": sha256(data), "lines": len(data.splitlines())}
        for license_name in ("LICENSE", "NOTICE"):
            try:
                license_member = tar.getmember(prefix + license_name)
            except KeyError:
                continue
            stream = tar.extractfile(license_member)
            if stream is not None:
                (output / f"hf_source_{license_name}").write_bytes(stream.read())
    summary = {
        "schema_version": "spobs-hf-source-audit/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Complete published source-distribution inventory and runtime Python AST signatures; no upstream execution",
        "manual_review": "Deep selected execution-path source reading is documented separately in TRANSFORMERS_AUDIT.md",
        "version": args.version, "metadata_url": metadata_url,
        "archive_url": distribution["url"], "archive_file": archive.name,
        "archive_sha256": actual_hash, "archive_bytes": len(archive_bytes),
        "archive_files": len(inventory), "archive_category_files": dict(sorted(categories.items())),
        "runtime_python_files": python_files, "runtime_python_bytes": source_bytes,
        "runtime_python_lines": source_lines,
        "parsed_runtime_python_files": len(api), "ast_failures": failures,
        "class_and_function_declarations": declarations_total,
        "non_underscore_declarations": public_total,
        "declaration_count_limit": "Static syntactic declarations, including methods; not the dynamic public import/API set",
        "model_family_directories": len(families), "model_families": sorted(families),
        "critical_sources": critical_found,
        "critical_sources_not_present": [path for path in CRITICAL if path not in critical_found],
        "interpreter": sys.version,
    }
    for filename, content in (("hf_audit_inventory.json", inventory), ("hf_audit_api.json", api),
                              ("hf_audit_summary.json", summary), ("hf_audit_catalogs.json", catalogs)):
        (output / filename).write_text(json.dumps(content, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"model_families", "critical_sources"}}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
