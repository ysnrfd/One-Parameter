"""Export a portable, explicit Python entry point for an observed workflow.

The project is inert JSON data. Its compressed payload is never Python source;
importing the generated file does not open a workspace or execute any node.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from typing import Any

from . import __version__


def python_project(project: dict[str, Any], graph_id: str) -> bytes:
    """Capture complete portable state, including plugin and checkpoint inputs."""
    extension = project.get("extensions", {})
    graphs = extension.get("graphs", [])
    if graph_id not in {graph.get("id") for graph in graphs}:
        raise KeyError("workflow does not exist")
    if any(job.get("status") in {"queued", "running", "cancelling"} for job in extension.get("jobs", [])):
        raise ValueError("Finish or cancel active jobs before exporting a runnable project")
    content = json.dumps(project, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    if len(content) > 64 * 1024 * 1024:
        raise ValueError("The portable Python project exceeds the 64 MiB project limit")
    digest = hashlib.sha256(content).hexdigest()
    payload = base64.b64encode(zlib.compress(content, 9)).decode("ascii")
    source = '''#!/usr/bin/env python3
"""Runnable Parameter Studio project, exported by Transformers Visual Coding.

Requires single-parameter-observatory __STUDIO_VERSION__ and its installed runtime dependencies.
The embedded snapshot includes experiments, plugins, workflows and checkpoints.
Importing this module does not execute the project. Run it explicitly to queue
the selected workflow. Edit project_data() / graph nodes before execution to
customize data, layers, optimizer parameters or interpreted forward code.

This exported file is a normal local Python program, outside the browser's
script sandbox. Review edits before running them.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

GRAPH_ID = __GRAPH_ID__
PROJECT_SHA256 = __PROJECT_SHA__
PROJECT_DATA = __PROJECT_DATA__


def project_data():
    content = zlib.decompress(base64.b64decode(PROJECT_DATA, validate=True))
    if hashlib.sha256(content).hexdigest() != PROJECT_SHA256:
        raise ValueError("Embedded project data does not match its SHA-256")
    return json.loads(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=600, help="Job wait deadline in seconds")
    parser.add_argument("--output", type=Path, help="Save the exact observed job JSON")
    parser.add_argument("--artifacts", type=Path, help="Save this job's immutable artifacts in this directory")
    args = parser.parse_args()
    if not 0 < args.timeout <= 86400:
        parser.error("--timeout must be finite and from 0 to 86400 seconds")
    from single_parameter_observatory.studio_session import StudioWorkspace

    workspace = StudioWorkspace()
    try:
        project = project_data()
        # Customize any declarative block here before importing the project.
        workspace.import_project(project)
        result = workspace.graph_action("graph.execute", {"graph_id": GRAPH_ID})
        job = workspace.graphs.wait_job(result["job_id"], timeout=args.timeout)
        text = json.dumps(job, ensure_ascii=True, allow_nan=False, indent=2) + "\\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        if args.artifacts:
            args.artifacts.mkdir(parents=True, exist_ok=True)
            for artifact in workspace.graphs.snapshot()["artifacts"]:
                if artifact["job_id"] != job["id"]:
                    continue
                data, filename, _mime = workspace.graphs.artifact(artifact["id"])
                # The server validates plain filenames; prefix identity to avoid collisions.
                (args.artifacts / (artifact["id"] + "-" + filename)).write_bytes(data)
        print(text, end="")
        return 0 if job["status"] == "succeeded" else 1
    finally:
        # Cancellation/timeout never leaves an owned worker running.
        workspace.close()


if __name__ == "__main__":
    raise SystemExit(main())
'''
    # Expand trusted alphabet-only payloads first, then insert the untrusted ID
    # last so marker-shaped graph names cannot be interpreted as placeholders.
    source = source.replace("__PROJECT_DATA__", repr(payload)).replace("__PROJECT_SHA__", repr(digest))
    source = source.replace("__STUDIO_VERSION__", __version__)
    source = source.replace("__GRAPH_ID__", repr(graph_id))
    return source.encode("utf-8")
