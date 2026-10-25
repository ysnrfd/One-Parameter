"""Export actual idle studio API fixtures for deterministic browser DOM checks.

This helper creates one ordinary StudioWorkspace, reads its real capabilities
and state, and closes it. Optional training uses the real engine; this helper
never fabricates observations or executes user scripts. Run with the checkout's
configured Python environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

_CHECKOUT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT))

from single_parameter_observatory import __version__  # noqa: E402 -- this helper deliberately exports the current checkout.
from single_parameter_observatory.studio_plugins import example_plugin  # noqa: E402
from single_parameter_observatory.studio_session import StudioWorkspace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON fixture file")
    parser.add_argument("--with-example-plugin", action="store_true",
                        help="Include the actual normalized example plugin, nodes, template and panel; do not run it")
    parser.add_argument("--steps", type=int, choices=range(11), default=0,
                        help="Run 0 to 10 actual initial-experiment training steps for chart/stage fixtures")
    parser.add_argument("--template", choices=("training", "sandbox", "custom-data"), default="training",
                        help="Select an actual starter workflow; do not execute it")
    args = parser.parse_args()
    workspace = StudioWorkspace()
    try:
        if args.with_example_plugin:
            workspace.graph_action("plugin.install", {"manifest": example_plugin()})
        if args.template != "training":
            workspace.graph_action("graph.create", {"template_id": args.template})
        active_run_id = workspace.workspace["active_run_id"]
        if args.steps:
            workspace.dispatch("run.step", {"run_id": active_run_id, "count": args.steps})
        payload = {
            "schema_version": "spobs-studio-ui-fixtures/1", "version": __version__,
            "capabilities": workspace.capabilities(), "state": workspace.snapshot(),
            "graph_catalog": workspace.graphs.catalog(), "graph_state": workspace.graphs.snapshot(),
            "drafts": {"drafts": copy.deepcopy(workspace.ui_drafts), "revision": workspace._ui_drafts_revision,
                       "workspace_epoch": workspace.workspace_epoch},
            "dataset": workspace.dataset(active_run_id),
            "record": workspace.record(active_run_id, 0) if args.steps else None,
        }
    finally:
        workspace.close()
    content = (json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(json.dumps({"output": str(args.output.resolve()), "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest(), "training_steps": args.steps,
                      "scripts_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
