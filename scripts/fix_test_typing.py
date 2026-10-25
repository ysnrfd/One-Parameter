#!/usr/bin/env python3
"""Apply the mypy test-suite fixes (enterprise pass, F-15/F-16 in AUDIT.md)."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path("/home/z/my-project/single_parameter_observatory/tests")


def patch(name: str, pairs: list[tuple[str, str]], count_expected: dict[str, int] | None = None) -> None:
    path = ROOT / name
    text = path.read_text()
    for old, new in pairs:
        occurrences = text.count(old)
        if occurrences == 0:
            if new in text:
                continue  # already applied by an earlier run
            raise SystemExit(f"{name}: pattern not found: {old[:80]!r}")
        if count_expected and old in count_expected and occurrences != count_expected[old]:
            raise SystemExit(f"{name}: pattern {old[:60]!r} found {occurrences}x, expected {count_expected[old]}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {name}")


# ---------------------------------------------------------------- bootstrap ---
BOOT_OLD = """except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap"""
BOOT_NEW = """except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]"""
for name in (
    "test_autograd.py", "test_fp32.py", "test_numerics.py", "test_optimizer.py",
    "test_parameter.py", "test_recording.py", "test_training.py",
    "test_views.py", "test_regressions.py",
):
    patch(name, [(BOOT_OLD, BOOT_NEW)])

# ---------------------------------------------------------------- autograd ---
patch("test_autograd.py", [
    # helper inserted after the imports
    (
        "class TestAutograd(unittest.TestCase):",
        '''def _grad_value(parameter: torch.nn.Parameter) -> float:
    """``parameter.grad`` as a float; every caller ran ``backward()`` first."""
    grad = parameter.grad
    assert grad is not None, "the caller ran loss.backward() first"
    return float(grad.item())


class TestAutograd(unittest.TestCase):''',
    ),
    ("        gradient = model.w.grad\n"
     "        self.assertIsNotNone(gradient, \"loss.backward() must populate parameter.grad\")",
     "        gradient = model.w.grad\n"
     "        self.assertIsNotNone(gradient, \"loss.backward() must populate parameter.grad\")\n"
     "        assert gradient is not None  # narrowing for the type checker"),
    ("        fp64 = float(reference.w.grad.item())",
     "        fp64 = _grad_value(reference.w)"),
    ("        autograd = float(model.w.grad.item())",
     "        autograd = _grad_value(model.w)"),
    ("            na.relative_error(float(fp64_model.w.grad.item()), analytical), 1e-12",
     "            na.relative_error(_grad_value(fp64_model.w), analytical), 1e-12"),
    ("            autograd_gradient=float(model.w.grad.item()),",
     "            autograd_gradient=_grad_value(model.w),"),
    ("        after_first = float(model.w.grad.item())",
     "        after_first = _grad_value(model.w)"),
    ("        after_second = float(model.w.grad.item())",
     "        after_second = _grad_value(model.w)"),
    ("        second_only = float(separate.w.grad.item())",
     "        second_only = _grad_value(separate.w)"),
    ("        self.assertEqual(float(model.w.grad.item()), 1234.0)",
     "        self.assertEqual(_grad_value(model.w), 1234.0)"),
    ("        self.assertNotEqual(float(model.w.grad.item()), 1234.0)",
     "        self.assertNotEqual(_grad_value(model.w), 1234.0)"),
])

# ---------------------------------------------------------------- optimizer ---
patch("test_optimizer.py", [
    ("        loss.backward()\n        gradient = parameter.grad.detach().clone()",
     "        loss.backward()\n"
     "        assert parameter.grad is not None\n"
     "        gradient = parameter.grad.detach().clone()"),
    ("        with torch.no_grad():\n            parameter.grad.zero_()",
     "        with torch.no_grad():\n"
     "            assert parameter.grad is not None\n"
     "            parameter.grad.zero_()"),
    ("        gradient_with_decay = float(parameter.grad.item())",
     "        assert parameter.grad is not None\n"
     "        gradient_with_decay = float(parameter.grad.item())"),
    ("        self.assertEqual(gradient_with_decay, float(other.w.grad.item()))",
     "        assert other.w.grad is not None\n"
     "        self.assertEqual(gradient_with_decay, float(other.w.grad.item()))"),
])

# ---------------------------------------------------------------- parameter ---
patch("test_parameter.py", [
    ("""                self.assertEqual(
                    model.w.detach().view(torch.int32).item()
                    & 0xFFFFFFFF,
                    int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item())
                    & 0xFFFFFFFF,
                )""",
     """                self.assertEqual(
                    int(model.w.detach().view(torch.int32).item())
                    & 0xFFFFFFFF,
                    int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item())
                    & 0xFFFFFFFF,
                )"""),
])

# ---------------------------------------------------------------- recording ---
patch("test_recording.py", [
    ("            record.loss = 0.0  # type: ignoreisc]",
     "            record.loss = 0.0  # type: ignore[misc]"),
    ("            record.steps = 5  # type: ignoreisc]",
     "            record.steps = 5  # type: ignore[misc]"),
    ('''            self.assertIsNotNone(written[name], f"{name} was not written")
            self.assertTrue(
                os.path.exists(written[name]), f"{name} missing on disk"
            )''',
     '''            self.assertIsNotNone(written[name], f"{name} was not written")
            path = written[name]
            assert path is not None
            self.assertTrue(
                os.path.exists(path), f"{name} missing on disk"
            )'''),
    ("        with open(written[\"trajectory.csv\"], newline=\"\", encoding=\"utf-8\") as handle:",
     "        trajectory_csv = written[\"trajectory.csv\"]\n"
     "        assert trajectory_csv is not None\n"
     "        with open(trajectory_csv, newline=\"\", encoding=\"utf-8\") as handle:"),
    ("        with open(written[\"trajectory.json\"], encoding=\"utf-8\") as handle:",
     "        trajectory_json = written[\"trajectory.json\"]\n"
     "        assert trajectory_json is not None\n"
     "        with open(trajectory_json, encoding=\"utf-8\") as handle:"),
    ("        with open(written[\"summary.json\"], encoding=\"utf-8\") as handle:",
     "        summary_path = written[\"summary.json\"]\n"
     "        assert summary_path is not None\n"
     "        with open(summary_path, encoding=\"utf-8\") as handle:"),
])

# ---------------------------------------------------------------- numerics ---
patch("test_numerics.py", [
    ("        # The gradient is a separate tensor from the parameter.\n"
     "        self.assertIsNot(parameter.grad, parameter)\n"
     "        self.assertNotEqual(",
     "        # The gradient is a separate tensor from the parameter.\n"
     "        self.assertIsNot(parameter.grad, parameter)\n"
     "        assert parameter.grad is not None\n"
     "        self.assertNotEqual("),
])

# ---------------------------------------------------------------- regressions -
patch("test_regressions.py", [
    ("                path = written.get(name)\n"
     "                self.assertIsNotNone(path, f\"{name} was not written for the diverged run\")\n"
     "                with open(path, encoding=\"utf-8\") as handle:",
     "                path = written.get(name)\n"
     "                self.assertIsNotNone(path, f\"{name} was not written for the diverged run\")\n"
     "                assert path is not None\n"
     "                with open(path, encoding=\"utf-8\") as handle:"),
])

# ---------------------------------------------------------------- views -------
patch("test_views.py", [
    ('''class _Args:
    """The subset of an argparse namespace that ``main._finish`` reads."""

    export = None
    plot = None
    no_timestamp = True''',
     '''def _args() -> argparse.Namespace:
    """The subset of an argparse namespace that ``main._finish`` reads."""
    return argparse.Namespace(export=None, plot=None, no_timestamp=True)'''),
    ("import os\nimport shutil\nimport tempfile\nimport unittest",
     "import argparse\nimport os\nimport shutil\nimport tempfile\nimport unittest"),
    ("args = _Args()", "args = _args()"),
    ("engine, _Args(), quiet=False", "engine, _args(), quiet=False"),
    ("        names = set(actions[0].choices)",
     "        assert actions[0].choices is not None\n"
     "        names = set(actions[0].choices)"),
    ("                record = engine.step()\n"
     "                self.assertIsNotNone(record, f\"preset {name} produced no step\")\n"
     "                self.assertIn(record.finite, (True, False))",
     "                record = engine.step()\n"
     "                self.assertIsNotNone(record, f\"preset {name} produced no step\")\n"
     "                assert record is not None\n"
     "                self.assertIn(record.finite, (True, False))"),
    ('''        class Args:
            preset = "does-not-exist"
            steps = None
            optimizer = None
            weight_decay = None
            seed = None
            no_fp64_reference = False
            no_landscape = False
            no_validation = False
            set_to_none = None
            accumulate = None

        with self.assertRaises(ObservatoryError):
            cli.resolve_config(Args())''',
     '''        args = argparse.Namespace(
            preset="does-not-exist",
            steps=None,
            optimizer=None,
            weight_decay=None,
            seed=None,
            no_fp64_reference=False,
            no_landscape=False,
            no_validation=False,
            set_to_none=None,
            accumulate=None,
        )

        with self.assertRaises(ObservatoryError):
            cli.resolve_config(args)'''),
])

print("ALL PATCHES APPLIED")
