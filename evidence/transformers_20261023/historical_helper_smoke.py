"""Isolated trusted helpers; never runs historical scripts' top-level workflows."""
import ast
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

root = pathlib.Path(__file__).resolve().parents[2]
checks = []


def load_functions(relative, names, namespace):
    tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), relative, "exec"), namespace)
    return namespace


with tempfile.TemporaryDirectory(prefix="spobs-historical-helpers-") as folder:
    folder = pathlib.Path(folder)
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(200, dtype=np.uint8)
    arr[:, :, 1] = np.arange(100, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 180
    image = Image.fromarray(arr)
    path = folder / "source.png"
    image.save(path)
    tree = ast.parse((root / "scripts/make_frame_compare.py").read_text(encoding="utf-8"))
    crop_call = next(node.value for node in tree.body if isinstance(node, ast.Assign) and
                     any(isinstance(target, ast.Name) and target.id == "CROP" for target in node.targets))
    crop = {item.arg: ast.literal_eval(item.value) for item in crop_call.keywords}
    ns = load_functions("scripts/make_frame_compare.py", {"crop_series_panel", "resize_to_width"}, {"Image": Image, "CROP": crop})
    actual = ns["crop_series_panel"](str(path))
    expected = image.crop((int(200 * crop["x0"]), int(100 * crop["y0"]), int(200 * crop["x1"]), int(100 * crop["y1"])))
    assert actual.tobytes() == expected.tobytes()
    checks.append({"name": "historical crop helper source-derived bounds", "size": actual.size, "passed": True})
    resized = ns["resize_to_width"](image, 80)
    reference = image.resize((80, 40), 3)
    assert resized.tobytes() == reference.tobytes()
    checks.append({"name": "Pillow Resampling.LANCZOS preserves original numeric filter bytes", "size": resized.size, "passed": True})
    ns = load_functions("scripts/pixel_analyze_frames.py", {"analyze"}, {"np": np, "Image": Image, "os": os})
    metrics = ns["analyze"](str(path))
    assert all(np.isfinite(metrics[key]) for key in ["colored_pct", "white_pct", "dark_pct"])
    checks.append({"name": "pixel analysis helper on finite synthetic input", "metrics": metrics, "passed": True})
    styles = {name: ParagraphStyle(name, fontName="Helvetica", fontSize=10) for name in ["stat", "stat_label", "h2"]}
    ns = load_functions("scripts/build_audit_pdf.py", {"stat_row", "add_heading"}, {
        "Table": Table, "TableStyle": TableStyle, "Paragraph": Paragraph, "Any": Any, "AVAIL_W": 480.0,
        "S": styles, "CARD_BG": colors.whitesmoke, "ACCENT": colors.blue, "hashlib": hashlib,
    })
    row = ns["stat_row"]([("33", "parameters"), ("3", "steps"), ("0", "errors")])
    heading = ns["add_heading"]("Helper smoke", "h2", 1)
    assert heading.bookmark_level == 1 and heading.bookmark_text == "Helper smoke"
    pdf_path = folder / "helper-smoke.pdf"
    SimpleDocTemplate(str(pdf_path), pagesize=(600, 600)).build([heading, row])
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    checks.append({"name": "ReportLab variadic styles and heading metadata render to temporary PDF", "pdf_bytes": pdf_path.stat().st_size, "passed": True})
    tree = ast.parse((root / "scripts/rasterize_audit_pdf.py").read_text(encoding="utf-8"))
    statements = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)) or
                  isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pdfium" for target in node.targets)]
    bindings = {}
    exec(compile(ast.Module(body=statements, type_ignores=[]), "scripts/rasterize_audit_pdf.py", "exec"), bindings)
    document = bindings["pdfium"].PdfDocument(str(pdf_path))
    assert len(document) == 1
    page = next(iter(document))
    bitmap = page.render(scale=120 / 72)
    rendered = bitmap.to_pil()
    assert rendered.width > 0 and rendered.height > 0
    checks.append({"name": "typed native PDFium subset renders actual temporary PDF", "pages": len(document), "image_size": rendered.size, "passed": True})
    bitmap.close()
    page.close()
    document.close()

report = {"complete": True, "passed": len(checks), "failed": 0,
          "scope": "Isolated trusted helper functions on synthetic temporary inputs; original historical reports were not regenerated.", "checks": checks}
(root / "evidence/transformers_20261023/historical_helper_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
