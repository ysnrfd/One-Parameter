#!/usr/bin/env python3
"""Rasterize every page of the final audit PDF for VLM visual verification."""
from __future__ import annotations

import os
from collections.abc import Iterator
from importlib import import_module
from typing import Protocol, cast

from PIL import Image


class _Bitmap(Protocol):
    def to_pil(self) -> Image.Image: ...


class _Page(Protocol):
    def render(self, *, scale: float) -> _Bitmap: ...


class _Document(Protocol):
    def __iter__(self) -> Iterator[_Page]: ...
    def __len__(self) -> int: ...


class _Pdfium(Protocol):
    def PdfDocument(self, path: str) -> _Document: ...


# The optional native binding has no complete PEP 561 interface. Describe the
# exact public subset this historical recipe consumes, rather than inventing
# types for the full third-party module or suppressing unrelated type errors.
pdfium = cast(_Pdfium, import_module("pypdfium2"))

SRC = "/home/z/my-project/evidence/report_assets/SPOBS_Forensic_Audit.pdf"
OUT = "/home/z/my-project/evidence/report_assets/pdf_pages"
os.makedirs(OUT, exist_ok=True)

pdf = pdfium.PdfDocument(SRC)
for i, page in enumerate(pdf):
    bitmap = page.render(scale=120 / 72)  # 120 dpi
    img = bitmap.to_pil()
    img.save(f"{OUT}/page_{i + 1:02d}.png")
print("rendered", len(pdf), "pages to", OUT)
