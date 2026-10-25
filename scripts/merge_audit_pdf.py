#!/usr/bin/env python3
"""Merge the Playwright cover with the ReportLab body into the final audit PDF."""
from __future__ import annotations

from pypdf import PdfReader, PdfWriter

A4_W, A4_H = 595.28, 841.89

COVER = "/home/z/my-project/evidence/report_assets/cover.pdf"
BODY = "/home/z/my-project/evidence/report_assets/audit_body.pdf"
OUT = "/home/z/my-project/evidence/report_assets/SPOBS_Forensic_Audit.pdf"

TITLE = (
    "Single Float32 Parameter Observatory - Forensic Audit "
    "and Enterprise Rebuild"
)


def normalize_page_to_a4(page):
    """Scale a page to A4 if its dimensions do not match (sub-point Playwright drift)."""
    box = page.mediabox
    w, h = float(box.width), float(box.height)
    if abs(w - A4_W) > 0.5 or abs(h - A4_H) > 0.5:
        page.scale_to(A4_W, A4_H)
    return page


writer = PdfWriter()
cover_page = PdfReader(COVER).pages[0]
writer.add_page(normalize_page_to_a4(cover_page))
for page in PdfReader(BODY).pages:
    writer.add_page(normalize_page_to_a4(page))

writer.add_metadata(
    {
        "/Title": TITLE,
        "/Author": "Z.ai",
        "/Creator": "Z.ai",
        "/Subject": (
            "Forensic audit and enterprise rebuild of the Single Float32 "
            "Parameter Observatory (SPOBS-AUDIT-2026-10)"
        ),
    }
)
with open(OUT, "wb") as fh:
    writer.write(fh)

r = PdfReader(OUT)
sizes = {(round(float(p.mediabox.width), 1), round(float(p.mediabox.height), 1))
         for p in r.pages}
print("final pages:", len(r.pages), "sizes:", sizes)
print("written:", OUT)
