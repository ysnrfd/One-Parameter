#!/usr/bin/env python3
"""Consolidate the 18 per-frame VLM verdicts into one audit table."""
from __future__ import annotations

import glob
import json
import os
import re

VLMDIR = "/home/z/my-project/evidence/vlm"

def parse_verdict(text: str) -> dict:
    fields = {}
    for key in ("LAYOUT", "SERIES", "TEXT", "BITS", "DEFECTS", "VERDICT"):
        match = re.search(rf"{key}\s*:\s*(.*?)(?=\n[A-Z]+:|\Z)", text, re.DOTALL)
        fields[key] = " ".join(match.group(1).split()) if match else ""
    return fields

rows = []
for path in sorted(glob.glob(os.path.join(VLMDIR, "*.json"))):
    tag = os.path.basename(path)[:-5]
    with open(path) as handle:
        payload = json.load(handle)
    # CLI JSON envelope: content may be under choices[0].message.content or .content
    content = payload
    if isinstance(payload, dict):
        content = (
            payload.get("content")
            or (payload.get("choices") or [{}])[0].get("message", {}).get("content")
            or payload.get("data")
            or json.dumps(payload)
        )
    fields = parse_verdict(str(content))
    verdict = fields.get("VERDICT", "")
    rows.append({
        "frame": tag,
        "verdict": "PASS" if verdict.upper().startswith("PASS") else ("FAIL" if verdict.upper().startswith("FAIL") else "UNCLEAR"),
        "verdict_raw": verdict[:120],
        "defects": fields.get("DEFECTS", "")[:200],
        "text_read": fields.get("TEXT", "")[:160],
        "series": fields.get("SERIES", "")[:160],
        "bits": fields.get("BITS", "")[:120],
    })

out = os.path.join(VLMDIR, "_consolidated.json")
with open(out, "w") as handle:
    json.dump(rows, handle, indent=2)

passed = sum(1 for r in rows if r["verdict"] == "PASS")
failed = sum(1 for r in rows if r["verdict"] == "FAIL")
print(f"frames analyzed: {len(rows)}   PASS: {passed}   FAIL: {failed}   UNCLEAR: {len(rows)-passed-failed}")
print("-" * 100)
for r in rows:
    print(f"{r['frame']:<38} {r['verdict']:<8} defects: {r['defects'][:90] or '-'}")
print("-" * 100)
print("sample TEXT readings:")
for r in rows[:4] + rows[-3:]:
    print(f"  {r['frame']}: {r['text_read'][:140]}")
