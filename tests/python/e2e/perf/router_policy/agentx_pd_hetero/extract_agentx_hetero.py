#!/usr/bin/env python3
"""Regenerate the agentX PD policy-performance slices from the official dataset.

The slices are fixed lists of session ids taken from the official agentX replay
dataset (``traces.jsonl``, 393 sessions). Each session keeps its own `requests`
verbatim up to a turn cap, so the prompts are real agent traffic; nothing is
synthesised. Because the id lists are written down here, regeneration is exact
and the script verifies both the source and the output SHA-256.

Usage:
  python extract_agentx_hetero.py <path/to/agentx-data/traces.jsonl> [outdir]

Outputs, next to this script by default:
  agentx-hetero12x12.jsonl  12 sessions x 12 turns
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SOURCE_SHA256 = "e39cd2ff"  # prefix only: the full digest is printed for recording

# 12 sessions x 12 turns. Selection: max single-request input <= 118000 tokens
# (the replay rebuilds prompts, so keep headroom under the engine window), then
# four sessions from each of the 10th/50th/90th percentile of 12-turn input
# total. That gives a 2.24x session-size spread while every session has the same
# turn count, which is what exposes a request-count-based balancer.
HETERO_12x12 = [
    "a89db4725a97da42ea24bd5128a093ffb215",
    "751fbb2089c7c4303f3ed241f3a7a27a3f2c",
    "b4c638cc16e89d17c18da83951a87fbc44dd",
    "3c1fbf5cf98484c7f9ef07f2d4b00e65d829",
    "e34c6bb77dc3a1cd393b29f573135d9d76a0",
    "6d5c6b957d5f634a7f820ddff7389e161def",
    "1007f04d2c64580308858bb42f292992929d",
    "578354185933aedcdc6c17fbfbfcfb48876b",
    "7754cd1191c469354df23c7d4e9de242f7cb",
    "1a271af3dedd40be2efd413ef35fe7f0f53b",
    "f204bbf87b0680d205fe8a14441fbde0ee71",
    "4ba38d0b213f0aeab695e16da9c91635a817",
]

SLICES = (
    ("agentx-hetero12x12.jsonl", HETERO_12x12, 12),
)


def build(sessions: dict, ids: list[str], turns: int) -> list[dict]:
    missing = [i for i in ids if i not in sessions]
    if missing:
        raise SystemExit(f"source dataset is missing sessions: {missing}")
    out = []
    for sid in ids:
        one = json.loads(json.dumps(sessions[sid]))
        one["requests"] = one["requests"][:turns]
        if len(one["requests"]) != turns:
            raise SystemExit(f"{sid}: fewer than {turns} turns in the source")
        out.append(one)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
    raw = src.read_bytes()
    print(f"source {src} sha256={hashlib.sha256(raw).hexdigest()}")
    sessions = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            one = json.loads(line)
            sessions[one["id"]] = one
    print(f"source sessions={len(sessions)}")
    for filename, ids, turns in SLICES:
        rows = build(sessions, ids, turns)
        body = "".join(json.dumps(r) + "\n" for r in rows)
        target = outdir / filename
        target.write_text(body, encoding="utf-8")
        total_in = sum(int(r.get("in") or 0) for s in rows for r in s["requests"])
        total_out = sum(int(r.get("out") or 0) for s in rows for r in s["requests"])
        print(f"{filename}: sessions={len(rows)} turns={turns} "
              f"requests={sum(len(s['requests']) for s in rows)} "
              f"input={total_in:,} output={total_out:,} "
              f"in/out={total_in / max(total_out, 1):.1f} "
              f"sha256={hashlib.sha256(body.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
