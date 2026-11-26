"""Build the four trace slices used by this repro package from official data.

Every slice is a deterministic subset of an official dataset: agentX session
JSONL, TraceLab normalized JSONL (extracted by AgentInfer from
UW-SyFI/TraceLab v0.0.2), or the Inferact SWE-bench Pro records that AgentInfer
downloads itself. Nothing here synthesizes requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED = {
    "tracelab-fit6.jsonl": "c080001af998a7f0",
    "tracelab-fit.jsonl": "c12aa934aaf3a273",
    "agentx-hetero12x12.jsonl": "e7e7fd7cd31f1784",
}


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_sessions(dst: Path, sessions: list[dict]) -> None:
    dst.write_text("\n".join(json.dumps(s) for s in sessions) + "\n", encoding="utf-8")


def max_in(session: dict) -> int:
    return max(int(r.get("in") or 0) for r in (session.get("requests") or []))


def build_agentx(src: Path, dst_dir: Path) -> None:
    """Keep the N smallest sessions (by max input tokens) and cap their turns."""
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    for count, turns, name in ((12, 12, "agentx-hetero12x12.jsonl"),):
        picked = sorted(rows, key=max_in)[:count]
        trimmed = []
        for s in picked:
            s = json.loads(json.dumps(s))
            s["requests"] = (s.get("requests") or [])[:turns]
            trimmed.append(s)
        out = dst_dir / name
        write_sessions(out, trimmed)
        print(f"{name}: sessions={len(trimmed)} entries={sum(len(s['requests']) for s in trimmed)} sha={sha16(out)}")


def build_tracelab(task_num: int, dst_dir: Path, limit: int = 200_000) -> None:
    """Filter AgentInfer's TraceLab materialization by context length."""
    try:
        from agentinfer.agentbench.replay.tracelab_source import materialize_tracelab_source
    except ImportError:  # pragma: no cover - depends on the caller's env
        sys.exit("install agentinfer (or set PYTHONPATH) to materialize TraceLab")

    src = Path(materialize_tracelab_source(task_num))
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        sessions.setdefault(row["session_id"], []).append(row)

    def max_ctx(rs: list[dict]) -> int:
        return max(int(r.get("input_tokens_total") or 0) for r in rs)

    keep = [(sid, rs) for sid, rs in sessions.items() if max_ctx(rs) <= limit]
    keep.sort(key=lambda kv: -max_ctx(kv[1]))
    for count, name in ((6, "tracelab-fit6.jsonl"), (12, "tracelab-fit.jsonl")):
        chosen = {sid for sid, _ in keep[:count]}
        out = dst_dir / name
        out.write_text(
            "\n".join(json.dumps(r) for r in rows if r["session_id"] in chosen) + "\n",
            encoding="utf-8",
        )
        print(f"{name}: sessions={len(chosen)} rows={sum(1 for r in rows if r['session_id'] in chosen)} "
              f"sha={sha16(out)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentx", type=Path, help="official agentX traces.jsonl")
    parser.add_argument("--dst", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--tracelab-task-num", type=int, default=16,
                        help="sessions to materialize before filtering")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    if args.agentx:
        build_agentx(args.agentx, args.dst)
    build_tracelab(args.tracelab_task_num, args.dst)

    print("\n--- sha256 check (first 16 hex) ---")
    bad = 0
    for name, expected in EXPECTED.items():
        path = args.dst / name
        if not path.is_file():
            print(f"{name}: MISSING")
            bad += 1
            continue
        got = sha16(path)
        flag = "ok" if got == expected else f"MISMATCH (expected {expected})"
        print(f"{name}: {got} {flag}")
        bad += got != expected
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
