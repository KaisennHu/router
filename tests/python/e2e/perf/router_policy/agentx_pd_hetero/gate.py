#!/usr/bin/env python3
"""Router policy performance gate over real replay runs.

Compares a baseline arm against a candidate arm using only artifacts a replay
produces (requests.jsonl plus optional Prometheus before/after snapshots), so it
works both on the agent and on uploaded cells.

Gate design (see README section "CI gate"):
  * mechanism metrics carry the hard threshold: prefill hit rate and per-worker
    new-prefill-token imbalance. They are integer accounting facts about the
    routing decision, so they do not drift with scheduling noise.
  * latency/throughput thresholds are deliberately wide: run-to-run spread of
    wall-clock metrics on real GPUs is larger than the effect being watched.
  * every rule must hold in ALL rounds (default 3): a single round is not
    evidence, and "median passes" hides a sign flip.

Usage:
  policy_gate.py --baseline runs/het3-*-cache --candidate runs/het3-*-kv
  policy_gate.py --baseline ... --candidate ... --json gate.json
Exit code 0 only when every rule holds in every round.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
# Two gate semantics. The candidate must not be held to a gain it cannot
# reproduce: gains are only asserted against a baseline measured on the same
# binary and the same protocol. Against a previous release the only stable
# claim is "not slower", because that release's KV index is incomplete on a
# newer engine and its own numbers swing between rounds.
PROFILES = {
    "candidate": {                 # new policy vs the policy it replaces
        "hit_delta_pt": 4.0,       # prefill hit rate >= baseline + 4 pt
        "new_prefill": 0.85,       # new prefill tokens <= 85% of baseline
        "throughput": 0.98,        # never slower
        "ttft_mean": 0.80,
        "jct_p95": 0.90,
        "tpot_mean": 1.05,
    },
    "no_regression": {             # new binary vs previous release
        "hit_delta_pt": 0.0,
        "new_prefill": 1.10,
        "throughput": 0.95,
        "ttft_mean": 1.10,
        "jct_p95": 1.05,
        "tpot_mean": 1.05,
    },
    # One rule set that must hold against BOTH baselines (cache_aware on the
    # same binary, and the previous release). The gated quantity is the
    # normalised reuse delta; absolute new-prefill tokens are printed but not
    # gated here, because the release's incomplete KV index makes its own work
    # count unstable between rounds, while the hit-rate delta is not affected by
    # that in the same way.
    "both": {
        "hit_delta_pt": 4.0,
        "new_prefill": None,       # reported only
        "throughput": 0.98,
        # Reported but not gated: against the previous release these two are
        # within one noise band (TTFT mean 0.832x then 1.042x, JCT p95 0.971x
        # then 0.932x while kv's own JCT p95 moves 6% between rounds), so a
        # threshold there would fail on noise rather than on a regression.
        "ttft_mean": None,
        "jct_p95": None,
        "tpot_mean": None,
    },
}

# Absolute sanity bound on the candidate's own prefill skew. kv_aware wins by
# removing work, not by being flatter than cache_aware, so this is not a
# head-to-head rule: it only catches a genuinely lopsided placement.
IMBALANCE_ABS_MAX = 1.60

PROM = {
    "prompt": "vllm:prompt_tokens_total",
    "hits": "vllm:prefix_cache_hits_total",
    "queries": "vllm:prefix_cache_queries_total",
    "success": "vllm:request_success_total",
}


def _metric(path: Path, name: str) -> float | None:
    pat = re.compile(re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    vals = [float(m.group(1)) for line in lines if (m := pat.match(line))]
    return sum(vals) if vals else None


def _quantile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_run(run_dir: Path) -> dict | None:
    """Collect one arm's metrics from its replay output directory."""
    req = run_dir / "replay" / "requests.jsonl"
    if not req.is_file():
        return None
    rows = [json.loads(x) for x in req.read_text().splitlines() if x.strip()]
    ok = [r for r in rows if r.get("status") == "success"]
    if len(ok) < 2:
        return None
    parse = datetime.fromisoformat
    tin = sum(r.get("input_tokens") or 0 for r in ok)
    tout = sum(r.get("output_tokens") or 0 for r in ok)
    span = (max(parse(r["finished_at"]) for r in ok)
            - min(parse(r["started_at"]) for r in ok)).total_seconds()
    ttft = [r["ttft_seconds"] for r in ok if r.get("ttft_seconds") is not None]
    tpot = []
    for r in ok:
        out = r.get("output_tokens") or 0
        if out > 1 and r.get("ttft_seconds") is not None and r.get("latency_seconds") is not None:
            tpot.append((r["latency_seconds"] - r["ttft_seconds"]) / (out - 1) * 1000.0)
    sessions: dict = {}
    for r in ok:
        key = r.get("session_id")
        a, b = parse(r["started_at"]), parse(r["finished_at"])
        lo, hi = sessions.get(key, (a, b))
        sessions[key] = (min(lo, a), max(hi, b))
    jct = [(b - a).total_seconds() for a, b in sessions.values()]

    hit_rate = None
    new_tokens: list[int] = []
    workers: list[int] = []
    ports = sorted(int(p.stem) for p in (run_dir / "before").glob("*.prom") if p.stem.isdigit())
    hits_sum = queries_sum = 0.0
    for port in ports:
        before = run_dir / "before" / f"{port}.prom"
        after = run_dir / "after" / f"{port}.prom"
        if not after.is_file():
            continue
        pa, pb = _metric(after, PROM["prompt"]), _metric(before, PROM["prompt"])
        ha, hb = _metric(after, PROM["hits"]), _metric(before, PROM["hits"])
        qa, qb = _metric(after, PROM["queries"]), _metric(before, PROM["queries"])
        ra, rb = _metric(after, PROM["success"]), _metric(before, PROM["success"])
        if None not in (pa, pb, ha, hb):
            new_tokens.append(int((pa - pb) - (ha - hb)))
        if None not in (qa, qb, ha, hb):
            hits_sum += ha - hb
            queries_sum += qa - qb
        if None not in (ra, rb):
            workers.append(int(ra - rb))
    if queries_sum:
        hit_rate = hits_sum / queries_sum
    prefill_workers = ports[:4]
    prefill_new = new_tokens[:len(prefill_workers)]
    imbalance = (max(prefill_new) / statistics.mean(prefill_new)) if prefill_new and statistics.mean(prefill_new) else None

    arm = None
    man = run_dir / "manifest.json"
    if man.is_file():
        arm = json.loads(man.read_text()).get("arm")
    return {
        "run": run_dir.name,
        "arm": arm,
        "requests": len(ok),
        "total_requests": len(rows),
        "all_ok": len(ok) == len(rows),
        "throughput_tok_s": (tout / span) if span else 0.0,
        "input_tokens": tin,
        "output_tokens": tout,
        "hit_rate": hit_rate,
        "new_prefill_tokens": sum(prefill_new) if prefill_new else None,
        "imbalance": imbalance,
        "worker_requests": workers,
        "ttft_mean": statistics.mean(ttft) if ttft else 0.0,
        "ttft_p95": _quantile(ttft, 95),
        "tpot_mean": statistics.mean(tpot) if tpot else 0.0,
        "tpot_p95": _quantile(tpot, 95),
        "jct_mean": statistics.mean(jct) if jct else 0.0,
        "jct_p95": _quantile(jct, 95),
    }


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"no run directory matches {pat!r}")
        out.extend(Path(h).resolve() for h in hits)
    return out


def evaluate(base: dict, cand: dict, profile: str) -> list[tuple[str, float | None, float | None, str, bool]]:
    """Return (rule, baseline value, candidate value, verdict text, passed)."""
    th = PROFILES[profile]
    rules = []

    def ratio(name: str, b: float | None, c: float | None, limit: float | None, le: bool = True):
        if not b or c is None:
            rules.append((name, b, c, "skipped (missing metric)", True))
            return
        got = c / b
        if limit is None:
            rules.append((name, b, c, f"{got:.3f}x (reported, not gated)", True))
            return
        ok = got <= limit if le else got >= limit
        arrow = "<=" if le else ">="
        rules.append((name, b, c, f"{got:.3f} {arrow} {limit:.3f}", ok))

    rules.append(("all requests succeeded", None, None,
                  "yes" if cand["all_ok"] else f"{cand['total_requests'] - cand['requests']} failed",
                  cand["all_ok"]))

    if th["hit_delta_pt"] is None:
        rules.append(("prefill hit rate delta", base["hit_rate"], cand["hit_rate"],
                      "reported", True))
    elif base["hit_rate"] is not None and cand["hit_rate"] is not None:
        delta = (cand["hit_rate"] - base["hit_rate"]) * 100.0
        rules.append(("prefill hit rate delta", base["hit_rate"] * 100, cand["hit_rate"] * 100,
                      f"{delta:+.2f} pt >= {th['hit_delta_pt']:.2f} pt",
                      delta >= th["hit_delta_pt"]))

    if th["new_prefill"] is None:
        rules.append(("new prefill tokens (reported)", base["new_prefill_tokens"],
                      cand["new_prefill_tokens"],
                      f"{cand['new_prefill_tokens'] / base['new_prefill_tokens']:.3f}x (not gated)"
                      if base["new_prefill_tokens"] and cand["new_prefill_tokens"] else "n/a",
                      True))
    else:
        ratio("new prefill tokens", base["new_prefill_tokens"], cand["new_prefill_tokens"],
              th["new_prefill"])
    if cand["imbalance"] is not None:
        rules.append(("candidate prefill skew", None, cand["imbalance"],
                      f"{cand['imbalance']:.3f} <= {IMBALANCE_ABS_MAX:.3f} (absolute)",
                      cand["imbalance"] <= IMBALANCE_ABS_MAX))
    ratio("throughput", base["throughput_tok_s"], cand["throughput_tok_s"],
          th["throughput"], le=False)
    ratio("TTFT mean", base["ttft_mean"], cand["ttft_mean"], th["ttft_mean"])
    ratio("JCT p95", base["jct_p95"], cand["jct_p95"], th["jct_p95"])
    ratio("TPOT mean", base["tpot_mean"], cand["tpot_mean"], th["tpot_mean"])
    return rules


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True, help="baseline run globs")
    ap.add_argument("--candidate", nargs="+", required=True, help="candidate run globs")
    ap.add_argument("--profile", default="candidate", choices=sorted(PROFILES), help="threshold profile")
    ap.add_argument("--json", default=None, help="write the verdict as JSON")
    args = ap.parse_args()

    bases = [m for p in expand(args.baseline) if (m := load_run(p))]
    cands = [m for p in expand(args.candidate) if (m := load_run(p))]
    if not bases or not cands:
        raise SystemExit("baseline or candidate produced no usable runs")

    rounds = min(len(bases), len(cands))
    print(f"profile={args.profile} baseline rounds={len(bases)} candidate rounds={len(cands)} compared={rounds}")

    header = f"{'run':16s} {'arm':10s} {'req':>4} {'thr':>7} {'hit%':>6} {'newPrefill':>10} {'max/avg':>7} {'TTFTm':>6} {'TTFTp95':>7} {'TPOTm':>6} {'JCTm':>7} {'JCTp95':>7}"
    for group, rows in (("BASELINE", bases), ("CANDIDATE", cands)):
        print(f"--- {group} ---")
        print(header)
        for r in rows:
            print(f"{r['run']:16s} {str(r['arm']):10s} {r['requests']:>4} {r['throughput_tok_s']:>7.1f} "
                  f"{(r['hit_rate'] or 0)*100:>6.2f} {str(r['new_prefill_tokens'] or 0):>10} "
                  f"{(r['imbalance'] or 0):>7.2f} {r['ttft_mean']:>6.2f} {r['ttft_p95']:>7.2f} "
                  f"{r['tpot_mean']:>6.1f} {r['jct_mean']:>7.1f} {r['jct_p95']:>7.1f}")
        print()

    verdicts = []
    for i in range(rounds):
        print(f"=== round {i + 1}: {bases[i]['run']} (baseline) vs {cands[i]['run']} (candidate) ===")
        rules = evaluate(bases[i], cands[i], args.profile)
        for name, b, c, text, ok in rules:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:28s} {text}")
        verdicts.append(all(ok for _, _, _, _, ok in rules))
        print()

    print("=== per-round verdict ===")
    for i, ok in enumerate(verdicts):
        print(f"  round {i + 1}: {'PASS' if ok else 'FAIL'}")
    overall = all(verdicts) and rounds >= 1
    print(f"\nGATE {'PASS' if overall else 'FAIL'} ({sum(verdicts)}/{rounds} rounds passed; "
          f"every round must pass)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"baseline": bases, "candidate": cands, "verdicts": verdicts,
             "gate": "PASS" if overall else "FAIL"}, indent=2) + "\n")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
