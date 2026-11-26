"""Validate one #519-slice run and report request + session level metrics.

Engine counter names differ between vLLM builds (this node exports unsuffixed
names such as ``vllm:prefix_cache_hits`` while other builds export
``..._total`` / ``..._sum``), so every counter is resolved through a candidate
list instead of one hard-coded spelling.

Session JCT is the headline latency: a session's first request start to its last
request finish. Per-request latency is reported separately as request JCT.
"""
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/david_cwq/hhc/pr519-h20-20261123")
PORTS = [int(p) for p in os.environ.get("PORTS", "").split(",") if p]
PREFILL = 4

CANDIDATES = {
    "prefix_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
    "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
    "prompt_tokens": ["vllm:prompt_tokens_total", "vllm:prompt_tokens",
                      "vllm:prompt_tokens_sum"],
    "nixl_bytes": ["vllm:nixl_bytes_transferred_sum", "vllm:nixl_bytes_transferred_total",
                   "vllm:nixl_bytes_transferred"],
    "nixl_count": ["vllm:nixl_bytes_transferred_count", "vllm:nixl_num_transfers_total",
                   "vllm:nixl_num_descriptors_total", "vllm:nixl_num_descriptors"],
    "nixl_failed_transfers": ["vllm:nixl_num_failed_transfers_total",
                              "vllm:nixl_num_failed_transfers"],
    "nixl_failed_notifications": ["vllm:nixl_num_failed_notifications_total",
                                  "vllm:nixl_num_failed_notifications"],
    "nixl_expired": ["vllm:nixl_num_kv_expired_reqs_total", "vllm:nixl_num_kv_expired_reqs"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "request_success": ["vllm:request_success_total", "vllm:request_success"],
}

_cache = {}


def read_metrics(path):
    text = path.read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{", 1)[0]
        try:
            lines.append((base, float(value)))
        except ValueError:
            continue
    return lines


def metric(path, key):
    seen = _cache.get(str(path))
    if seen is None:
        seen = _cache[str(path)] = read_metrics(path)
    for candidate in CANDIDATES[key]:
        total = sum(value for base, value in seen if base == candidate)
        if any(base == candidate for base, _ in seen):
            return total
    return None


def delta(path, key):
    before, after = path / "before", path / "after"
    total = 0.0
    for port in PORTS:
        a = metric(after / f"{port}.prom", key)
        b = metric(before / f"{port}.prom", key)
        if a is None or b is None:
            return None
        total += a - b
    return total


def percentile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def quad(values):
    return {"mean": statistics.mean(values), "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
            "n": len(values)}


def group_delta(path, key, ports):
    total = 0.0
    for port in ports:
        a = metric(path / "after" / f"{port}.prom", key)
        b = metric(path / "before" / f"{port}.prom", key)
        if a is None or b is None:
            return None
        total += a - b
    return total


def run(path):
    meta = json.loads((path / "manifest.json").read_text())
    replay_manifest = json.loads((path / "replay/manifest.json").read_text())
    facts = [json.loads(x) for x in (path / "replay/requests.jsonl").read_text().splitlines() if x.strip()]
    failures = [r for r in facts if r["status"] != "success"]
    if failures:
        raise ValueError(f"{path}: {len(failures)} failed requests: "
                         f"{[(f.get('status_code'), (f.get('error') or '')[:160]) for f in failures]}")
    if replay_manifest.get("status") != "completed":
        raise ValueError(f"{path}: replay status {replay_manifest.get('status')!r}")
    parse = datetime.fromisoformat
    span = (max(parse(r["finished_at"]) for r in facts)
            - min(parse(r["started_at"]) for r in facts)).total_seconds()
    if span <= 0 or any(r["ttft_seconds"] is None or not r["output_tokens"] for r in facts):
        raise ValueError(f"{path}: missing timing or output")

    pre_q = group_delta(path, "prefix_queries", PORTS[:PREFILL])
    pre_h = group_delta(path, "prefix_hits", PORTS[:PREFILL])
    dec_q = group_delta(path, "prefix_queries", PORTS[PREFILL:])
    dec_h = group_delta(path, "prefix_hits", PORTS[PREFILL:])
    dec_prompt = group_delta(path, "prompt_tokens", PORTS[PREFILL:])
    nixl_bytes = group_delta(path, "nixl_bytes", PORTS[PREFILL:])
    nixl_count = group_delta(path, "nixl_count", PORTS[PREFILL:])
    nixl_failed = sum(v for v in (group_delta(path, k, PORTS[PREFILL:]) for k in
                                  ("nixl_failed_transfers", "nixl_failed_notifications", "nixl_expired"))
                      if v is not None) if any(
        group_delta(path, k, PORTS[PREFILL:]) is not None for k in
        ("nixl_failed_transfers", "nixl_failed_notifications", "nixl_expired")) else None
    preemptions = group_delta(path, "preemptions", PORTS[PREFILL:])
    if preemptions is not None and preemptions < -1e-6:
        raise ValueError(f"{path}: preemption counter reset")

    log = (path / "router.log").read_text()
    # Router warnings carry ANSI colour codes, so strip them before matching: the
    # literal "expected=0" never appears while escapes are in the line.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", log)
    issues = {k: plain.count(k) for k in ("KV event batch rejected",
                                          "KV event batch exceeds the event limit",
                                          "KV replay sequence gap")}
    # "expected=0" gaps mean the publisher ring had already advanced past its start
    # when this subscriber attached (buffer_steps overflow at subscribe time).
    # A gap with a non-zero expected sequence means events were dropped while the
    # run was in flight; PR #493 backfills those through replay, so they are
    # recorded as a data-quality caveat.
    gap_lines = [line for line in plain.splitlines() if "KV replay sequence gap" in line]
    issues["KV replay sequence gap (attach)"] = sum(1 for line in gap_lines if "expected=0" in line)
    issues["KV replay sequence gap (in-flight)"] = sum(1 for line in gap_lines if "expected=0" not in line)
    router_metrics = (path / "after/router.prom").read_text()

    def decisions(stage, token):
        return sum(float(line.rsplit(" ", 1)[1]) for line in router_metrics.splitlines()
                   if line.startswith("vllm_router") and f'stage="{stage}"' in line
                   and token in line and line.rsplit(" ", 1)[1].replace(".", "").isdigit())

    decode_decisions = {"context_tokens_marginal": decisions("decode", "context_tokens_marginal"),
                        "occupancy": decisions("decode", "occupancy")}

    by_session = {}
    for r in facts:
        by_session.setdefault(r["session_id"], []).append(r)
    session_jct = []
    for rows in by_session.values():
        start = min(parse(r["started_at"]) for r in rows)
        end = max(parse(r["finished_at"]) for r in rows)
        session_jct.append((end - start).total_seconds())
    if len(by_session) != 12:
        pass  # session count varies by dataset (agentX slices vs TraceLab)

    return {
        "label": path.name, "arm": meta["arm"],
        "router_sha256": meta["router_sha256"], "router_binary": meta["router_binary"],
        "trace_sha256": meta["trace_sha256"],
        "plan_namespace": json.loads((path / "replay/replay-plan.json").read_text())["plan_namespace"],
        "requests": len(facts),
        "input_tokens": sum(r["input_tokens"] for r in facts),
        "output_tokens": sum(r["output_tokens"] for r in facts),
        "request_span_seconds": span,
        "replay_wall_seconds": meta["wall_seconds"],
        "throughput_output_tokens_per_second": sum(r["output_tokens"] for r in facts) / span,
        "throughput_input_tokens_per_second": sum(r["input_tokens"] for r in facts) / span,
        "ttft_seconds": quad([r["ttft_seconds"] for r in facts]),
        "tpot_ms": quad([1000 * (r["latency_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
                         for r in facts if r["output_tokens"] > 1]),
        "request_jct_seconds": quad([r["latency_seconds"] for r in facts]),
        "session_jct_seconds": quad(session_jct),
        "session_jct_values": sorted(session_jct),
        "sessions": len(by_session),
        "prefill_hit_pct": (100 * pre_h / pre_q) if (pre_q and pre_h is not None) else None,
        "prefill_queries_tokens": pre_q, "prefill_hits_tokens": pre_h,
        "decode_hit_pct": (100 * dec_h / dec_q) if (dec_q and dec_h is not None) else None,
        "decode_queries_tokens": dec_q, "decode_hits_tokens": dec_h,
        "decode_transferred_tokens": (dec_prompt - dec_h) if (dec_prompt is not None and dec_h is not None) else None,
        "decode_transferred_bytes": nixl_bytes, "nixl_transfers": nixl_count,
        "nixl_failures": nixl_failed, "preemptions": preemptions,
        "event_issues": issues, "decode_decisions": decode_decisions,
    }


for arg in sys.argv[1:]:
    print(json.dumps(run(ROOT / "runs" / arg), indent=2))
