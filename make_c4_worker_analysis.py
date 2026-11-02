"""Reproduce the C4 worker contributions from the published formal-run CSVs."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
workers = list(csv.DictReader((ROOT / "data/workers.csv").open()))
aggregates = list(csv.DictReader((ROOT / "data/repeats-metrics.csv").open()))
rows = []
for run in ("1", "2"):
    def worker(arm, name):
        matches = [r for r in workers if r["concurrency"] == "4" and r["repeat"] == run
                   and r["arm"] == arm and r["worker"] == name]
        if len(matches) != 1:
            raise ValueError((run, arm, name, len(matches)))
        return matches[0]

    def aggregate(arm):
        matches = [r for r in aggregates if r["concurrency"] == "4" and r["repeat"] == run
                   and r["arm"] == arm]
        if len(matches) != 1:
            raise ValueError((run, arm, len(matches)))
        return matches[0]

    for name in ("P0", "P1", "P2", "P3"):
        off, on = worker("kv_aware_aff", name), worker("kv_aware_aff_progress", name)
        n0, n1 = float(off["requests"]), float(on["requests"])
        q0, q1 = (float(x["queue_mean_s"]) * 1000 for x in (off, on))
        p0, p1 = (float(x["prefill_mean_s"]) * 1000 for x in (off, on))
        rows.append({"run": run, "worker": name, "requests_off": int(n0), "requests_on": int(n1),
                     "first_computed_tokens_off": int(float(off["local_computed_tokens"])),
                     "first_computed_tokens_on": int(float(on["local_computed_tokens"])),
                     "queue_mean_off_ms": q0, "queue_mean_on_ms": q1,
                     "queue_delta_contribution_ms": (n1 * q1 - n0 * q0) / 730,
                     "prefill_mean_off_ms": p0, "prefill_mean_on_ms": p1,
                     "prefill_delta_share_component_ms": (n1 - n0) / 730 * (p0 + p1) / 2,
                     "prefill_delta_within_worker_component_ms": (p1 - p0) * (n0 + n1) / (2 * 730)})
    group = rows[-4:]
    assert sum(r["requests_off"] for r in group) == sum(r["requests_on"] for r in group) == 730
    a, b = aggregate("kv_aware_aff"), aggregate("kv_aware_aff_progress")
    dq = float(b["prefill_queue_s_mean_ms"]) - float(a["prefill_queue_s_mean_ms"])
    dp = float(b["prefill_prefill_s_mean_ms"]) - float(a["prefill_prefill_s_mean_ms"])
    assert abs(sum(r["queue_delta_contribution_ms"] for r in group) - dq) < 1e-6
    assert abs(sum(r["prefill_delta_share_component_ms"] + r["prefill_delta_within_worker_component_ms"]
                   for r in group) - dp) < 1e-6

output = ROOT / "data/c4-worker-decomposition.csv"
with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} worker rows to {output}")
