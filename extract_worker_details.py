"""Extract audited worker and NIXL distributions from the original report data."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report_data", type=Path, help="Original report-data.json")
args = parser.parse_args()
report = json.loads(args.report_data.read_text())
if report.get("fixture") is not False or len(report["repeats"]) != 36:
    raise ValueError("Expected 36 formal, non-fixture runs")
official = {(r["concurrency"], r["arm"], r["repeat"]): r for r in
            csv.DictReader((ROOT / "data/repeats-metrics.csv").open())}
rows, seen = [], set()
for cell in report["repeats"]:
    key = (str(cell["concurrency"]), cell["arm"], str(cell["repeat"]))
    if key not in official or key in seen:
        raise ValueError(("unmatched or duplicate run", key))
    seen.add(key)
    source = official[key]
    if (cell["requests"], cell["sessions"], cell["input_tokens"], cell["output_tokens"]) != (730, 32, 56570112, 552859):
        raise ValueError(("workload mismatch", key))
    out = {"concurrency": key[0], "arm": key[1], "repeat": key[2]}
    for side, prefix in (("prefill", "P"), ("decode", "D")):
        aggregate = cell["stages"][side]
        audit = aggregate["iteration_metric_audit"]
        expected = float(source[f"{side}_local_computed_prompt_tokens"]) + float(source[f"{side}_generated_tokens"])
        if not audit["use_as_actual_work"] or audit["excess"] != 0 or audit["raw_iteration_token_sum"] != expected:
            raise ValueError(("aggregate iteration-token audit failed", key, side))
        out[f"{prefix}_iteration_tokens"] = int(expected)
        worker_sum = 0
        for idx in range(4):
            name = f"{prefix}{idx}"
            matches = [w for w in cell["workers"] if w["worker"] == name]
            if len(matches) != 1:
                raise ValueError(("missing worker", key, name))
            worker = matches[0]
            wa = worker["iteration_metric_audit"]
            if not wa["use_as_actual_work"] or wa["excess"] != 0:
                raise ValueError(("worker iteration-token audit failed", key, name))
            out[f"{name}_iteration_tokens"] = int(wa["raw_iteration_token_sum"])
            worker_sum += wa["raw_iteration_token_sum"]
        if worker_sum != expected:
            raise ValueError(("worker iteration sum mismatch", key, side))
    nixl = cell["stages"]["decode"]["nixl"]
    if nixl["transfers"] != 730 or nixl["failed_transfers"] != 0:
        raise ValueError(("incomplete NIXL data", key))
    for metric, label, divisor in (("xfer_time_s", "xfer_ms", .001),
                                    ("bytes_per_transfer", "size_mib", 2**20)):
        hist = nixl[metric]
        if hist["n"] != 730:
            raise ValueError(("incomplete NIXL histogram", key, metric))
        out[f"{label}_mean"] = hist["mean"] / divisor
        for q in ("p50", "p95", "p99"):
            low, high = hist["quantile_bucket_bounds"][q]
            out[f"{label}_{q}_low"] = low / divisor
            out[f"{label}_{q}_high"] = "" if high is None else high / divisor
    rows.append(out)
if seen != set(official):
    raise ValueError("Not all formal runs were extracted")
destination = ROOT / "data/repeats-worker-detail.csv"
with destination.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} audited runs to {destination}")
