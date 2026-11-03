# AgentX prefill-pressure benchmark

This measures Router PR #444 in colocated P/D mode on four B300 GPUs with
Qwen3-4B-Instruct-2507, BF16, TP1 and 128 GiB of KV cache per worker.

The trace is a deliberately prefill-heavy stratum of the original AgentX
snapshot. For each source session, retain its first four chronological model
requests, including subagent calls. Require positive original output lengths,
context no longer than 196,608 tokens, total output at most 2,048 tokens and
at least 32,768 estimated novel input tokens. Among the eligible sessions select
the 128 shortest output totals, using a fixed hash as the tie-breaker. Inputs,
outputs and original hash-block snapshots are not altered. This stratum is not
representative of the full dataset; the manifest records the source checksum,
all selected source locations and the exact selection rule.

Each of C4, C8, C16 and C32 runs Cache, KV, KV+Progress, KV+Affinity and
KV+Affinity+Progress once. Same-concurrency runs use the same trace. AgentInfer
reconstructs token IDs from the original hash blocks, sets the same-agent gap
to zero and allows one request in flight per session. Consequently, this is a
pressure replay rather than a replay of original wall-clock arrival times.

The instrumented eight-session pilot must complete successfully, accept
progress updates, change the router ledger, confirm KV subscriptions, and hit
session affinity before the full matrix is started. The pilot's diagnostic
logs are excluded from performance measurements. In the matrix, each arm has
fresh workers and cache; interrupted or contaminated attempts are archived and
excluded. Background activity on other GPUs and host CPU is recorded where
visible. Single runs do not establish a stable or causal performance effect.

`python3 status.py` shows matrix progress. The full matrix automatically
generates a self-contained HTML report and a portable results archive. Average
prefill/decode phase concurrency is derived from completed-request phase time
and replay wall time; it is not GPU utilization. Worker histogram percentiles
are reported as bucket ranges. JCT is replay session duration, without actual
tool execution.

After the matrix, separate instrumented replays trace 32 of the selected
sessions at C16 and C32. They join router candidate scores and slot snapshots
with confirmed worker scheduling steps and per-request client timings. These
diagnostics are excluded from performance comparisons. A shadow no-progress
ranking uses the actual progress-on trajectory and cannot by itself establish
what an independently executed no-progress run would have done.

`diagnostic-hotspot-analysis.json` summarizes the C32 affinity hotspot and
request-level evidence. Physical worker labels can rotate across independent
runs; relabeled placement counts are descriptive, not causal estimates.
