# C16 overlapping-prefill candidate

Goal: test whether Progress changes real candidate ranking while multiple long prefills progress, and whether that translates to end-to-end improvement. Not a claim of representative AgentX performance.

32 sessions × 4 original requests, C16, four B300 TP1 workers, 96 GiB KV per worker, HOST off, max batched tokens 8192; first 16 AgentInfer seed-0 sessions include 8 long-first sessions. Cache baseline uses default absolute balance threshold 64. One instrumented KV+P diagnostic is followed by one uninstrumented run each of Cache, KV, and KV+P. Report unfavorable outcomes as well as gains. Do not treat shadow scores as observed alternate latency.

Same-agent gaps are scaled to zero as in prior runs. Token IDs are reconstructed from session-local source hashes, so cross-session prompt sharing is not represented. Record TTFT, TPOT, session JCT, throughput, cache hits, actual work, queue, prefill and GPU interference. The trace was selected on workload features, not measured policy outcomes.

## Follow-up C32 diagnostic

The completed C16 Progress diagnostic reported 38 candidate snapshots with changed remaining work but no change in the minimum-score candidate set. Of 128 requests, 69 had exactly one indexed cache hit, selected in 68 cases; many other candidate snapshots had an idle prefill ledger. Test the same 32 sessions at C32, where all 32 sessions can overlap and 16 have long first requests. Hypothesis: with more concurrent prefills, fewer idle-prefill alternatives will make the updated remaining-work term relevant to the minimum. This is a hypothesis, not a prediction of better latency.

After C16 completes, run one C32 instrumented KV+P diagnostic with identical hardware and trace. If at least one same-state minimum candidate set changes when replacing initial reservations with progress, or a strict ranking flip is observed, run one C32 uninstrumented Cache, KV, and KV+P cell in that order. Otherwise stop the C32 screen and report the null result. Record all outcomes, including regressions and any run invalidated by interference. The gate is decided before observing C32 results.

The C32 screen passed (8 same-state minimum-set changes). Following the user's earlier request to compare affinity as well, add one KV+A and one KV+A+P cell after the three primary C32 cells, irrespective of their results. This five-arm extension was declared before seeing the KV or KV+P C32 performance results. Do not use the KV+P diagnostic shadow as evidence for affinity routing.
