# Two GPUs, C16: replay-order sensitivity

Keep the same 32-session AgentX subset, GPUs 0/1, model, 96 GiB HBM KV/worker, C16, software versions, policies and warmup/reset procedures as the completed seed-0 run. Change only AgentInfer sample_seed from 0 to 1. This changes the deterministic replay order and any seed-derived replay inputs; it is not a repetition of an identical arrival schedule.

For seed 0 the first 16 sampled sessions contain 4 long-first sessions; seed 1 contains 6. Hypothesis: more long prefills in the first wave create more opportunities for progress-aware decisions. This is a workload-order sensitivity test selected before seeing its performance.

Run Cache, KV, KV+P once each; report all metrics and unfavorable outcomes. Confirm 128 requests, exact token totals, progress acceptance, no selected-GPU interference, per-worker cache hit/new work, queue, prefill, preemptions, TTFT/TPOT/JCT distributions and throughput.
