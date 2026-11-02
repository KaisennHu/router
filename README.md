# gRPC P/D routing benchmark data for router PR 469

The figures and CSV files contain the completed formal C4/C8/C16/C32
benchmark results described in https://github.com/vllm-project/router/pull/469.
No prompts, model weights, or raw dataset are included.

- `figures/progress-impact.png`: progress on versus off with decode affinity;
  each bar compares the means of two independent runs per setting.
- `figures/full-vs-cache.png`: the full KV-aware configuration versus
  Cache-aware P; Cache has one run and Full is the mean of two runs.
- `figures/five-policy-end-to-end.png`: five-policy comparison; each KV
  variant is the mean of its two runs, and Cache has one run.
- `data/c4-worker-decomposition.csv`: C4 per-worker queue and prefill
  contributions, reproducible from the two published formal CSVs.
- `data/c4-isolation-affinity-r1*.csv`: the five-mode C4 diagnostic run and
  its comparisons. This is a separate instrumented closed-loop experiment,
  not an extra repetition of the production benchmark.
- `data/c4-instrumented-example.json`: a sanitized record of one routing
  decision and its worker timings, including source hashes and limitations.
- `report/`: the prior detailed report in Chinese and its small supporting CSV
  files. The interactive HTML has its figures and display data embedded;
  download the directory to use its CSV links. The copy replaces links to the
  large raw JSON with the corresponding aggregate CSV and removes a local path
  from the status file. The PDF is unchanged.
- `data/repeats-metrics.csv`: one row per validated run (36 total) with
  request/task metrics and aggregate P/D counters.
- `data/workers.csv`: per-worker metrics for those runs.
- `data/repeats-worker-detail.csv`: per-worker vLLM iteration-token sums and
  receiver-side NIXL transfer time and size histograms. All 36 aggregate and
  288 worker iteration sums match first-computed prompt plus generated tokens;
  these counters are not GPU FLOPs or exact per-scheduler-step work.
- `data/repeat-pairs.csv`: matched-run progress and Cache comparisons.

Workload: 8 B300 GPUs, Qwen3-4B-Instruct-2507 BF16 TP1, 4P+4D, NIXL,
128 GiB KV per worker. AgentInfer AgentX replay used a 32-session / 730-request
subset of `semianalysisai/cc-traces-weka-062126` at revision
`23f152f6f0f9399a85901b89a6458def0ef16729`. Subset SHA-256:
`3d7e5e46b5488ade48073ad273199c4abb34ec1c2b5a91e7db607760e013a279`.
Same-agent gap scale was zero; each session allowed one in-flight request.
Each run completed 730 requests, with 56,570,112 input and 552,859 output
tokens. Different concurrency caps use different synthetic token IDs, with the
same lengths and prefix-sharing structure. Results are descriptive, not
confidence intervals or causal counterfactuals.

The charts can be recreated with Python, matplotlib, and numpy:

```sh
python3 make_figures.py
python3 make_full_vs_cache.py
python3 make_c4_worker_analysis.py
# To re-extract the compact worker data from the original collected report:
python3 extract_worker_details.py /path/to/report-data.json
```

`SHA256SUMS` records the published data, scripts, and image contents.
