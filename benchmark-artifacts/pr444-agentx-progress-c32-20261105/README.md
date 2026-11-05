# Colocated AgentX Progress experiment

`report.html` is self-contained and contains the C32 five-arm results; `c16-report.html` contains the C16 three-arm comparison. `results.json` contains aggregate distributions and worker metrics. `workload-manifest.json` records selection and source hashes. The selected trace is derived from the public AgentX dataset; source prompt data and unrelated host-process details are not redistributed.

The archive additionally contains per-request client timing/token rows and per-session durations, plus the C16/C32 instrumented diagnostics. One C16 KV+P attempt was stopped after a foreign process entered a selected GPU; only its replacement is included in the valid results. One successful run per arm is not evidence of repeatability.
