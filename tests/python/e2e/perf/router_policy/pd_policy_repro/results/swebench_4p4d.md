# SWE-bench Pro / 4P4D

## 配置

| 项 | 值 |
|---|---|
| trace | `inferact_codex_swebenchpro`（官方 Inferact，`task_num=12` 自动下载固定版本） |
| 拓扑 | 4P4D（WORKERS=8，PREFILL=4） |
| 引擎 | `--max-model-len 163840`，`--kv-cache-memory-bytes 27917287424`（26 GiB） |
| 引擎并发 | `--max-num-seqs 8`、`--gpu-memory-utilization 0.80`、`--block-size 16`、`--enable-prefix-caching`、`--prefix-caching-hash-algo sha256_cbor` |
| `experiment.task_num` = `experiment.max_concurrency` | **12** |
| `replay.max_inflight_requests` | **1**（每个会话同一时刻最多 1 个在途请求，故实际并发≈会话数） |
| `experiment.task_timeout_seconds` / `run_timeout_seconds` | 3600 / 10800 |
| 每臂请求 | 247 |
| 轮次 | 2 |
| Router | main `9605956f`（`--policy` / `--prefill-policy` / `--decode-policy` 三者相同） |
| 其它 Router 参数 | `--router-temperature 0`、`--cache-threshold 0.3`、`--balance-abs 2`、`--balance-rel 1.5`、`--kv-penalty-host/external/disk 0.8/0.6/0.3` |
| 回放 | agentinfer，`trace_same_agent_gap_scale=0.002`、`sample_seed=0` |
| `replay.interval_mode` | **lognormal（p50=1.0 s / p95=7.20 s / p99=16.31 s，σ=1.2）** |
| `replay.prompt_transport` / 容差 | **chat_text** / `prompt_calibration_tolerance_tokens=0` |
| `backend.endpoint` | `/v1/chat/completions` |


## 完整配置（可逐字复现）

### 硬件与软件

| 项 | 值 |
|---|---|
| 集群 | H20 集群，SLURM `--partition=h20`，本次用 8 张卡（每节点 8× H20） |
| Router 二进制 | `bin/router-main-9605956f`（main `9605956f`），sha256 `c330911d49cedc86…` |
| 引擎 | vLLM 0.33 runtime：`kv-unified-vllm033-runtime/bin/vllm serve` |
| KV 传输 | NIXL（`PYTHONPATH=<nixl152>`），`--kv-transfer-config {kv_connector: NixlConnector, kv_role: kv_producer\|kv_consumer, kv_connector_extra_config: {backends: [UCX]}}` |
| 模型 | `Qwen3-4B-Instruct-2507`，`--served-model-name qwen3`，`--dtype bfloat16` |
| 端口分配 | `PORT_SEARCH_START = 50000 + (jobid % 30) * 400`，取 8 个端口步长 20；Router = 末端口+110，Prometheus = 末端口+111 |

### Router 完整命令行（四组实验除策略与拓扑外完全相同）

```bash
$BIN --host 127.0.0.1 --port $ROUTER_PORT \
  --prometheus-host 127.0.0.1 --prometheus-port $((ROUTER_PORT+1)) \
  --vllm-pd-disaggregation --kv-connector nixl \
  --policy <POLICY> --prefill-policy <POLICY> --decode-policy <POLICY> \
  --worker-startup-check-interval 1 --request-timeout-secs 3600 \
  --router-temperature 0 --cache-threshold 0.3 \
  --balance-abs-threshold 2 --balance-rel-threshold 1.5 \
  --kv-penalty-host 0.8 --kv-penalty-external 0.6 --kv-penalty-disk 0.3 \
  --prefill http://127.0.0.1:<P1> --prefill http://127.0.0.1:<P2> --prefill http://127.0.0.1:<P3> --prefill http://127.0.0.1:<P4> \
  --decode http://127.0.0.1:<D1> --decode http://127.0.0.1:<D2> --decode http://127.0.0.1:<D3> --decode http://127.0.0.1:<D4>
```

`<POLICY>` 取 `round_robin` / `cache_aware` / `kv_aware`；同一臂的三个 policy 参数必须相同（P 与 D 同策略）。

### 引擎完整命令行（Prefill 与 Decode 同一模板，仅 `kv_role` 不同）

```bash
$VLLM serve $MODEL --served-model-name qwen3 --host 127.0.0.1 --port $WORKER_PORT \
  --max-model-len 163840 --max-num-seqs 8 \
  --max-num-batched-tokens 32768 --dtype bfloat16 \
  --gpu-memory-utilization 0.80 --kv-cache-memory-bytes 27917287424 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --enable-request-id-headers \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:$((WORKER_PORT+1))", "replay_endpoint": "tcp://127.0.0.1:$((WORKER_PORT+2))", "buffer_steps": 10000}' \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer|kv_consumer", "kv_connector_extra_config": {"backends": ["UCX"]}}'
```

引擎环境变量：`CUDA_VISIBLE_DEVICES=<gpu>`、`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`、`VLLM_SERVER_DEV_MODE=1`、`VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1`、`VLLM_NIXL_SIDE_CHANNEL_PORT=<worker_port+3>`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`TOKENIZERS_PARALLELISM=false`、`OMP_NUM_THREADS=4`。

## swe-sb1（requests/臂 = 247）

### 吞吐与复用

| 臂 | 请求 | span (s) | 吞吐 out tok/s | 吞吐总 tok/s | 命中率 | 新算 Prefill | 倾斜 max/avg |
|---|---|---|---|---|---|---|---|
| round_robin | 247 | 1,207 | 148.5 | 12,496 | 47.12% | 15,768,198 | 1.13 |
| cache_aware | 247 | 1,103 | 162.6 | 13,676 | 52.85% | 14,058,070 | 1.16 |
| kv_aware | 247 | 990 | 181.2 | 15,240 | 57.57% | 12,652,598 | 1.46 |

### 延迟分位数

| 指标 | 统计 | round_robin | cache_aware | kv_aware |
|---|---|---|---|---|
| TTFT (s) | mean | 9.60 | 8.24 | 6.84 |
| TTFT (s) | p50 | 5.49 | 5.00 | 3.76 |
| TTFT (s) | p95 | 30.43 | 28.44 | 26.93 |
| TTFT (s) | p99 | 36.74 | 35.01 | 32.79 |
| TPOT (ms) | mean | 13.51 | 11.40 | 11.72 |
| TPOT (ms) | p50 | 12.57 | 10.34 | 11.37 |
| TPOT (ms) | p95 | 25.83 | 19.07 | 20.05 |
| TPOT (ms) | p99 | 28.52 | 24.33 | 22.96 |
| 请求延迟（单请求端到端） (s) | mean | 20.21 | 17.05 | 15.66 |
| 请求延迟（单请求端到端） (s) | p50 | 12.79 | 10.72 | 9.40 |
| 请求延迟（单请求端到端） (s) | p95 | 58.40 | 52.12 | 54.27 |
| 请求延迟（单请求端到端） (s) | p99 | 125.19 | 88.83 | 81.78 |
| JCT（会话） (s) | mean | 455.65 | 390.63 | 361.86 |
| JCT（会话） (s) | p50 | 342.11 | 291.51 | 267.79 |
| JCT（会话） (s) | p95 | 1,118.59 | 1,006.07 | 934.83 |
| JCT（会话） (s) | p99 | 1,189.60 | 1,083.78 | 978.97 |

### kv_aware 相对基线（✅ = 更好）

| 指标 | 统计 | vs round_robin | vs cache_aware |
|---|---|---|---|
| TTFT (s) | mean | -28.8% ✅ | -17.0% ✅ |
| TTFT (s) | p50 | -31.5% ✅ | -24.8% ✅ |
| TTFT (s) | p95 | -11.5% ✅ | -5.3% ✅ |
| TTFT (s) | p99 | -10.7% ✅ | -6.4% ✅ |
| TPOT (ms) | mean | -13.3% ✅ | +2.8% ⚠️ |
| TPOT (ms) | p50 | -9.5% ✅ | +10.0% ⚠️ |
| TPOT (ms) | p95 | -22.4% ✅ | +5.2% ⚠️ |
| TPOT (ms) | p99 | -19.5% ✅ | -5.6% ✅ |
| 请求延迟（单请求端到端） (s) | mean | -22.5% ✅ | -8.2% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | -26.5% ✅ | -12.3% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | -7.1% ✅ | +4.1% ⚠️ |
| 请求延迟（单请求端到端） (s) | p99 | -34.7% ✅ | -7.9% ✅ |
| JCT（会话） (s) | mean | -20.6% ✅ | -7.4% ✅ |
| JCT（会话） (s) | p50 | -21.7% ✅ | -8.1% ✅ |
| JCT（会话） (s) | p95 | -16.4% ✅ | -7.1% ✅ |
| JCT（会话） (s) | p99 | -17.7% ✅ | -9.7% ✅ |
| 吞吐（输出 tok/s） | — | +22.0% ✅ | +11.4% ✅ |
| 吞吐（总 tok/s） | — | +22.0% ✅ | +11.4% ✅ |
| 新算 Prefill | — | -19.8% ✅ | -10.0% ✅ |
| 命中率 | — | +10.45 pt | +4.72 pt |

### cache_aware 相对 round_robin（✅ = 更好）

| 指标 | 统计 | vs round_robin |
|---|---|---|
| TTFT (s) | mean | -14.1% ✅ |
| TTFT (s) | p50 | -8.9% ✅ |
| TTFT (s) | p95 | -6.5% ✅ |
| TTFT (s) | p99 | -4.7% ✅ |
| TPOT (ms) | mean | -15.7% ✅ |
| TPOT (ms) | p50 | -17.7% ✅ |
| TPOT (ms) | p95 | -26.2% ✅ |
| TPOT (ms) | p99 | -14.7% ✅ |
| 请求延迟（单请求端到端） (s) | mean | -15.6% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | -16.2% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | -10.8% ✅ |
| 请求延迟（单请求端到端） (s) | p99 | -29.0% ✅ |
| JCT（会话） (s) | mean | -14.3% ✅ |
| JCT（会话） (s) | p50 | -14.8% ✅ |
| JCT（会话） (s) | p95 | -10.1% ✅ |
| JCT（会话） (s) | p99 | -8.9% ✅ |
| 吞吐（输出 tok/s） | — | +9.5% ✅ |
| 吞吐（总 tok/s） | — | +9.4% ✅ |
| 新算 Prefill | — | -10.8% ✅ |
| 命中率 | — | +5.73 pt |

## swe-sb2（requests/臂 = 247）

### 吞吐与复用

| 臂 | 请求 | span (s) | 吞吐 out tok/s | 吞吐总 tok/s | 命中率 | 新算 Prefill | 倾斜 max/avg |
|---|---|---|---|---|---|---|---|
| round_robin | 247 | 1,165 | 153.9 | 12,948 | 49.44% | 15,075,286 | 1.07 |
| cache_aware | 247 | 1,076 | 166.8 | 14,029 | 54.46% | 13,579,030 | 1.14 |
| kv_aware | 247 | 914 | 196.2 | 16,504 | 58.92% | 12,249,926 | 1.51 |

### 延迟分位数

| 指标 | 统计 | round_robin | cache_aware | kv_aware |
|---|---|---|---|---|
| TTFT (s) | mean | 9.18 | 8.01 | 6.59 |
| TTFT (s) | p50 | 5.39 | 4.34 | 3.50 |
| TTFT (s) | p95 | 29.42 | 26.31 | 23.25 |
| TTFT (s) | p99 | 35.89 | 33.06 | 29.35 |
| TPOT (ms) | mean | 13.93 | 11.61 | 11.43 |
| TPOT (ms) | p50 | 12.57 | 11.14 | 11.20 |
| TPOT (ms) | p95 | 26.88 | 21.39 | 19.13 |
| TPOT (ms) | p99 | 29.05 | 25.48 | 23.77 |
| 请求延迟（单请求端到端） (s) | mean | 20.51 | 16.52 | 15.48 |
| 请求延迟（单请求端到端） (s) | p50 | 11.31 | 10.24 | 9.10 |
| 请求延迟（单请求端到端） (s) | p95 | 58.64 | 51.67 | 50.89 |
| 请求延迟（单请求端到端） (s) | p99 | 113.72 | 83.53 | 82.79 |
| JCT（会话） (s) | mean | 461.75 | 379.48 | 358.16 |
| JCT（会话） (s) | p50 | 351.00 | 270.43 | 303.08 |
| JCT（会话） (s) | p95 | 1,088.59 | 999.84 | 857.53 |
| JCT（会话） (s) | p99 | 1,149.90 | 1,060.35 | 902.82 |

### kv_aware 相对基线（✅ = 更好）

| 指标 | 统计 | vs round_robin | vs cache_aware |
|---|---|---|---|
| TTFT (s) | mean | -28.2% ✅ | -17.6% ✅ |
| TTFT (s) | p50 | -35.0% ✅ | -19.2% ✅ |
| TTFT (s) | p95 | -21.0% ✅ | -11.6% ✅ |
| TTFT (s) | p99 | -18.2% ✅ | -11.2% ✅ |
| TPOT (ms) | mean | -17.9% ✅ | -1.5% ✅ |
| TPOT (ms) | p50 | -10.8% ✅ | +0.6% ≈ |
| TPOT (ms) | p95 | -28.9% ✅ | -10.6% ✅ |
| TPOT (ms) | p99 | -18.2% ✅ | -6.7% ✅ |
| 请求延迟（单请求端到端） (s) | mean | -24.5% ✅ | -6.3% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | -19.5% ✅ | -11.2% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | -13.2% ✅ | -1.5% ✅ |
| 请求延迟（单请求端到端） (s) | p99 | -27.2% ✅ | -0.9% ≈ |
| JCT（会话） (s) | mean | -22.4% ✅ | -5.6% ✅ |
| JCT（会话） (s) | p50 | -13.7% ✅ | +12.1% ⚠️ |
| JCT（会话） (s) | p95 | -21.2% ✅ | -14.2% ✅ |
| JCT（会话） (s) | p99 | -21.5% ✅ | -14.9% ✅ |
| 吞吐（输出 tok/s） | — | +27.5% ✅ | +17.6% ✅ |
| 吞吐（总 tok/s） | — | +27.5% ✅ | +17.6% ✅ |
| 新算 Prefill | — | -18.7% ✅ | -9.8% ✅ |
| 命中率 | — | +9.48 pt | +4.46 pt |

### cache_aware 相对 round_robin（✅ = 更好）

| 指标 | 统计 | vs round_robin |
|---|---|---|
| TTFT (s) | mean | -12.8% ✅ |
| TTFT (s) | p50 | -19.5% ✅ |
| TTFT (s) | p95 | -10.6% ✅ |
| TTFT (s) | p99 | -7.9% ✅ |
| TPOT (ms) | mean | -16.6% ✅ |
| TPOT (ms) | p50 | -11.4% ✅ |
| TPOT (ms) | p95 | -20.4% ✅ |
| TPOT (ms) | p99 | -12.3% ✅ |
| 请求延迟（单请求端到端） (s) | mean | -19.5% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | -9.4% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | -11.9% ✅ |
| 请求延迟（单请求端到端） (s) | p99 | -26.6% ✅ |
| JCT（会话） (s) | mean | -17.8% ✅ |
| JCT（会话） (s) | p50 | -23.0% ✅ |
| JCT（会话） (s) | p95 | -8.2% ✅ |
| JCT（会话） (s) | p99 | -7.8% ✅ |
| 吞吐（输出 tok/s） | — | +8.4% ✅ |
| 吞吐（总 tok/s） | — | +8.3% ✅ |
| 新算 Prefill | — | -9.9% ✅ |
| 命中率 | — | +5.02 pt |

## 2 轮平均

| 臂 | 吞吐 out tok/s | 吞吐总 tok/s | 命中率 | 新算 Prefill | TTFT mean (s) | TPOT mean (ms) | 请求延迟 mean (s) | JCT mean (s) | JCT p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| round_robin | 151.2 | 12,722 | 48.28% | 15,421,742 | 9.39 | 13.72 | 20.36 | 458.70 | 1,103.59 |
| cache_aware | 164.7 | 13,852 | 53.66% | 13,818,550 | 8.12 | 11.50 | 16.78 | 385.05 | 1,002.96 |
| kv_aware | 188.7 | 15,872 | 58.25% | 12,451,262 | 6.72 | 11.57 | 15.57 | 360.01 | 896.18 |

### 吞吐 vs TPOT 取舍

`N̄` 是 decode 侧**平均同时运行的序列数**，按 token-时间积分严格计算：

`N̄ = Σ_i(out_i · TPOT_i) / T`（`out_i` 为该请求输出 token 数，`T` 为该臂墙钟）。

这不是「吞吐 × 平均 TPOT」那种近似乘法，也不需要假设 TPOT 与单请求输出长度无关。
但它是**由客户端时间反推**的推断量（我们只有每臂前/后两点 Prometheus 快照，取不到引擎侧
`num_requests_running` 时间序列），不是引擎自报值。

`round_robin` 完全不复用前缀，系统卡在 Prefill，decode 侧 `N̄` 明显偏低；cache/kv 拆掉
Prefill 瓶颈后 decode 填得更满，`N̄` 上升、TPOT 随之上升，但总吞吐也上去了。
这是吞吐与单 token 延迟的取舍，不是调度缺陷。

| 臂 | 吞吐 out tok/s | TPOT mean (ms) | N̄ = Σout·TPOT/T | 相对 round_robin 吞吐 | 相对 round_robin TPOT |
|---|---|---|---|---|---|
| round_robin | 151.2 | 13.7 | 2.29 | +0.0% ≈ | +0.0% ≈ |
| cache_aware | 164.7 | 11.5 | 1.97 | +8.9% ✅ | -16.1% ✅ |
| kv_aware | 188.7 | 11.6 | 2.30 | +24.8% ✅ | -15.6% ✅ |

### kv_aware 相对基线的平均优化比例

| 指标 | 统计 | vs round_robin | vs cache_aware |
|---|---|---|---|
| TTFT (s) | mean | -28.5% ✅ | -17.3% ✅ |
| TTFT (s) | p50 | -33.2% ✅ | -22.2% ✅ |
| TTFT (s) | p95 | -16.2% ✅ | -8.4% ✅ |
| TTFT (s) | p99 | -14.4% ✅ | -8.7% ✅ |
| TPOT (ms) | mean | -15.6% ✅ | +0.6% ≈ |
| TPOT (ms) | p50 | -10.2% ✅ | +5.1% ⚠️ |
| TPOT (ms) | p95 | -25.7% ✅ | -3.2% ✅ |
| TPOT (ms) | p99 | -18.8% ✅ | -6.2% ✅ |
| 请求延迟（单请求端到端） (s) | mean | -23.5% ✅ | -7.2% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | -23.3% ✅ | -11.8% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | -10.2% ✅ | +1.3% ⚠️ |
| 请求延迟（单请求端到端） (s) | p99 | -31.1% ✅ | -4.5% ✅ |
| JCT（会话） (s) | mean | -21.5% ✅ | -6.5% ✅ |
| JCT（会话） (s) | p50 | -17.6% ✅ | +1.6% ⚠️ |
| JCT（会话） (s) | p95 | -18.8% ✅ | -10.6% ✅ |
| JCT（会话） (s) | p99 | -19.6% ✅ | -12.2% ✅ |
| 吞吐（输出 tok/s） | — | +24.8% ✅ | +14.6% ✅ |
| 吞吐（总 tok/s） | — | +24.8% ✅ | +14.6% ✅ |
| 新算 Prefill | — | -19.3% ✅ | -9.9% ✅ |
| 命中率 | — | +9.97 pt | +4.59 pt |

### cache_aware 相对 round_robin 的平均优化比例

| 指标 | 统计 | vs round_robin |
|---|---|---|
| TTFT (s) | mean | -13.5% ✅ |
| TTFT (s) | p50 | -14.2% ✅ |
| TTFT (s) | p95 | -8.5% ✅ |
| TTFT (s) | p99 | -6.3% ✅ |
| TPOT (ms) | mean | -16.1% ✅ |
| TPOT (ms) | p50 | -14.6% ✅ |
| TPOT (ms) | p95 | -23.2% ✅ |
| TPOT (ms) | p99 | -13.5% ✅ |
| 请求延迟（单请求端到端） (s) | mean | -17.6% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | -13.0% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | -11.3% ✅ |
| 请求延迟（单请求端到端） (s) | p99 | -27.9% ✅ |
| JCT（会话） (s) | mean | -16.1% ✅ |
| JCT（会话） (s) | p50 | -18.9% ✅ |
| JCT（会话） (s) | p95 | -9.1% ✅ |
| JCT（会话） (s) | p99 | -8.4% ✅ |
| 吞吐（输出 tok/s） | — | +8.9% ✅ |
| 吞吐（总 tok/s） | — | +8.9% ✅ |
| 新算 Prefill | — | -10.4% ✅ |
| 命中率 | — | +5.38 pt |

