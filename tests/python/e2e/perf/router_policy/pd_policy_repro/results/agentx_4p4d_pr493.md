# agentX 12 会话 / 4P4D（PR493 二进制历史对照）

## 配置

| 项 | 值 |
|---|---|
| trace | `agentx-hetero12x12.jsonl`（官方 agentX，12 会话 × 12 轮，sha256 `e7e7fd7c…`） |
| 拓扑 | 4P4D（WORKERS=8，PREFILL=4） |
| 引擎 | `--max-model-len 163840`，`--kv-cache-memory-bytes 27917287424`（26 GiB） |
| 引擎并发 | `--max-num-seqs 8`、`--gpu-memory-utilization 0.80`、`--block-size 16`、`--enable-prefix-caching`、`--prefix-caching-hash-algo sha256_cbor` |
| `experiment.task_num` = `experiment.max_concurrency` | **12** |
| `replay.max_inflight_requests` | **1**（每个会话同一时刻最多 1 个在途请求，故实际并发≈会话数） |
| `experiment.task_timeout_seconds` / `run_timeout_seconds` | 3600 / 10800 |
| 每臂请求 | 315 |
| 轮次 | 2 |
| Router | main `9605956f`（`--policy` / `--prefill-policy` / `--decode-policy` 三者相同） |
| 其它 Router 参数 | `--router-temperature 0`、`--cache-threshold 0.3`、`--balance-abs 2`、`--balance-rel 1.5`、`--kv-penalty-host/external/disk 0.8/0.6/0.3` |
| 回放 | agentinfer，`trace_same_agent_gap_scale=0.002`、`sample_seed=0` |
| `replay.interval_mode` | **trace** |
| `replay.prompt_transport` / 容差 | **completion_token_ids** / `prompt_calibration_tolerance_tokens=0` |
| `backend.endpoint` | `/v1/chat/completions`（token 请求仍走 `/v1/completions`） |


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

## het3-h1（requests/臂 = 315）

### 吞吐与复用

| 臂 | 请求 | span (s) | 吞吐 out tok/s | 吞吐总 tok/s | 命中率 | 新算 Prefill | 倾斜 max/avg |
|---|---|---|---|---|---|---|---|
| cache_aware | 315 | 1,163 | 181.1 | 17,851 | 77.92% | 9,074,928 | 1.84 |
| kv_aware | 315 | 1,103 | 191.0 | 18,828 | 84.35% | 6,433,904 | 1.35 |

### 延迟分位数

| 指标 | 统计 | cache_aware | kv_aware |
|---|---|---|---|
| TTFT (s) | mean | 5.89 | 4.01 |
| TTFT (s) | p50 | 1.67 | 1.01 |
| TTFT (s) | p95 | 23.64 | 21.04 |
| TTFT (s) | p99 | 38.81 | 34.11 |
| TPOT (ms) | mean | 14.55 | 14.14 |
| TPOT (ms) | p50 | 14.38 | 12.43 |
| TPOT (ms) | p95 | 24.89 | 25.70 |
| TPOT (ms) | p99 | 27.14 | 28.34 |
| 请求延迟（单请求端到端） (s) | mean | 15.87 | 14.20 |
| 请求延迟（单请求端到端） (s) | p50 | 9.95 | 7.22 |
| 请求延迟（单请求端到端） (s) | p95 | 50.90 | 52.36 |
| 请求延迟（单请求端到端） (s) | p99 | 93.95 | 89.94 |
| JCT（会话） (s) | mean | 423.15 | 379.28 |
| JCT（会话） (s) | p50 | 289.52 | 292.38 |
| JCT（会话） (s) | p95 | 968.06 | 832.91 |
| JCT（会话） (s) | p99 | 1,124.17 | 1,048.85 |

### kv_aware 相对基线（✅ = 更好）

| 指标 | 统计 | vs round_robin | vs cache_aware |
|---|---|---|---|
| TTFT (s) | mean | — | -31.8% ✅ |
| TTFT (s) | p50 | — | -39.6% ✅ |
| TTFT (s) | p95 | — | -11.0% ✅ |
| TTFT (s) | p99 | — | -12.1% ✅ |
| TPOT (ms) | mean | — | -2.8% ✅ |
| TPOT (ms) | p50 | — | -13.5% ✅ |
| TPOT (ms) | p95 | — | +3.3% ⚠️ |
| TPOT (ms) | p99 | — | +4.4% ⚠️ |
| 请求延迟（单请求端到端） (s) | mean | — | -10.5% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | — | -27.5% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | — | +2.9% ⚠️ |
| 请求延迟（单请求端到端） (s) | p99 | — | -4.3% ✅ |
| JCT（会话） (s) | mean | — | -10.4% ✅ |
| JCT（会话） (s) | p50 | — | +1.0% ≈ |
| JCT（会话） (s) | p95 | — | -14.0% ✅ |
| JCT（会话） (s) | p99 | — | -6.7% ✅ |
| 吞吐（输出 tok/s） | — | — | +5.5% ✅ |
| 吞吐（总 tok/s） | — | — | +5.5% ✅ |
| 新算 Prefill | — | — | -29.1% ✅ |
| 命中率 | — | — | +6.43 pt |

## het3-h2（requests/臂 = 315）

### 吞吐与复用

| 臂 | 请求 | span (s) | 吞吐 out tok/s | 吞吐总 tok/s | 命中率 | 新算 Prefill | 倾斜 max/avg |
|---|---|---|---|---|---|---|---|
| cache_aware | 315 | 1,228 | 171.5 | 16,907 | 77.20% | 9,374,080 | 1.63 |
| kv_aware | 315 | 1,072 | 196.5 | 19,372 | 83.61% | 6,736,560 | 1.65 |

### 延迟分位数

| 指标 | 统计 | cache_aware | kv_aware |
|---|---|---|---|
| TTFT (s) | mean | 7.34 | 4.04 |
| TTFT (s) | p50 | 1.44 | 1.23 |
| TTFT (s) | p95 | 32.98 | 19.39 |
| TTFT (s) | p99 | 52.14 | 29.70 |
| TPOT (ms) | mean | 14.56 | 14.33 |
| TPOT (ms) | p50 | 13.68 | 12.19 |
| TPOT (ms) | p95 | 26.18 | 26.83 |
| TPOT (ms) | p99 | 29.57 | 30.05 |
| 请求延迟（单请求端到端） (s) | mean | 17.38 | 14.03 |
| 请求延迟（单请求端到端） (s) | p50 | 8.63 | 7.47 |
| 请求延迟（单请求端到端） (s) | p95 | 60.30 | 53.52 |
| 请求延迟（单请求端到端） (s) | p99 | 108.88 | 84.79 |
| JCT（会话） (s) | mean | 462.66 | 374.67 |
| JCT（会话） (s) | p50 | 380.52 | 268.81 |
| JCT（会话） (s) | p95 | 988.30 | 785.58 |
| JCT（会话） (s) | p99 | 1,180.19 | 1,014.63 |

### kv_aware 相对基线（✅ = 更好）

| 指标 | 统计 | vs round_robin | vs cache_aware |
|---|---|---|---|
| TTFT (s) | mean | — | -45.0% ✅ |
| TTFT (s) | p50 | — | -14.4% ✅ |
| TTFT (s) | p95 | — | -41.2% ✅ |
| TTFT (s) | p99 | — | -43.0% ✅ |
| TPOT (ms) | mean | — | -1.6% ✅ |
| TPOT (ms) | p50 | — | -10.9% ✅ |
| TPOT (ms) | p95 | — | +2.5% ⚠️ |
| TPOT (ms) | p99 | — | +1.6% ⚠️ |
| 请求延迟（单请求端到端） (s) | mean | — | -19.3% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | — | -13.4% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | — | -11.2% ✅ |
| 请求延迟（单请求端到端） (s) | p99 | — | -22.1% ✅ |
| JCT（会话） (s) | mean | — | -19.0% ✅ |
| JCT（会话） (s) | p50 | — | -29.4% ✅ |
| JCT（会话） (s) | p95 | — | -20.5% ✅ |
| JCT（会话） (s) | p99 | — | -14.0% ✅ |
| 吞吐（输出 tok/s） | — | — | +14.6% ✅ |
| 吞吐（总 tok/s） | — | — | +14.6% ✅ |
| 新算 Prefill | — | — | -28.1% ✅ |
| 命中率 | — | — | +6.41 pt |

## 2 轮平均

| 臂 | 吞吐 out tok/s | 吞吐总 tok/s | 命中率 | 新算 Prefill | TTFT mean (s) | TPOT mean (ms) | 请求延迟 mean (s) | JCT mean (s) | JCT p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| cache_aware | 176.3 | 17,379 | 77.56% | 9,224,504 | 6.62 | 14.56 | 16.62 | 442.91 | 978.18 |
| kv_aware | 193.8 | 19,100 | 83.98% | 6,585,232 | 4.03 | 14.24 | 14.11 | 376.98 | 809.25 |

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
| cache_aware | 176.3 | 14.6 | 2.65 | — | — |
| kv_aware | 193.8 | 14.2 | 2.92 | — | — |

### kv_aware 相对基线的平均优化比例

| 指标 | 统计 | vs round_robin | vs cache_aware |
|---|---|---|---|
| TTFT (s) | mean | — | -39.1% ✅ |
| TTFT (s) | p50 | — | -28.0% ✅ |
| TTFT (s) | p95 | — | -28.6% ✅ |
| TTFT (s) | p99 | — | -29.8% ✅ |
| TPOT (ms) | mean | — | -2.2% ✅ |
| TPOT (ms) | p50 | — | -12.2% ✅ |
| TPOT (ms) | p95 | — | +2.9% ⚠️ |
| TPOT (ms) | p99 | — | +3.0% ⚠️ |
| 请求延迟（单请求端到端） (s) | mean | — | -15.1% ✅ |
| 请求延迟（单请求端到端） (s) | p50 | — | -20.9% ✅ |
| 请求延迟（单请求端到端） (s) | p95 | — | -4.8% ✅ |
| 请求延迟（单请求端到端） (s) | p99 | — | -13.9% ✅ |
| JCT（会话） (s) | mean | — | -14.9% ✅ |
| JCT（会话） (s) | p50 | — | -16.2% ✅ |
| JCT（会话） (s) | p95 | — | -17.3% ✅ |
| JCT（会话） (s) | p99 | — | -10.5% ✅ |
| 吞吐（输出 tok/s） | — | — | +9.9% ✅ |
| 吞吐（总 tok/s） | — | — | +9.9% ✅ |
| 新算 Prefill | — | — | -28.6% ✅ |
| 命中率 | — | — | +6.42 pt |

