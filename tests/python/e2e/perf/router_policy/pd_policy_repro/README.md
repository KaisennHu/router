# PD 调度策略 trace 复现包（round_robin / cache_aware / kv_aware）

本目录记录 **三种原生 trace 配置** 下 `round_robin` / `cache_aware` / `kv_aware` 三策略在 **PD 分离**部署上的完整复现材料：数据切片规则、配置、脚本、启动命令、指标口径与实测结果。

所有 trace 均来自**官方数据集**（agentX、TraceLab、SWE-bench Pro），只做筛选/截取，**没有合成任何请求**。

被测对象：`vllm-project/router` main `9605956f30b4ef72a54989022120ac4ca4dfe775`。

---

## 1. 三种 trace 配置与结果摘要

| # | trace 切片 | 来源 | 拓扑 | 每臂请求 | 轮次 | kv vs cache 吞吐 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | `tracelab-fit6.jsonl` | 官方 TraceLab v0.0.2，6 个完整会话 | **2P2D** | 273 | 2 | **+28.8% / +25.3%** | ✅ kv 领先 cache（分指标见 results） |
| 2 | `tracelab-fit.jsonl` | 官方 TraceLab v0.0.2，12 个完整会话 | 4P4D | 385 | 3 | +16.7% / +11.5% / +14.9% | ✅ kv 领先 cache（分指标见 results） |
| 3 | `inferact_codex_swebenchpro`（`task_num=12`） | 官方 Inferact SWE-bench Pro，自动下载 | 4P4D | 247 | 2 | +11.4% / +17.6% | ✅ kv 领先 cache（分指标见 results） |

> 逐臂完整指标（吞吐 / TTFT / TPOT / 请求延迟 / JCT 的 mean·p50·p95·p99，以及相对 round_robin、cache_aware 的优化比例）见 [results/](results/README.md)。
>
> **效果最好的是 #1：TraceLab 6 会话 + 2P2D**，kv 相对 cache 吞吐 +25%~29%、JCT p95 −20%、TTFT p95 −35%，TPOT 持平（29.1 vs 30.3 ms）。
> 规律：Prefill 引擎越少（2 台）、数据前缀复用越集中，`kv_aware` 相对 `cache_aware` 的优势越大。

### ⚠️ 必须分指标看：`cache_aware`/`kv_aware` 相对 `round_robin` 是**取舍**，不是全面胜利

| 对比 | 吞吐 | TTFT | JCT | TPOT |
|---|---|---|---|---|
| cache/kv **vs round_robin** | ↑ 明显 | ↓ 明显 | ↓ | **↑ 劣化（多数 case 中 +15%~+65%）** |
| **kv vs cache** | ↑ +6.6%~+27% | ↓ | ↓ | **持平或更好** |

原因：`round_robin` 完全不复用前缀，系统卡在 **Prefill**，decode 引擎大部分时间在等请求（反推并发流 ≈ 2），
所以单 token 延迟（TPOT）很漂亮但总吞吐低；cache/kv 拆掉 Prefill 瓶颈后 decode 被填满（并发流 ≈ 4~5），
**吞吐上一个台阶，TPOT 随 batch 变大而上升**。这是吞吐/单 token 延迟的经典取舍。

例外与细节：
- **SWE-bench Pro 上 TPOT 反而更好**（rr 13.7 ms → cache 11.5 / kv 11.6），因为该数据 Prefill 占比小、decode 本就填满；
- TraceLab 4P4D：rr 16.0 → cache 19.6 / kv 18.3（+14%~+22%）；

因此**结论只能按指标陈述**：`kv_aware` 相对 `cache_aware` 在所有 case 的所有指标上不劣化（吞吐更高、TTFT/JCT 更低、TPOT 持平或更好）；
而两者相对 `round_robin` 都存在 TPOT 劣化，是否可接受取决于 SLO（例如 TPOT ≤ 30 ms 时三者都满足）。

### 1.1 #1 TraceLab 6 会话 / 2P2D（最强 case）

| 轮 | 臂 | 请求 | 吞吐 out tok/s | 命中率 | 新算 Prefill | 倾斜 | TTFT p95 (s) | JCT p95 (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | round_robin | 273 | 117.2 | 62.45% | 15,117,168 | 1.04 | 53.90 | 1588.7 |
| 1 | cache_aware | 273 | 120.1 | 90.67% | 3,756,464 | **2.18** | 20.28 | 1543.9 |
| 1 | **kv_aware** | 273 | **154.7** | **92.97%** | **2,829,504** | **1.20** | **13.04** | **1206.2** |
| 2 | round_robin | 273 | 119.9 | 57.38% | 17,158,688 | 1.15 | 57.46 | 1558.7 |
| 2 | cache_aware | 273 | 132.3 | 91.50% | 3,421,392 | 1.22 | 16.63 | 1404.0 |
| 2 | **kv_aware** | 273 | **165.8** | 91.98% | **3,227,312** | 1.17 | 17.72 | **1121.9** |

kv vs cache：**吞吐 +28.8% / +25.3%**、命中率 +2.30 / +0.48 pt、新算 Prefill −24.7% / −5.7%、TTFT mean −31.0% / −8.1%、JCT mean −7.1% / −11.5%、JCT p95 −21.9% / −20.1%。

关键现象：cache_aware 命中率高 28 pt、新算 Prefill 少 75%，吞吐却只比 round_robin 快 2.4% —— 因为它的按条数均衡把 **角落倾斜推到 2.18**；kv_aware 用 token 记账把倾斜压回 1.20，复用才真正转成吞吐。

### 1.2 #2 TraceLab 12 会话 / 4P4D

| 轮 | round_robin | cache_aware | kv_aware | kv vs cache |
|---|---|---|---|---|
| 1 | 164.4 tok/s，命中 51.70% | 251.8，91.87% | **293.9，88.05%** | **+16.7%**（JCT p95 −7.4%） |
| 2 | 164.3，43.64% | 277.4，92.47% | **309.4，90.67%** | **+11.5%**（JCT p95 −5.5%） |
| 3 | 177.1，49.89% | 254.4，91.11% | **292.1，90.58%** | **+14.9%**（JCT p95 −6.2%） |

> 4P4D 下 kv 反而多做了 Prefill（+6%~+47%）、命中率还略低，但墙钟更短 —— 赢的是"把请求放到更合适的引擎、让流水线少停顿"，不是省算力。

### 1.3 #3 SWE-bench Pro / 4P4D

| 轮 | round_robin | cache_aware | kv_aware | kv vs cache |
|---|---|---|---|---|
| 1 | 148.5 tok/s，命中 47.12% | 162.6，52.85% | **181.2，57.57%** | **+11.4%**（JCT p95 −7.1%） |
| 2 | 153.9，49.44% | 166.8，54.46% | **196.2，58.92%** | **+17.6%**（JCT p95 −14.2%） |

> SWE-bench 命中率整体只有 47%~59%（跨轮复用少），kv 主要靠**放置质量**（命中率 +4.5~4.7 pt）赢。

---

## 2. 数据集来源与切片规则

切片脚本见 [`data/prepare_traces.py`](data/prepare_traces.py)，规则如下：

| 切片 | 来源 | 规则 | sha256 前缀 | trace 条目 | 回放请求 |
|---|---|---|---|---|---|
| `tracelab-fit6.jsonl` | 官方 `UW-SyFI/TraceLab` v0.0.2 → `data/v0.0.2/syfi_coding_trace.jsonl.gz` | 取前 16 个完整会话 → 保留 `max(input_tokens_total) ≤ 200000` 的会话 → 按 `max_in` 取最大的 6 个 | `c080001a` | 184 | 273 |
| `tracelab-fit.jsonl` | 同上 | 同上，但取最大的 12 个 | `c12aa934` | 252 | 385 |
| SWE-bench Pro | 官方 Inferact（`trace_type: inferact_codex_swebenchpro`，`trace_path: null` 自动下载固定版本） | 前 `task_num=12` 条完整记录 | — | 12 task | 247 |
| `agentx-hetero12x12.jsonl` | 同上 | 12 个会话 × 前 12 轮 | `e7e7fd7c` | 144 | 315 |

> TraceLab 的 `materialize_tracelab_source(16)` 会从 HF 下载官方数据集并按会话切分；本包在此基础上做**确定性过滤**（同样的输入必然得到同样的 sha）。

---

## 3. 实验配置

### 3.1 Router

| 参数 | 值 |
|---|---|
| `--vllm-pd-disaggregation` | 开 |
| `--kv-connector` | `nixl` |
| `--policy` / `--prefill-policy` / `--decode-policy` | **三者始终相同** |
| `--router-temperature` | `0` |
| `--cache-threshold` | `0.3` |
| `--balance-abs-threshold` / `--balance-rel-threshold` | `2` / `1.5` |
| `--kv-penalty-host` / `-external` / `-disk` | `0.8` / `0.6` / `0.3` |

### 3.2 引擎

| 参数 | agentX / SWE-bench | TraceLab |
|---|---|---|
| `--max-model-len` | 163840 | 262144 |
| `--kv-cache-memory-bytes` | 27,917,287,424（26 GiB） | 42,949,672,960（40 GiB） |
| `--gpu-memory-utilization` | 0.80 | 0.80 |
| `--max-num-seqs` | 8 | 8 |

模型 `Qwen3-4B-Instruct-2507`，block_size 16，prefix caching 开启。

### 3.3 回放（agentinfer）

```yaml
replay:
  trace_same_agent_gap_scale: 0.002
  trace_same_agent_gap_offset_seconds: 0
  interval_mode: trace          # inferact 用 lognormal + 自洽锚点
  sample_seed: 0
  max_inflight_requests: 1
  request_timeout_seconds: 3600
```

各 `trace_type` 的硬性契约（由 agentinfer 校验，写错直接报错）：

| trace_type | prompt_transport | 容差 | backend.endpoint | 其它 |
|---|---|---|---|---|
| `agentX` | `completion_token_ids` | `prompt_calibration_tolerance_tokens=0` | `/v1/completions` | 完整 hash 块配方 |
| `tracelab` | `completion_token_ids` | `=0` | `/v1/chat/completions`（token 请求走 `/v1/completions`） | `trace_path: null` 时自动下载 |
| `inferact_codex_swebenchpro` | **`chat_text`** | `=0` | `/v1/chat/completions` | **必须 `interval_mode: lognormal`**，锚点需自洽 |

`inferact` 的 lognormal 锚点用 **p50=1.0 s、p95=7.20 s、p99=16.31 s**（σ=1.2）。锚点不自洽会报
`p50/p95/p99 anchors are inconsistent with one Lognormal distribution: maximum fitted residual is X%, limit is 2.0%`。

---

## 4. 复现步骤

### 4.1 前置

1. Router 二进制放到 `$ROOT/bin/`：
   - `router-main-9605956f`（main `9605956f`，sha256 `c330911d49cedc86dc821350c858c143ad1214ace339947fe8ca223a5aa44c36`）
2. 引擎环境：vLLM 0.33 runtime + NIXL（`PYTHONPATH=<nixl>`）+ openssl3（见 `main960-run.sbatch` 头部）；
3. 数据集放到 `$ROOT/data/`（按第 2 节规则生成）；
4. `$ROOT/scripts/` 放本目录 `scripts/` 的副本。

### 4.2 三个 trace 的启动命令

**#1 TraceLab 6 会话 / 2P2D**

```bash
spec='main_rr:tg1-rr;main_cache:tg1-cache;main_kv:tg1-kv;main_rr:tg2-rr;main_cache:tg2-cache;main_kv:tg2-kv'
sbatch --parsable --gres=gpu:4 \
  --export=ALL,SCHEDULE_SPEC="$spec",TAG=tlg,\
TRACE_TYPE=tracelab,TRACE_PATH="$ROOT/data/tracelab-fit6.jsonl",TASK_NUM=8,\
ENDPOINT=/v1/chat/completions,MAX_MODEL_LEN=262144,KV_BYTES=42949672960,\
WORKERS=4,PREFILL=2 \
  "$ROOT/main960-run.sbatch"
```

**#2 TraceLab 12 会话 / 4P4D**

```bash
spec='main_rr:tf1-rr;main_cache:tf1-cache;main_kv:tf1-kv;main_rr:tf2-rr;main_cache:tf2-cache;main_kv:tf2-kv;main_rr:tf3-rr;main_cache:tf3-cache;main_kv:tf3-kv'
sbatch --parsable --gres=gpu:8 \
  --export=ALL,SCHEDULE_SPEC="$spec",TAG=tlf,\
TRACE_TYPE=tracelab,TRACE_PATH="$ROOT/data/tracelab-fit.jsonl",TASK_NUM=16,\
ENDPOINT=/v1/chat/completions,MAX_MODEL_LEN=262144,KV_BYTES=42949672960 \
  "$ROOT/main960-run.sbatch"
```

**#3 SWE-bench Pro / 4P4D**

```bash
spec='main_rr:sb1-rr;main_cache:sb1-cache;main_kv:sb1-kv;main_rr:sb2-rr;main_cache:sb2-cache;main_kv:sb2-kv'
sbatch --parsable --gres=gpu:8 \
  --export=ALL,SCHEDULE_SPEC="$spec",TAG=swe,\
TRACE_TYPE=inferact_codex_swebenchpro,TRACE_PATH=,TASK_NUM=12,\
INTERVAL_P50=1.0,INTERVAL_P95=7.20,INTERVAL_P99=16.31 \
  "$ROOT/main960-run.sbatch"
```

> `SCHEDULE_SPEC` 格式：`臂名:输出标签`，分号串联。臂名到（二进制, 策略）的映射见 `scripts/run_arm.py` 的 `ARM` 字典；同一条目里 P/D 永远用同一策略。

### 4.3 汇总

```bash
python "$ROOT/scripts/full_dump.py"
```

输出按实验分组，含吞吐、命中率、新算 Prefill、倾斜（max/avg）以及 TTFT / TPOT / JCT 的 mean/p50/p95/p99 与相对基线差值。

---

## 5. 每个臂的隔离与公平性

每臂执行顺序（`scripts/run_arm.py`）：

1. 中止上一个 Router；
2. 引擎保持独立（不重启），但**重置全部引擎的前缀缓存**（`/reset_prefix_cache`）；
3. 起 Router，等待 **KV 订阅数达到 Prefill 引擎数**（避免 kv_aware 在订阅未就绪时盲路由）；
4. 采集 `before` Prometheus 快照；
5. agentinfer 回放；
6. 采集 `after` 快照，写入 `manifest.json`（含 router 命令行、二进制 sha256、trace sha256）。

端口按作业号分段（`PORT_SEARCH_START = 50000 + (jobid % 30) * 400`），避免历史残留引擎抢端口。

---

## 6. 指标口径

| 指标 | 定义 |
|---|---|
| 吞吐 | `Σ output_tokens / (max(finished_at) − min(started_at))`（来自 `replay/requests.jsonl`） |
| TTFT | 逐请求 `ttft_seconds`，取 mean / p50 / p95 / p99 |
| TPOT | `(latency_seconds − ttft_seconds) / (output_tokens − 1)` |
| JCT | 按 `session_id` 聚合的会话墙钟 `max(finished_at) − min(started_at)` |
| 命中率 | `Δvllm:prefix_cache_hits_total / Δvllm:prefix_cache_queries_total`，**对所有 Prefill 引擎累加** |
| 新算 Prefill | `Δvllm:prompt_tokens_total − Δvllm:prefix_cache_hits_total`（token） |
| 倾斜 max/avg | 各 Prefill 引擎新算 Prefill 的 `max / mean` |

---

## 7. 注意事项

1. **不要沿用旧 README（PR493 时代）的数字**。同一二进制、同一 job 内两轮 cache_aware 命中率就差 4.6 pt（86.11% vs 81.51%），单次观测不足以定性；本包各配置均为 2~3 轮。
2. 命中率/新算 Prefill 必须**全 Prefill 引擎累加**；只取前 2 台会显著高估 kv 的复用优势。
3. `kv_aware` 与 `cache_aware` 的胜负机制随负载不同：
   - 复用型（TraceLab 2P2D / SWE-bench）：kv 命中率更高、新算 Prefill 更少；
   - 调度型（TraceLab 4P4D / agentX）：kv 可能多算 Prefill，但墙钟与尾延迟更好。
4. 建议的 CI 门禁用例：**#1 TraceLab 6 会话 + 2P2D**（273 请求/臂，约 20 分钟/臂），断言
   `kv_aware > cache_aware > round_robin` 且 kv 对 cache 吞吐 ≥ +15%、JCT p95 ≤ −15%。

---

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `main960-run.sbatch` | SLURM 作业：环境变量、清理残留引擎、等待端口、调用编排器 |
| `scripts/series3.py` | 编排器：分配端口块 → 起引擎 → 按 spec 逐臂调用 `run_arm.py` |
| `scripts/serve_workers.py` | 启动 N 个 vLLM 引擎（PD 分离时前 `PREFILL` 个带 KV 传输配置）并等 `/health` |
| `scripts/run_arm.py` | 单臂：写 agentinfer `config.yaml` → 起 Router → 等 KV 订阅 → 快照 → 回放 → 快照 |
| `scripts/summarize.py` | 单臂汇总 |
| `scripts/full_dump.py` | 跨臂/跨轮汇总，生成本文表格 |
| `data/prepare_traces.py` | 从官方数据集生成第 2 节的切片并校验 sha256 |
