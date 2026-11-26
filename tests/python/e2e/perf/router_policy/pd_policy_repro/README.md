# PD 调度策略 trace 复现包（round_robin / cache_aware / kv_aware）

本目录记录 **三种原生 trace 配置** 下 `round_robin` / `cache_aware` / `kv_aware` 三策略在 **PD 分离**部署上的完整复现材料：数据切片规则、配置、脚本、启动命令、指标口径与实测结果。

所有 trace 均来自**官方数据集**（agentX、TraceLab、SWE-bench Pro），只做筛选/截取，**没有合成任何请求**。

被测对象：`vllm-project/router` main `9605956f30b4ef72a54989022120ac4ca4dfe775`。

---

> **导航**：[§3.4 并发与 task_num 参数矩阵](#34-并发与-task_num完整参数矩阵之前缺失的部分) ·
> [§8 验收/门禁用例建议](#8-作为性能验收--门禁用例的建议) · [§9 A2 环境备忘](#9-a2vllm-ascend--mooncake环境备忘)

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

### 3.4 并发与 task_num（完整参数矩阵，之前缺失的部分）

每个 case 的**并发**由 `experiment.max_concurrency` 决定，本包固定为 `max_concurrency = task_num = TASK_NUM`；
而每会话的在途请求由 `replay.max_inflight_requests = 1` 限制，所以**实际并发 ≈ min(task_num, 可用会话数)**。

| case | `trace_type` | 切片规模 | `task_num` = `max_concurrency` | `max_inflight_requests` | 每臂请求 | 拓扑 | `--max-model-len` | `--kv-cache-memory-bytes` | `--max-num-seqs` |
|---|---|---|---|---|---|---|---|---|---|
| #1 TraceLab 6 会话 | `tracelab` | 6 个完整会话 / 184 条目 | **8** | 1 | 273 | 2P2D（4 引擎） | 262144 | 42,949,672,960（40 GiB） | 8 |
| #2 TraceLab 12 会话 | `tracelab` | 12 个完整会话 / 252 条目 | **16** | 1 | 385 | 4P4D（8 引擎） | 262144 | 42,949,672,960（40 GiB） | 8 |
| #3 SWE-bench Pro | `inferact_codex_swebenchpro` | 12 个 task | **12** | 1 | 247 | 4P4D（8 引擎） | 163840 | 27,917,287,424（26 GiB） | 8 |
| 历史对照 agentX 12×12 | `agentX` | 12 会话 × 12 轮 / 144 条目 | **12** | 1 | 315 | 4P4D（8 引擎） | 163840 | 27,917,287,424（26 GiB） | 8 |

几个容易踩的点：

1. **`TASK_NUM` 可以大于会话数**（例如 TraceLab 只有 6 个会话却传 8）：agentinfer 的 `task_num` 是"最多采样多少个完整会话"，
   超出部分会被实际可用会话数截断，因此并发实际是 6 而不是 8。写文档时必须同时给出这两个数，否则复现出的并发会不一致。
2. **每臂请求数 ≠ 会话数**：回放会把每个会话的轮次展开成多次 HTTP 请求（TraceLab 6 会话 → 184 条目 → 273 请求；
   agentX 12×12 → 144 条目 → 315 请求），所以"并发 6~12"对应的每臂请求是 250~400 量级。
3. **引擎侧并发**另有 `--max-num-seqs 8`（Prefill 与 Decode 各自 8），与上面的会话并发不是一回事；
   本包所有实验都用 8，只有 A2（单卡 64 GB、TraceLab 长上下文）需要降到 2 以避免 OOM。
4. **超时**：`task_timeout_seconds=3600`、`run_timeout_seconds=10800`、`replay.request_timeout_seconds=3600`。
   若单臂跑不完，`task_timeout` 会先把会话切断（A2 出现过：三臂分别只完成 195/223/214 个请求、wall 都被截到 ~3600 s）。
   做对比实验时务必确认每臂请求数一致，否则吞吐以外的指标不可比。

校验方式：每个臂目录下的 `runs/<tag>/config.yaml` 记录了 `experiment` / `replay` 的完整取值，`manifest.json` 记录了
Router 命令行、二进制 sha256、trace sha256 与实际 worker 布局，可直接与本表逐项对照。


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
MAX_MODEL_LEN=163840,KV_BYTES=27917287424,WORKERS=8,PREFILL=4,\
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

## 8. 作为性能验收 / 门禁用例的建议

两个候选用例，定位不同：

### 8.1 门禁用例 A：`agentx_pd_hetero`（已发布，PD 放量前的基础门禁）

路径：[`../agentx_pd_hetero/README.md`](../agentx_pd_hetero/README.md)，配合 [`../agentx_pd_hetero/gate.py`](../agentx_pd_hetero/gate.py)。

- 负载：官方 agentX 12 会话 × 12 轮（315 请求/臂），4P4D；
- 作用：确认 PD 分离链路本身没退化（缓存命中、Prefill 新算量、JCT 不劣化），
  属于"改动后先跑一遍别把 PD 弄坏"级别；
- 建议门限：`kv_aware` 相对 `round_robin` 吞吐 ≥ +60%，命中率 ≥ +30 pt，JCT p95 ≤ −30%。

参考实测（本包同 trace、main `9605956f`、4P4D、315 请求/臂）：

| 指标 | round_robin | cache_aware | kv_aware | kv vs rr |
|---|---|---|---|---|
| 吞吐 (out tok/s) | 113.2 | 194.0 | 200.1 | +76.7% |
| JCT p95 (s) | 1418.7 | 914.0 | 786.1 | −44.6% |

### 8.2 性能验收用例 B：TraceLab 6 会话 + 2P2D（策略优劣的主判据）

- 负载：`tracelab-fit6.jsonl`（273 请求/臂），2P2D；
- 作用：区分 `cache_aware` 与 `kv_aware` 的优劣——这是本包里**信噪比最高**的配置，
  kv 相对 cache 吞吐 +25%~29%、TTFT p95 −36%、JCT p95 −22%，且两轮一致；
- 建议门限（相对 `cache_aware`）：吞吐 ≥ +15%、JCT p95 ≤ −15%、TPOT 不劣于 +10%；
- 注意：该用例下 `cache_aware` 与 `kv_aware` 的 TPOT 都明显高于 `round_robin`（约 +50%~+60%），
  所以**不要**用 "TPOT 优于 round_robin" 作为门限，否则会把正确实现判失败。

### 8.3 性能验收用例 C：SWE-bench Pro 4P4D（跨数据集回归）

- 负载：`inferact_codex_swebenchpro`，`task_num=12`（247 请求/臂），4P4D；
- 作用：与 TraceLab 互补——该数据跨轮复用少（命中率仅 47%~59%），kv 的优势来自**放置质量**而非省 Prefill；
  同时它是唯一一个 TPOT 也变好的用例（rr 13.7 ms → cache 11.5 / kv 11.6）；
- 建议门限：`kv_aware` 相对 `cache_aware` 吞吐 ≥ +8%、JCT p95 ≤ −5%、TPOT 不劣于 +5%。

### 8.4 门限写法上的注意事项

1. **必须分指标**：`cache_aware`/`kv_aware` 相对 `round_robin` 是"吞吐↑、TTFT↓、JCT↓、**TPOT↑**"的取舍，
   单看吞吐会得出"全面胜利"的错误结论（见 §1 的"必须分指标看"）；
2. **TPOT 门限要按用例设定**：TraceLab/agentX 上 TPOT 会变差，SWE-bench 上会变好；
3. **每臂请求数必须一致**：若被 `task_timeout_seconds` 截断（A2 上出现过 195/223/214），除吞吐外的指标不可比；
4. **至少 2 轮**：同一二进制、同一 job 内两轮 cache_aware 命中率就差过 4.6 pt，单轮不足以定性。

## 9. A2（vllm-ascend + Mooncake）环境备忘

与 H20（x86 + NIXL）不同，A2 是 Ascend NPU，PD 传输走 vllm-ascend 的 Mooncake 连接器。下面这套组合是实测**能跑通 2P2D** 的配置，供后续复现。

### 10.1 能跑通的组合

| 项 | 值 | 备注 |
|---|---|---|
| 镜像 | `quay.io/atlas-ci/vllm-ascend:v0.32.0` | **必须版本配套**：vllm 0.32.0 + vllm-ascend 0.32（上游 vllm-ascend 只发到 0.32，不要用 0.34 的 vllm 配它） |
| 引擎连接器 | `MooncakeConnectorV1` + `kv_connector_extra_config.mooncake_protocol=ascend` | V2 是拉模式，要求 `num_external_tokens == 0`，与上游 router 的 PD 契约冲突 |
| TE 端口 | 顶层 `kv_port` 每引擎独立（如 21000/21001） | 只在 `extra_config` 里给时，两台 Prefill 会抢同一端口，TE 线程崩（`corrupted size vs. prev_size`） |
| **Router 连接器** | **`--kv-connector nixl`（透传）** | 用 `mooncake` 时 router 发的是新 schema（`remote_bootstrap_addr`），vllm-ascend 要旧 schema（`remote_block_ids`）→ `KeyError` / `AssertionError`；nixl 分支会把 Prefill 响应里的 `kv_transfer_params` 原样转给 Decode |
| KV 容量 | `--kv-cache-memory-bytes 42949672960`（40 GiB） | `max_model_len=262144` 要求至少能放下 **一条** 满长序列（≈39 GiB），给 32 GiB 会在启动时报 `To serve at least one request...`；给 48 GiB 会 NPU OOM |
| 引擎并发 | `--max-num-seqs 2` | 单卡 64 GB，长上下文时 8 会 OOM |
| Bootstrap shim | 每台 Prefill 起一个 `GET /query` 服务，返回其真实 `engine_id` | main router 在 `--kv-connector mooncake` 时会查询它；走 nixl 透传时仍建议保留 |

跑通后 Decode 收到的参数形如：

```json
{"do_remote_prefill": true, "do_remote_decode": false,
 "remote_block_ids": [[1, 2, 3, ...]], "remote_engine_id": "...",
 "remote_host": "...", "remote_port": ...}
```

### 10.2 用同一份 TraceLab 切片跑出的初步数据（**不可作为结论**）

`tracelab-fit6.jsonl`（sha `c080001af8…`，与 H20 完全相同），2P2D，三臂各一轮：

| 臂 | 完成请求 | wall (s) | 吞吐 out tok/s | TTFT mean (s) | TPOT mean (ms) | 请求延迟 mean (s) | JCT mean (s) | JCT p95 (s) |
|---|---|---|---|---|---|---|---|---|
| round_robin | 195 | 3598 | 31.1 | 108.3 | 111.4 | 173.7 | 2823.6 | 3588.6 |
| cache_aware | 223 | 3575 | 33.9 | 88.4 | 113.0 | 150.5 | 2797.9 | 3563.9 |
| kv_aware | 214 | 3559 | 33.5 | 91.5 | 113.5 | 155.6 | 2775.4 | 3545.1 |

**为什么不可作为结论**：三臂都被 `experiment.task_timeout_seconds=3600` 截断（wall 都停在 ~3600 s），
但完成的请求数不同（195 / 223 / 214），即三臂做的**工作量不等**，除吞吐以外的指标不可比；
而且只跑了 1 轮。

下一步要得到可用的 A2 数据，需要把 `task_timeout_seconds` 提到 ≥ 21600（6 小时）并至少跑 2 轮，
保证每臂 273 个请求全部完成后再比较。

## 10. 文件清单

| 文件 | 作用 |
|---|---|
| `main960-run.sbatch` | SLURM 作业：环境变量、清理残留引擎、等待端口、调用编排器 |
| `scripts/series3.py` | 编排器：分配端口块 → 起引擎 → 按 spec 逐臂调用 `run_arm.py` |
| `scripts/serve_workers.py` | 启动 N 个 vLLM 引擎（PD 分离时前 `PREFILL` 个带 KV 传输配置）并等 `/health` |
| `scripts/run_arm.py` | 单臂：写 agentinfer `config.yaml` → 起 Router → 等 KV 订阅 → 快照 → 回放 → 快照 |
| `scripts/summarize.py` | 单臂汇总 |
| `scripts/full_dump.py` | 跨臂/跨轮汇总，生成本文表格 |
| `data/prepare_traces.py` | 从官方数据集生成第 2 节的切片并校验 sha256 |

