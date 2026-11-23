# agentX PD 路由策略性能看护用例

一句话：用一份固定的 agentX 真实回放负载，在 4P4D 上对比 `kv_aware`、`cache_aware` 与上一版发布（v0.1.17），把"路由有没有把前缀复用搞坏"变成可重复的门禁。

## 一、结论摘要

| 对比 | 命中率 | 新算 Prefill | 吞吐 | TTFT mean | 会话 JCT p95 |
| --- | --- | --- | --- | --- | --- |
| kv_aware vs cache_aware（同一二进制） | **+6.42 / +6.42 pt** | **−35.2% / −43.2%** | +5.5% / +14.6% | −31.8% / −45.0% | −14.0% / −20.5% |
| kv_aware vs v0.1.17 原版 | **+7.66 / +6.06 pt** | −32.5% / +5.6% | +7.0% / +6.2% | −16.8% / +4.2% | −2.9% / −6.8% |

两轮交错，符号一致的部分：命中率（两个基线各 2/2）、吞吐（各 2/2）、JCT p95（各 2/2）。符号会翻转的部分：新算 Prefill 与 TTFT mean（对 v0.1.17），原因见第六节。

门禁只钉两条：**命中率差 ≥ +4.0 pt**、**吞吐 ≥ 0.98×**。其余指标只打印不判定，因为它们的裕度小于轮间噪声。

## 二、用例构造

数据来源是官方 agentX replay 数据集 `agentx-data/traces.jsonl`（393 个会话，`hash_id_scope: local`，`block_size: 64`）。每个会话原样保留自己的 `requests`，只截取轮数，不做任何合成。

| 属性 | 值 |
| --- | --- |
| 会话数 × 轮数 | 12 × 12 |
| trace 请求数 | 144 |
| 回放实际请求数 | 315（agentinfer 会把 trace 条目展开成多次调用） |
| 总 input | 8,159,680 token |
| 总 output | 116,618 token |
| in/out 比 | 70 : 1（Prefill 主导） |
| 每会话 input | 415K / 417K / 417K / 418K / 697K / 697K / 698K / 701K / 920K / 924K / 925K / 930K |
| 会话大小极差 | 2.24× |
| 单请求最大 input | 116,160 token |
| 理想 LPT 四台分配 | 2.04M × 4，max/avg = 1.00 |
| trace sha256 | `e7e7fd7cd31f178483e2027281ce4592fca38bd35252c09b88e2a9dee1078638` |

三条筛选理由，缺一不可：

1. **12 个会话对 4 台 Prefill**：每台都会被用到，任何一臂都不可能靠"有台闲置"取胜。会话数少于 Prefill 台数时 `cache_aware` 会结构性用不满 worker，那是另一种机制，会把结论退化成极端情形。
2. **会话大小差 2.24×，但 12 个会话轮数完全相等**：这正是"按请求条数再平衡"的盲区。`cache_aware` 的失衡判据是 `max - min > balance_abs_threshold` 且 `max > min × balance_rel_threshold`，单位是**请求条数**；12 个请求打到 4 台时条数差最多 2，过不了阈值 2，于是它从头到尾不看会话大小。
3. **单请求 ≤ 118K**：回放会重建 prompt，实测比 trace 里的 `in` 多出约 32K token，所以引擎窗口要给到 163840。

数据集的一个性质让这个 case 变尖锐：`hash_id_scope: local` 意味着**跨会话前缀复用几乎为零**。一个会话只能待在第一个持有它前缀的那台 worker 上，所以放置决策在首轮就定死了，两个策略唯一的差别是**留不留得住**。`kv_aware` 的成本是 `新算 token + 在飞 token`，会话落位后新算趋零，它不搬；`cache_aware` 按条数均衡，条数一过阈值就把会话挪走，而挪动就丢掉该会话的私有前缀。

## 三、复现步骤

### 1. 环境

8 卡单机做 4P4D：4 台 Prefill、4 台 Decode。vLLM 0.33 + NIXL 连接器，`--prefix-caching-hash-algo sha256_cbor`、`--block-size 16`，与基准实验一致。

### 2. 生成 trace

```bash
python tests/python/e2e/perf/router_policy/agentx_pd_hetero/extract_agentx_hetero.py \
    /path/to/agentx-data/traces.jsonl .
```

脚本里写死了 12 个会话 id，输出 `agentx-hetero12x12.jsonl` 并打印 sha256，应与上表的 `e7e7fd7c…` 完全一致（已在 H20 上复验）。

### 3. 运行三臂交错

三个臂、两个二进制、同一份负载。每臂重启 Router 并重置全部引擎的前缀缓存，所以每臂都是冷启动。

| 臂 | 二进制 | 策略 |
| --- | --- | --- |
| `cache` | 本仓库 | `--policy cache_aware`（两段都是） |
| `current_kv` | 本仓库 | `--policy kv_aware`（两段都是） |
| `release_kv` | 上一版发布包 | 自带的 `kv_aware` |

```bash
export ROUTER_POLICY_ROOT=/path/to/run-root
export ROUTER_POLICY_PYTHON=/path/to/python
export AGENT_INFER_SRC=/path/to/agent-infer
export SCHEDULE_SPEC='cache:h1-cache:agentx-hetero12x12.jsonl;current_kv:h1-kv:agentx-hetero12x12.jsonl;release_kv:h1-rel:agentx-hetero12x12.jsonl;cache:h2-cache:agentx-hetero12x12.jsonl;current_kv:h2-kv:agentx-hetero12x12.jsonl;release_kv:h2-rel:agentx-hetero12x12.jsonl'
python series3.py
```

`SCHEDULE_SPEC` 用分号分隔条目，因为 `sbatch --export` 会按逗号切分环境变量值。

### 4. 关键引擎与路由参数

| 参数 | 值 |
| --- | --- |
| `--max-model-len` | 163840（回放重建 prompt，比 trace 的 `in` 多约 32K） |
| `--kv-cache-memory-bytes` | 27,917,287,424（26 GiB / 台） |
| `--router-temperature` | 0 |
| `--cache-threshold` | 0.3 |
| `--balance-abs-threshold` / `--balance-rel-threshold` | 2 / 1.5 |
| `--kv-penalty-host` / `external` / `disk` | 0.8 / 0.6 / 0.3 |
| 回放 `trace_same_agent_gap_scale` | 0.002 |
| 回放 `max_inflight_requests` | 1 |

### 5. 产物

`runs/<label>/` 下：`replay/requests.jsonl`（逐请求输入/输出/首 token/完成时刻）、`manifest.json`（臂、二进制 sha、trace sha）、`before/` 与 `after/` 的 Prometheus 快照（每台引擎的 `prompt_tokens_total`、`prefix_cache_hits_total`、`prefix_cache_queries_total`、`request_success_total`）。

## 四、性能对比

### 1. 主表（315 请求 / 臂，零失败）

| 臂 | 轮 | 吞吐（out tok/s） | 总吞吐（tok/s） | 命中率 | 新算 Prefill | TTFT mean | TPOT mean | JCT mean | JCT p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cache_aware | 1 | 181.1 | 17,851 | 77.92% | 4,037,120 | 5.89 s | 14.6 ms | 423.2 s | 968.1 s |
| **kv_aware** | 1 | **191.0** | **18,828** | **84.35%** | **2,614,448** | **4.01 s** | **14.1 ms** | **379.3 s** | **832.9 s** |
| v0.1.17 | 1 | 178.5 | 17,594 | 76.69% | 3,870,576 | 4.83 s | 14.2 ms | 397.6 s | 857.9 s |
| cache_aware | 2 | 171.5 | 16,907 | 77.20% | 4,915,568 | 7.34 s | 14.6 ms | 462.7 s | 988.3 s |
| **kv_aware** | 2 | **196.5** | **19,372** | **83.61%** | **2,792,224** | **4.04 s** | **14.3 ms** | **374.7 s** | **785.6 s** |
| v0.1.17 | 2 | 185.1 | 18,249 | 77.55% | 2,645,008 | 3.88 s | 15.8 ms | 381.7 s | 842.8 s |

### 2. 分位数（mean / p50 / p95 / p99）

| 臂 | 轮 | TTFT (s) | TPOT (ms) | 会话 JCT (s) |
| --- | --- | --- | --- | --- |
| cache_aware | 1 | 5.89 / 1.67 / 23.64 / 38.81 | 14.6 / 14.4 / 24.9 / 27.1 | 423.2 / 289.5 / 968.1 / 1124.2 |
| kv_aware | 1 | 4.01 / 1.01 / 21.04 / 34.11 | 14.1 / 12.4 / 25.7 / 28.3 | 379.3 / 292.4 / 832.9 / 1048.9 |
| v0.1.17 | 1 | 4.83 / 1.42 / 21.93 / 33.82 | 14.2 / 12.7 / 25.8 / 28.0 | 397.6 / 310.4 / 857.9 / 1115.8 |
| cache_aware | 2 | 7.34 / 1.44 / 32.98 / 52.14 | 14.6 / 13.7 / 26.2 / 29.6 | 462.7 / 380.5 / 988.3 / 1180.2 |
| kv_aware | 2 | 4.04 / 1.23 / 19.39 / 29.70 | 14.3 / 12.2 / 26.8 / 30.1 | 374.7 / 268.8 / 785.6 / 1014.6 |
| v0.1.17 | 2 | 3.88 / 1.21 / 19.36 / 30.54 | 15.8 / 14.6 / 27.6 / 30.8 | 381.7 / 264.9 / 842.8 / 1078.9 |

### 3. 配对差值（每轮相对各自基线）

| 对比 | 轮 | 吞吐 | 命中率 | 新算 Prefill | TTFT mean | TTFT p95 | TPOT mean | JCT mean | JCT p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kv vs cache | 1 | +5.5% | +6.42 pt | −35.2% | −31.8% | −11.0% | −2.8% | −10.4% | −14.0% |
| kv vs cache | 2 | +14.6% | +6.42 pt | −43.2% | −45.0% | −41.2% | −2.0% | −19.0% | −20.5% |
| kv vs v0.1.17 | 1 | +7.0% | +7.66 pt | −32.5% | −16.8% | −4.0% | −0.1% | −4.6% | −2.9% |
| kv vs v0.1.17 | 2 | +6.2% | +6.06 pt | +5.6% | +4.2% | +0.1% | −9.0% | −1.8% | −6.8% |

### 4. 效应 vs 轮间噪声（这是能不能做门禁的唯一依据）

| 指标 | 效应（两轮） | 噪声（配对基线自身轮间） | 效应/噪声 | 是否门禁 |
| --- | --- | --- | --- | --- |
| Prefill 命中率 | +6.42 / +6.42 pt | 0.72 pt | **8.9×** | 门禁，阈值 +4.0 pt |
| 吞吐 | +5.5% / +14.6% | 5.6% | 1.0–2.6× | 门禁，只做地板 0.98× |
| JCT p95 | −14.0% / −20.5% | 2.1% | 6.7–9.8× | 只报告（对 v0.1.17 降到 1.8×） |
| 新算 Prefill | −35.2% / −43.2% | 21.8% | 1.6–2.0× | 只报告 |
| TTFT mean | −31.8% / −45.0% | 24.6% | 1.3–1.8× | 只报告 |

## 五、门禁规则

一套规则，两个基线各跑一遍，**每一轮都必须通过**，否则非零退出。

| 规则 | 阈值 | vs cache_aware | vs v0.1.17 | 结论 |
| --- | --- | --- | --- | --- |
| 全部请求成功 | 必须 | 315/315 ×2 | 315/315 ×2 | 通过 |
| Prefill 命中率差 | ≥ +4.0 pt | +6.42、+6.42 | +7.66、+6.06 | 通过 |
| 吞吐 | ≥ 0.98× | 1.055×、1.146× | 1.070×、1.062× | 通过 |
| 候选自身 Prefill 倾斜 max/avg | ≤ 1.60（绝对） | 1.235、1.365 | 同 | 通过 |

```bash
python gate.py --profile both --baseline 'runs/*-cache' --candidate 'runs/*-kv'
python gate.py --profile both --baseline 'runs/*-rel'   --candidate 'runs/*-kv'
```

`report` 项（打印但不判定）：新算 Prefill、TTFT mean/p95、TPOT mean、JCT mean/p95。理由是同一条规则的裕度在换基线后会小于噪声，例如 JCT p95 对 v0.1.17 是 0.971× / 0.932×，裕度只有 3%–7%，而 kv 自身 JCT p95 轮间就摆 6%。

## 六、限制与结论

1. **吞吐效应只有个位数到十几个百分点，不适合"必须快 X%"门禁。** 第一轮 +5.5% 与噪声（5.6%）同量级。真正稳定的是命中率差（8.7 倍噪声）和 Prefill 工作量（−35% 至 −43%）。
2. **端到端收益的上限不在 trace，在 `kv_aware` 自己的放置质量。** 本 case 里它的台间倾斜是 1.23–1.37，第一轮甚至比 `cache_aware`（1.14）更倾斜；wall 由最忙那台决定，所以省下来的 Prefill 工作量有一大半被它自己的不均衡吃掉。把倾斜压到 ≤1.1 才会把 5%–15% 放大到几十个百分点。
3. **v0.1.17 的绝对数字不能当算法基准。** 它在 vLLM 0.33 上会整批丢弃 KV 事件（`kv_index/decode.rs` 里 `events.len() > 4096` 直接返回错误，硬编码无开关），实测两轮分别丢 46 批与 22 批；它自己的命中率两轮从 76.69% 摆到 77.55%、新算 Prefill 从 3.87M 摆到 2.65M。所以对它只能写"不退化"（吞吐 ≥0.95×、JCT p95 ≤1.05×），不能写收益，并且必须把这句备注随数字一起给出。
4. **还差 5 轮才能设成阻塞门禁。** 目前 2 轮；按准入门槛需要连续 5 轮交错、门禁指标 5/5 同号，然后把阈值钉在**最差**一轮。在此之前按 nightly 趋势 + 失败告警运行。
5. **这个用例暂时不是 pytest cell。** 目录里的 `stack_launcher.py` 只支持 colocated（4 worker，没有 `--prefill/--decode` 拆分），所以本 case 用同目录的 PD 运行器驱动；`gate.py` 消费的产物（`replay/requests.jsonl` 加 Prometheus 快照）与 pytest cell 完全一致，给 `stack_launcher` 补上 PD 支持即可变成 cell，门禁不用改。
