# AgentX 混部 KV-aware：快速验证结论

KV-aware 在本次定向样本中明显减少了实际 prefill 计算，但尚未稳定优于 Cache aware 的端到端结果。Progress 相对纯 KV 降低了平均 P 排队和 TTFT，但总吞吐下降。

## 测试范围

4 张 B300、4 个 TP1 混部 worker、Qwen3-4B BF16，C32；128 GiB KV/worker。32 个 AgentX session 各取前 4 次模型请求，3 组各跑一次，共 128 请求/组。请求长度和块共享结构来自原始数据，token ID 由 AgentInfer 重建。

样本按固定种子和请求特征选择：每 session 输出不超过 2,048 token，至少一个请求的估计新增输入不少于 32,768 token；没有按策略运行结果筛选。它用于观察较强 prefill 压力，不代表全部 AgentX。实验使用相同 workload，worker backend 均校验为 TRTLLM prefill / trtllm-gen decode。选定 GPU 未观察到外部进程，其他 GPU 有活动，比例在各组间不同。

## KV-aware 相对 Cache

| 指标 | Cache | KV | 变化 |
|---|---:|---:|---:|
| P 本地命中率 | 55.45% | 64.06% | +8.60 个百分点 |
| 实际新算 P token | 2,559,552 | 2,065,216 | -19.31% |
| P prefill 均值 | 514.54 ms | 448.08 ms | -12.92% |
| P queue 均值 | 510.26 ms | 663.22 ms | +29.98% |
| TTFT 均值 / P95 | 1,256.42 / 5,188.87 ms | 1,361.01 / 6,702.83 ms | +8.32% / +29.18% |
| TPOT 均值 | 25.52 ms | 19.44 ms | -23.83% |
| 总吞吐 | 167,906.54 token/s | 168,779.64 token/s | +0.52% |
| JCT 均值 | 26.88 s | 26.23 s | -2.45% |

这里可见的直接收益是少算 494,336 个 P token，P prefill 也缩短；但 P 排队和 TTFT 均值、P95 同时变差，不能把省下的计算量等同于端到端收益。P prefill 为 worker 从调度到首 token 的阶段时间，混部执行也会影响它，并非纯 GPU 计算时间。

Cache 的 128 次决策中有 31 次触发负载失衡回退。代码在失衡模式按活跃请求数选节点，KV-aware 则结合前缀及 prefill 工作量打分；这是值得核查的差异，但当前数据没有逐请求证明这 31 次造成了多少额外计算。

## Progress 相对纯 KV

| 指标 | KV | KV+Progress | 变化 |
|---|---:|---:|---:|
| P 命中率 / 新算 P token | 64.06% / 2,065,216 | 64.06% / 2,065,216 | 相同 |
| P queue 均值 | 663.22 ms | 537.41 ms | -18.97% |
| P prefill 均值 | 448.08 ms | 445.85 ms | -0.50% |
| TTFT 均值 / P95 | 1,361.01 / 6,702.83 ms | 1,238.25 / 5,526.55 ms | -9.02% / -17.55% |
| TPOT 均值 | 19.44 ms | 20.33 ms | +4.59% |
| 总吞吐 | 168,779.64 token/s | 164,505.69 token/s | -2.53% |
| JCT 均值 | 26.23 s | 26.78 s | +2.10% |

两组各有 128 个 queue 和 prefill 样本；queue 均值下降约 125.81 ms，与 TTFT 均值下降约 122.76 ms 接近，而实际 P 计算量没有变化。KV+Progress 接收了 635 条 progress 更新。这支持「收益主要体现在排队」这一轮的观察，但单轮闭环对照不能证明因果。

请求逐一配对后，128 个请求只有 51 个 TTFT 下降、77 个上升。改善最大的 5 个请求合计下降约 23.64 秒，而全部请求合计仅下降约 15.71 秒；均值收益高度集中。TTFT P95 虽比纯 KV 下降，但仍比 Cache 高 6.51%。

吞吐降低的直接数值原因是三组 token 总量固定，而 KV → KV+Progress 的回放时间由 34.27 s 增至 35.16 s；同时 worker decode 均值 5.15 → 5.40 s，TPOT 均值 +4.59%。这些指标的同步变化尚不能证明是哪次选路或哪个 worker 导致了整轮变慢。

## 为什么 TTFT 和总吞吐方向不同

TTFT 使用相同 128 个请求的均值；P queue 与 P prefill 的 histogram 均各有 128 个样本。下表的「余量」是 `TTFT 均值 - P queue 均值 - P prefill 均值`，包含 Router、传输及其他未单独测出的时间，也可能包含计时边界差异，并非独立实测阶段。

| 方案 | P queue 均值 | P prefill 均值 | 余量 | TTFT 均值 |
|---|---:|---:|---:|---:|
| Cache | 510.26 ms | 514.54 ms | 231.61 ms | 1,256.42 ms |
| KV | 663.22 ms | 448.08 ms | 249.71 ms | 1,361.01 ms |
| KV+Progress | 537.41 ms | 445.85 ms | 255.00 ms | 1,238.25 ms |

Cache → KV 时，P queue **增加 152.96 ms**，超过 P prefill **减少的 66.46 ms**，所以少算 P 并未让平均 TTFT 下降。再开 Progress 后，P queue 降低 125.81 ms，TTFT 均值降低 122.76 ms；两者的均值变化接近，但这不能证明全部改善都由推送引起。KV+Progress 相对 Cache 的 TTFT 均值下降 1.45%，P95 仍上升 6.51%。

逐 worker 数据也显示，请求数量接近并不代表 prefill 负担接近。纯 KV 的 W3 处理 26 个请求，平均输入约 65,866 token、平均新算约 21,755 token，P queue / prefill 均值为 863.83 / 715.13 ms；W0 处理 36 个请求，平均输入约 37,399 token、平均新算约 13,639 token，对应 585.82 / 363.67 ms。请求分配和批次不同，不能据此单独断定 W3 的长输入就是排队的根因。

总吞吐统计的是整轮固定 token 总数除以最后完成时间，与每请求平均 TPOT 不是同一个口径。Cache → KV+Progress 的回放时间 **34.45 → 35.16 s（+2.07%）**，因此总吞吐为 **-2.03%**，即使 TPOT 均值下降了 20.34%。

把各轮相对首请求的共同 5.00–29.27 s 作为窗口，按 worker 约 1 秒一次的计数器采样，Cache / KV / KV+Progress 的输出速率约为 **1,331.64 / 1,448.92 / 1,417.66 token/s**；不同轮次该窗口的请求组成可能不同。窗口结束至最后一个请求完成期间，客户端平均在途请求数为 **6.33 / 3.62 / 3.16**，对应时长约 **5.14 / 4.96 / 5.85 s**。KV+Progress 的尾段并发更低且结束更晚，共同窗口的输出速率优势没有转化为整轮吞吐优势；这些数据不能证明具体是哪一次调度造成了尾段变化。

这轮所有组都没有 preemption，4 个 worker 的 KV 使用率采样峰值均低于 58%；没有观察到支持「抢占或 KV 占满」的证据，但 1 秒采样无法排除短时波动。当前 KV 成本分数仍是 `(本请求新算 P + 已在途剩余 P) × (活动请求数 + 1)`；decode 上下文只在分数完全相同时用于次级选择。该分数并不直接等于 worker 的实际排队时间。要确认具体哪次选路导致排队或尾部变慢，还需关联请求的候选 worker 状态、选路和 vLLM 调度步；本次常规性能日志没有这些完整信息。

## 使用建议与边界

在长输入、前缀可复用、多个 prefill 并发的 AgentX 场景，值得重点测 **真实 P 命中率、新算 token 和 P queue**：这轮的 KV 直接减少了 P 计算，Progress 对照同时出现了更低的 P 排队。不要只以逻辑 input/total token/s 或 TTFT 均值判断；还应看尾部、TPOT、实际新算量和相同请求配对变化。

本实验每组一次且共享主机，其结果不能证明普适或稳定收益。若需要验证稳定性，应保持 workload 和 backend 不变，先针对 Cache / KV / KV+Progress 做小规模重复及请求级诊断，而不是直接扩展成多并发大矩阵。

完整分位数和 worker 数据：[REPORT.md](REPORT.md)。均值拆分与共同窗口计算可由 `python3 analyze_root_cause.py` 重建，数据见 `root-cause.json`。请求配对数据：`runs/c32-kv_aware/paired-c32-cache_aware.json`、`runs/c32-kv_aware_progress/paired-c32-kv_aware.json` 和 `runs/c32-kv_aware_progress/paired-c32-cache_aware.json`。
