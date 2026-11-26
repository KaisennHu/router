# H20 实测结果（按 trace 配置分文件）

所有指标定义见 [../README.md](../README.md) 第 6 节；优化比例中 ✅ 表示该指标变好（吞吐更高、延迟更低）。

| trace 配置 | 明细 | 拓扑 | rr 吞吐 | cache 吞吐 | kv 吞吐 | kv vs cache（吞吐） | rr TPOT | cache TPOT | kv TPOT | 轮次 |
|---|---|---|---|---|---|---|---|---|---|---|
| TraceLab 6 会话 / 2P2D（最强 case） | [tracelab_2p2d.md](tracelab_2p2d.md) | **2P2D**（WORKERS=4，PREFILL=2，每台 Prefill 3 个会话） | 118.6 | 126.2 | 160.2 | +27.0% ✅ | 19.1 | 30.3 | 29.1 | 2 |
| TraceLab 12 会话 / 4P4D | [tracelab_4p4d.md](tracelab_4p4d.md) | 4P4D（WORKERS=8，PREFILL=4，每台 Prefill 3 个会话） | 168.6 | 261.2 | 298.5 | +14.3% ✅ | 16.0 | 19.6 | 18.3 | 3 |
| SWE-bench Pro / 4P4D | [swebench_4p4d.md](swebench_4p4d.md) | 4P4D（WORKERS=8，PREFILL=4） | 151.2 | 164.7 | 188.7 | +14.6% ✅ | 13.7 | 11.5 | 11.6 | 2 |
| agentX 12 会话 / 4P4D（PR493 二进制历史对照） | [agentx_4p4d_pr493.md](agentx_4p4d_pr493.md) | 4P4D（WORKERS=8，PREFILL=4） | — | 176.3 | 193.8 | +9.9% ✅ | — | 14.6 | 14.2 | 2 |

> **吞吐单位：输出 token/s；TPOT 单位：ms（越低越好）。**
>
> 读表提醒：`cache_aware`／`kv_aware` 相对 `round_robin` 通常是「吞吐 ↑、TTFT ↓、JCT ↓、**TPOT ↑（劣化）**」的取舍——
> `round_robin` 完全不复用前缀、系统卡在 Prefill，decode 侧平均只挂着约 2 条序列，所以 TPOT 好看但总吞吐低；
> cache/kv 拆掉 Prefill 瓶颈后 decode 填得更满（约 4~5 条序列），TPOT 随 batch 变大而上升。
> 每个 case 文件里的「吞吐 vs TPOT 取舍」小节给出 `N̄ = Σ(out_i·TPOT_i)/T` 的严格计算（推断量，非引擎自报）。
>
> 真正该对比的是 **kv_aware vs cache_aware**：在所有 case 的所有指标上 kv 都不劣于 cache（吞吐更高、TTFT/JCT 更低、TPOT 持平或更好）。

> 原始数据来自各臂的 `replay/requests.jsonl` 与前后 Prometheus 快照；汇总脚本见 [../scripts/full_dump.py](../scripts/full_dump.py) 与 [../scripts/metrics_json.py](../scripts/metrics_json.py)。

