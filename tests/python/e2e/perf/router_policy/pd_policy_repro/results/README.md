# H20 实测结果（按 trace 配置分文件）

所有指标定义见 [../README.md](../README.md) 第 6 节；优化比例中 ✅ 表示该指标变好（吞吐更高、延迟更低）。

| trace 配置 | 明细 | 拓扑 | rr 吞吐 | cache 吞吐 | kv 吞吐 | kv vs cache（吞吐，多轮均值） | 轮次 |
|---|---|---|---|---|---|---|---|
| TraceLab 6 会话 / 2P2D（最强 case） | [tracelab_2p2d.md](tracelab_2p2d.md) | **2P2D**（WORKERS=4，PREFILL=2，每台 Prefill 3 个会话） | 118.6 | 126.2 | 160.2 | +27.0% ✅ | 2 |
| TraceLab 12 会话 / 4P4D | [tracelab_4p4d.md](tracelab_4p4d.md) | 4P4D（WORKERS=8，PREFILL=4，每台 Prefill 3 个会话） | 168.6 | 261.2 | 298.5 | +14.3% ✅ | 3 |
| SWE-bench Pro / 4P4D | [swebench_4p4d.md](swebench_4p4d.md) | 4P4D（WORKERS=8，PREFILL=4） | 151.2 | 164.7 | 188.7 | +14.6% ✅ | 2 |
| agentX 6 会话 / 2P2D | [agentx_2p2d.md](agentx_2p2d.md) | **2P2D**（WORKERS=4，PREFILL=2） | 141.4 | 150.8 | 160.7 | +6.6% ✅ | 2 |
| agentX 12 会话 / 4P4D（PR493 二进制历史对照） | [agentx_4p4d_pr493.md](agentx_4p4d_pr493.md) | 4P4D（WORKERS=8，PREFILL=4） | — | 176.3 | 193.8 | +9.9% ✅ | 2 |

> 吞吐单位：输出 token/s。原始数据来自各臂的 `replay/requests.jsonl` 与前后 Prometheus 快照，
> 汇总脚本见 [../scripts/full_dump.py](../scripts/full_dump.py) 与 [../scripts/metrics_json.py](../scripts/metrics_json.py)。
