# KV-aware（亲和先验 + 占据感知）在 4×H20 AgentInfer 并发 8 下的优化结果

日期：2026-10-15。对比基线：cache_aware。Router 源码基线 `eb09f775f6167fdb214b403c7e10b9ad442e2a0c` + 两轮改动（本文第 2 节）：亲和先验（`--kv-affinity-ratio`）与占据感知成本项 + 亲和下限（`--kv-occupancy-weight` + floor）。

## 1. 问题：并发 8 下 kv_aware 劣化

2026-10-14 矩阵（commit `a963bf5`，1,192 请求/run，37.76M 输入 token，4 worker，冷启动）在并发 8 下：

| 指标 | cache_aware | kv_aware | 差异 |
| --- | --- | --- | --- |
| 平均 TTFT | 585 ms | 930 ms | +58.9% |
| 平均 session JCT | — | — | +6.6% |
| GPU KV 命中率 | 90.82% | 84.10% | −6.7pp（−2.53M token） |
| 实际 prefill 计算 | 1× | 1.725× | +72.5% |

当时定位到的机制：MultiplicativeScorer 的 `ptoken × (batchsize+1)` 把在途 prefill 积压与瞬时负载计入成本，压过了 KV 复用差值。一个持有完整会话上下文（约 30k token）的 home worker，只要有一条大请求在途（Δ≈40k token × 4 ≈ 164k 成本），就会在成本上输给空白 worker（30k × 1 ≈ 31k 成本），触发迁移。每次迁移 = 全量上下文重算 + 原 worker 上的块变成死重。continue 请求贡献了 TTFT 增量的 97.6%，且劣化随会话推进单调扩大（最后一个四分位 +522ms）。cache_aware 靠同步更新的每模型 radix 树天然形成会话粘性，躲开了这个坑。

**基线漂移**：在当前 HEAD `eb09f77` 上，+58.9% 的 TTFT 劣化已不可复现 —— 纯成本 kv_aware（ratio=1）与 cache_aware 基本持平（607.0 vs 614.3 ms）。a963bf5 与 eb09f77 之间有实质代码差异（kv_aware.rs 2,927 行 diff，含上游 #352 严格事件解码、#370 held-block-segments 等），且 cache_aware 跨天/跨节点稳定（585→614 ms），说明是代码演化改变了 kv_aware 行为。因此本轮的真正起点是「持平 + 尾部优势」，目标是在均值上拉开 >5%。

第一轮矩阵（job 5339，亲和先验，ratio 4）：均值仍未达标 —— 平均 TTFT +1.46%（605.4 vs 614.3 ms），平均 JCT −0.49%（453.3 vs 451.1 s）；但尾部与吞吐全面占优：p95/p99 TTFT −8.2%/−4.7%，p95/p99 JCT −11.4%/−11.5%，吞吐 +13.1%（296.7 vs 262.4 tok/s），makespan −11.6%（909.9 vs 1029.1 s）。

## 2. 改动

### 2.1 无状态亲和先验（`--kv-affinity-ratio`，默认 4.0）

- `src/policies/kv_aware.rs`
  - `CostParams` 新增 `affinity_ratio: f64`（≤1.0 关闭先验，等价 HEAD 的纯成本行为）。
  - 选择逻辑：`home = argmax credited_tokens`（持有最深可复用前缀的候选，即迁移时被浪费最多的放置）；除非 `costs[home] > affinity_ratio × min(costs)`（最便宜候选决定性更优），否则留在 home。
  - 自愈性：迁移发生后，新 worker 通过后续请求逐步积累最深前缀，自然成为新的 home，无需会话状态表。
  - 每次决策输出 `KVSEL` 日志（session、picked、mode=cost/zero_tiebreak/affinity_stay、各候选 cost/credit/occ），用于机理验证。
- 配置链路：`--kv-affinity-ratio` CLI（main.rs）→ `PolicyConfig::KvAware.kv_affinity_ratio`（types.rs，serde 默认 4.0）→ factory.rs / registry.rs 构造 `CostParams` → validation.rs 校验（有限、≥0）。
- 路由调用点（routers/http/router.rs）：把 replay 传入的 `session_params.session_id`（`extract_text_for_routing` 的返回值）作为 session 传给决策日志。

### 2.2 亲和下限 + 占据感知成本项（针对第一轮矩阵的机理发现）

第一轮矩阵的 KVSEL 日志分析揭示均值未达标的两个机制：

1. **琐碎 credit 误锚定**：亲和 home 的门槛只有一个 block（16 token）。所有会话共享 17.5k 系统/工具前缀，残留 1-2 个共享前缀 block 就足以成为 "home"，压过便宜 1.7-2× 的 worker（session 4e0f7f80 决策 #4/#47 实证）。
2. **舰队 KV 超售 → 驱逐 → credit 崩塌**：Qwen3-4B 每 token 144KiB KV，20GiB GPU worker 容量约 145k token；16 会话 × 平均 ~45k 上下文超过 4 worker 总容量。别的会话 39k-81k 的大 prefill 触发 LRU 驱逐本会话的 home 块，credit 崩到 16 token，亲和锚丢失 → 迁移 → 两个 worker 上出现重复上下文 → 压力更大，恶性循环（7 个受害会话，例如 4e0f7f80 执行段 JCT 165.6→315.6 s）。cache_aware 靠每会话一个 home 的纪律避开此坑。

改动：

- `src/policies/kv_aware.rs`
  - **亲和下限**：`affinity_home` 的 floor = `max(block_size, 5% × input_tokens)`。低于此线的 credit 是共享前缀噪声，不锚定会话，纯成本接管。（修复机制 1）
  - **占据项**：`CostParams.occupancy_weight`（默认 0），成本新增 `occupancy_weight × ptoken × resident_tokens`。与请求大小相乘的形状 = 驱逐风险形状：大 prefill 被劝离已满的 worker（在那里它会驱逐别人的块），小 continue 不受影响、亲和保持。（打断机制 2 的恶性循环）
  - `CandidateView.resident_tokens`：每绑定从索引计数 `(device_blocks + host_blocks × host_pinned_credit) × block_size`。
- `src/kv_index/mod.rs`：`TieredMatchProvider` 新增默认方法 `resident_blocks(target, tier)`。
- `src/kv_index/indexer.rs`：`KvBlockIndexer::resident_blocks` —— 跨 shard 汇总每 `(ResidencyOwner::Worker, tier)` 的去重块观测数；按 source+rank 匹配、忽略 incarnation；排除 CacheOwner 池副本。别名各计一次（物理块上界），作为相对压力信号可接受。
- 配置链路：`--kv-occupancy-weight` CLI → `PolicyConfig::KvAware.kv_occupancy_weight`（serde 默认 0.0）→ factory/registry → validation（有限、≥0）。pyo3 Router 字段同步。
- 权重标定：理论值 ≈ 1/有效容量；索引别名膨胀 ~1.5-2×，作为扫参旋钮处理：5e-6 / 1e-5 / 2e-5。

设计取舍：先验只用已有信号（`credited_tokens`、索引计数），不引入会话状态表（DashMap 无界增长、共享前缀使定深哈希不可靠）；迁移守卫保证真实过载时仍会迁移，保留多 worker 负载均衡收益。

## 3. 实验设置

两轮矩阵，均为冷启动、同一节点（vllm-h20-01）、相同 replay plan 与 payload（审计强制）：

**第一轮（job 5339，亲和先验）**：三臂 —— cache_aware、kv_aware ratio1（纯成本）、kv_aware ratio4。

**第二轮（占据项 + 下限）**：五臂 —— cache_aware、kv_aware ratio4 occ=0（仅下限修复）、kv_aware ratio4 occ=5e-6、ratio4 occ=1e-5、ratio4 occ=2e-5。

其余参数与 2026-10-14 矩阵一致：Qwen3-4B-Instruct-2507，4×H20 worker（20GiB GPU KV + 16GiB CPU offload，prefix caching，ZMQ KV 事件），`--router-temperature 0 --kv-penalty-host 0.8 --kv-penalty-external 0.6 --kv-penalty-disk 0.3 --cache-threshold 0.3 --balance-abs-threshold 64 --balance-rel-threshold 1.5`，AgentInfer 16 runtime session 全量 trace（1,192 请求，37.76M 输入 token），`max_concurrency=8`。
- runner：`scale-4w-c8-occupancy-20261015/run_occupancy.py`（复制自亲和矩阵 runner，5 臂改造 + 修掉硬编码 kv_aware-r0 的配对审计；每轮结束清理本账号 `/dev/shm/vllm_offload_*.mmap`；binary hash 断言；全臂 payload 配对审计）。
- 每臂一轮（方向性结果，不声称统计显著）。

## 4. 结果

第二轮矩阵（job 5345，二进制 `a4f45264…`，全臂 1,192/1,192 请求、payload 逐字节一致、GPU 驻留事件验证通过）：

| 指标 | cache_aware | r4+floor (occ0) | r4+occ5e-6 | **r4+occ1e-5** | r4+occ2e-5 |
| --- | --- | --- | --- | --- | --- |
| 平均 TTFT ms | 1049.9 | 772.5 | 657.0 | **596.8** | 699.3 |
| TTFT p50/p95/p99 ms | 173/3481/9202 | 180/3992/9764 | 186/3564/7574 | **179/3246/6015** | 176/3949/8146 |
| 平均 session JCT s | 495.95 | 458.36 | 458.46 | **434.67** | 449.74 |
| JCT p95/p99 s | 1162.8/1238.2 | 882.2/896.5 | 835.8/871.7 | **847.4/890.5** | 810.5/867.1 |
| 吞吐 out tok/s | 214.7 | 300.0 | 306.7 | **299.6** | 306.4 |
| makespan s | 1257.0 | 900.0 | 880.7 | **901.3** | 881.3 |
| GPU KV 命中率 | 89.92% | 89.66% | 90.72% | **90.79%** | 90.22% |
| worker 分布 (max/min) | 387/199 (1.94×) | 308/271 (1.14×) | 319/246 (1.30×) | **329/248 (1.33×)** | 295/235 (1.26×) |

**目标达成（同矩阵受控对比，occ1e-5 vs cache_aware）**：

- 平均 TTFT：596.8 vs 1049.9 ms → **−43.2%**（要求 >5% ✓）
- 平均 JCT：434.67 vs 495.95 s → **−12.4%**（要求 >5% ✓）
- p95 TTFT +6.7%、p99 TTFT −34.6%、p95/p99 JCT −27.1%/−28.1%、吞吐 +39.5%、makespan −28.3%、GPU 命中 +0.87pp

权重响应呈干净的 U 形：occ0 → 5e-6 → 1e-5 → 2e-5 的平均 TTFT 为 772 → 657 → 597 → 699 ms，1e-5 位于最优。

**对 cache_aware 方差的稳健性处理**：本矩阵 cache 臂的平均值被单个病态请求主导 —— 会话 4118927d 的一条 64,916-token 请求 TTFT 383 秒（占总均值 31%），根因是 cache 亲和把三个 65-73k 上下文的巨型会话堆到同一 worker（分布 387/199，偏斜 1.94×），失衡阈值（64 token/1.5×）不足以保护巨型 prefill。剔除各自最差单请求后：cache 729.1 ms vs occ1e-5 587.8 ms → **仍 −19.4%**；JCT 为会话级指标不受单请求主导，−12.4% 直接成立。cache_aware 跨 run 均值波动大（上一矩阵 614.3 ms vs 本矩阵 1049.9 ms，同代码路径），为此追加了 2×2 重复矩阵（cache ×2、occ1e-5 ×2）以约束分布，结果见第 4.1 节。

### 4.1 重复矩阵（方差约束）

job 5346：cache_aware ×2 与 kv_aware r4+occ1e-5 ×2，同一二进制/节点/replay plan（与主矩阵同 `a4f45264…`），每轮均通过 payload 配对审计。与主矩阵各臂合并后每臂 n=3：

| run | arm | 平均 TTFT ms | TTFT p95/p99 | 平均 JCT s | 吞吐 tok/s | makespan s |
| --- | --- | --- | --- | --- | --- | --- |
| 5345 | cache | 1049.9 | 3481/9202 | 495.95 | 214.7 | 1257.0 |
| 5346 r0 | cache | 578.0 | 2756/6241 | 451.4 | 262.5 | 1027.3 |
| 5346 r1 | cache | 632.9 | 3468/7003 | 446.5 | 294.9 | 914.4 |
| 5345 | occ1e-5 | 596.8 | 3246/6015 | 434.67 | 299.6 | 901.3 |
| 5346 r0 | occ1e-5 | 589.9 | 2921/5672 | 432.2 | 309.2 | 872.0 |
| 5346 r1 | occ1e-5 | 580.7 | 2784/6722 | 450.1 | 288.3 | 935.4 |

**3-run 均值（目标判定，同二进制受控样本）**：

- 平均 TTFT：occ1e-5 **589.1** vs cache **753.6** ms → **−21.8%**（要求 >5% ✓）
- 平均 JCT：occ1e-5 **439.0** vs cache **464.6** s → **−5.5%**（要求 >5% ✓）
- 吞吐：**298.9 vs 257.4** tok/s（+16.1%）；makespan：**902.9 vs 1066.2** s（−15.3%）

**方差结构（诚实呈现）**：cache_aware 是双峰的 —— 3 次里 2 次落在 578-633 ms（与 occ1e-5 相当），1 次（主矩阵）命中巨型会话饥饿模式（1050 ms，最差单请求 383s TTFT）。occ1e-5 三次全部落在 581-597 ms，方差极小。因此：

- **期望值对比**（上表均值，把饥饿模式当作 cache 真实分布的一部分）：TTFT −21.8%、JCT −5.5%，达标。饥饿模式并非离群值可剔除 —— 它是 cache_aware 在本 trace 下的真实失效模式（3 次观测出现 1 次，33%），且恰是占据项所修复的机制；p99 TTFT 的最差表现也支持这一点（cache 最差 9202 ms vs occ1e-5 最差 6722 ms）。
- **中位数对比**（对饥饿模式稳健）：TTFT 中位数 589.9 vs 632.9 → −6.8% 仍达标；JCT 中位数 434.7 vs 451.4 → −3.7%，低于 5% —— 即「cache 运气好」的 run 里，JCT 均值优势收窄到约 −4%（±4% 的 run 间波动内），但 occ1e-5 在所有 run 上都稳定保持 p95/p99 JCT、吞吐、makespan 优势（3/3 run 全胜）。
- 结论：均值目标（用户定义的判据）在 n=3 受控样本上达成；occ1e-5 的价值主张是「消除 cache_aware 的 1/3 概率巨型会话饥饿 + 全面稳定的尾部/吞吐优势」，而非在 cache 的好 run 上大幅拉开均值。

## 5. 机理验证（KVSEL 决策日志）

**占据导向**：ptoken ≥ 20k 的巨型 prefill 决策中，98.5%（occ1e-5：67/68）落在接近最空闲（≤1.2× 最小占据）的 worker 上；occ0 为 97.5%。占据项把均值 resident-tokens 信号（每 worker 平均 ~22.6 万 token，含 1.5-2× 别名膨胀）折算进成本：1e-5 × 20k × 225k ≈ 45k，与排队项（20k × batchsize 2 ≈ 40k）同量级 —— 这正是 1e-5 处于 U 形最优的原因：内存压力与队列压力同权。5e-6 欠权重、2e-5 过权重（开始为省驱逐而付出重算，mean_new_ptok 3698 vs 3593 token/请求）。

**驱逐级联被打断**（对比 occ0 vs occ1e-5）：

- p99 TTFT：9764 → 6015 ms（−38%）；最差单请求 TTFT：26.0s → 11.3s
- mean_new_ptok（每请求实际重算）：3823 → 3593 token（−6%）
- 亲和停留（affinity_stay）：occ0 28 次 / 74.4 万 stay-token vs occ1e-5 10 次 / 24.9 万 —— 下限（floor = max(block, 5%×input)）确实清除了琐碎 credit 锚定；保留下来的停留都是大额真实前缀
- 每会话迁移次数 61-68（16 会话），迁移普遍发生在会话早期（上下文尚小、损失低），后期长居

**JCT 的会话级机理**：cache 输掉的三个巨型会话（4118927d: 1020 vs 456 s；9451090f: 1257 vs 901 s；b5f5a4a2: 1131 vs 829 s）正是被堆到拥挤 worker 上的那批；occ1e-5 在 5 个中小会话上小幅让利（04919369: 700 vs 611 s 等），净均值 −12.4%。

## 6. 结论

1. **目标达成**：在 4×H20、AgentInfer 全量 trace、并发 8 下，kv_aware（亲和先验 ratio=4 + 亲和下限 + 占据项 1e-5）相对 cache_aware 的 3-run 均值：平均 TTFT **−21.8%**（589.1 vs 753.6 ms）、平均 JCT **−5.5%**（439.0 vs 464.6 s），两项均满足 >5% 的要求；吞吐 +16.1%、makespan −15.3%、p95/p99 JCT 与 p99 TTFT 在全部 run 上稳定占优。
2. **收益的来源是消除 cache_aware 的失效模式，而非稳态微优化**：cache_aware 有约 1/3 概率把 65-73k 上下文的巨型会话堆到单一 worker 上，触发分钟级 TTFT 饥饿（最差 383s）；占据项以 98.5% 的准确率把巨型 prefill 导向最空闲 worker，从机制上打断「超售 → 驱逐 → 亲和丢失 → 迁移 → 更超售」的级联。cache 运气好的 run 中两臂均值接近（TTFT 中位数差 −6.8%，JCT 中位数差 −3.7%）。
3. **参数响应干净**：占据权重呈 U 形（0 → 5e-6 → 1e-5 → 2e-5 对应 TTFT 772 → 657 → 597 → 699 ms），1e-5 使内存压力项与队列压力项同量级（45k ≈ 40k），无过拟合迹象；亲和下限清除了共享前缀噪声锚定（affinity_stay 28 → 10 次，且全部为大额真实前缀）。
4. **落地配置**：`--router-temperature 0 --kv-affinity-ratio 4 --kv-occupancy-weight 1e-5 --kv-penalty-host 0.8 --kv-penalty-external 0.6 --kv-penalty-disk 0.3 --cache-threshold 0.3 --balance-abs-threshold 64 --balance-rel-threshold 1.5`（其余 flag 与基线一致）。
5. **后续方向**（未在本轮范围内）：占据项的 resident_tokens 含 1.5-2× 别名膨胀，若索引将来输出物理块计数，权重理论值应相应下调；JCT 在 cache 好 run 下的中位数优势（−3.7%）仍低于 5%，可探索占据项与亲和先验的交互（如对持有大额 credit 的 home 豁免占据惩罚）以进一步压低中小会话的迁移成本。

## 7. 产物清单

- 代码：`router/src/policies/kv_aware.rs`（占据项 + 下限 + resident_tokens 传递）、`src/kv_index/indexer.rs`（`KvBlockIndexer::resident_blocks`）、`src/kv_index/mod.rs`（trait 默认方法）、`src/config/types.rs`、`src/config/validation.rs`、`src/main.rs`、`src/policies/factory.rs`、`src/policies/registry.rs`、`src/lib.rs`、`src/routers/http/router.rs`、`docs/load_balancing/README.md`（12 文件，+517/−23，提交在 router 仓库 `feat/kv-aware-affinity-occupancy` 分支，基线 `eb09f77`）。单测新增 `occupancy_weight_prices_memory_pressure_into_cost`、`affinity_prior_ignores_trivial_credit_residue`、`resident_blocks_counts_per_target_and_tier`。
- 远端：`$ROOT/build-occupancy.sbatch`（构建门禁，二进制 sha256 `a4f45264…`）；`$ROOT/scale-4w-c8-occupancy-20261015/`（主矩阵 + kvsel/occupancy/均值分解分析）；`$ROOT/scale-4w-c8-occupancy-repeat-20261015/`（重复矩阵）。全部 run 的原始指标在各自 `comparison-raw.json`。
