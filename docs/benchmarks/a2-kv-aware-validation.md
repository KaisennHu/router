# KV-aware prefix reuse on Ascend A2

We validated incremental KV-event routing on Qwen3-4B, Qwen3.5-4B, Qwen3.6-27B, Qwen3.8-27B, Qwen4Exp, and two DeepSeek V4 Flash checkpoints. The matrix covers colocated serving and Prefill/Decode (P/D) disaggregation, with MTP and DSpark controls where tested. It reports **Router prefix credit**, **actual local reuse**, and **Decode remote reads** separately.

- **Prefix reuse matched** in the stable probes on the pinned Router build. The full DeepSeek V4 Flash checkpoint matched at 16,384 tokens with DSpark enabled and disabled, including probes one and two tokens past that boundary.
- **P/D worked with the connector combinations listed below.** Some Qwen requests using Ascend's `MooncakeHybridConnector` failed before the remote read even when Prefill reuse matched and Router forwarded the entire transfer payload. A direct worker-to-worker control reproduced the Qwen3.8 MTP failure.
- **Text quality varied.** JSON and SSE completion are reported as API checks. Repetitive or incorrect output is recorded independently of cache accuracy and transfer.

## Results

Read each pair as **Router credit / worker cached tokens**. For P/D, the worker count is the selected Prefill engine's local reuse. The boundary probe uses a shorter prefix with one token beyond a page boundary. The divergent prompt uses different token IDs from the first page. `P` and `D` count independent Prefill and Decode workers, respectively.

### Colocated serving

| Model and mode | TP | Warm prefix | Boundary + 1 | Divergent | Draft tokens during activity probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B, full attention | 1 | 384 / 384 | 256 / 256 | 0 / 0 | — |
| Qwen3.5-4B, speculation off | 1 | 1,920 / 1,920 | 0 / 0 | 0 / 0 | — |
| Qwen3.5-4B, MTP | 1 | 1,920 / 1,920 | 0 / 0 | 0 / 0 | 48 |
| Qwen3.5-4B, DSpark | 1 | 1,920 / 1,920 | 0 / 0 | 0 / 0 | 376 |
| Qwen3.6-27B W8A8, speculation off | 1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | — |
| Qwen3.6-27B W8A8, MTP | 1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 51 |
| Qwen3.6-27B W8A8, DSpark | 1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 56 |
| Qwen3.8-27B W8A8, speculation off | 1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | — |
| Qwen3.8-27B W8A8, MTP | 1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 48 |
| Qwen3.8-27B BF16, DSpark | 2 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 376 |
| Qwen4Exp W4A8, speculation off | 2 | 4,992 / 4,992 | 3,328 / 3,328 | 0 / 0 | — |
| Qwen4Exp W4A8, MTP | 2 | 4,992 / 4,992 | 3,328 / 3,328 | 0 / 0 | 45 |
| DeepSeek V4 Flash mini4 W8A8, speculation off | 1 | 16,384 / 16,384 | 16,384 / 16,384 | 0 / 0 | — |
| DeepSeek V4 Flash full W8A8, speculation off | 8 | 16,384 / 16,384 | 16,384 / 16,384 | 0 / 0 | — |
| DeepSeek V4 Flash full W8A8, DSpark | 8 | 16,384 / 16,384 | 16,384 / 16,384 | 0 / 0 | 224 |

The earlier two-worker controls also covered Qwen3-4B, Qwen3.5-4B with MTP on and off, and Qwen3.8-27B W8A8 with MTP on and off. When distinct prefixes were warmed on two workers, the relevant affinity probes selected the matching worker. The Qwen3.5 and Qwen3.8 warm counts were 1,920 and 2,688 tokens per worker.

### P/D disaggregation

“Reads / needed” counts requests with a connector transfer completion for the same Decode request ID. The denominator excludes probes where Decode already had the needed prefix locally. Connector completion is counted per request, including for TP deployments. The warm column refers to the local Prefill lookup.

| Model and mode | P/D | Ascend connector | Warm prefix | Boundary + 1 | Divergent | Unchanged payload | Reads / needed | Decode draft tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 1P1D | MooncakeConnectorV1 | 384 / 384 | 256 / 256 | 0 / 0 | 8/8 | 8/8 | — |
| Qwen3.5-4B, speculation off | 1P1D | MooncakeConnectorV1 | 1,920 / 1,920 | 0 / 0 | 0 / 0 | 8/8 | 7/7 | — |
| Qwen3.5-4B, MTP | 1P1D | MooncakeConnectorV1 | 1,920 / 1,920 | 0 / 0 | 0 / 0 | 9/9 | 8/8 | 51 |
| Qwen3.5-4B, DSpark | 1P1D | MooncakeConnectorV1 | 1,920 / 1,920 | 0 / 0 | 0 / 0 | 9/9 | 9/9 | 384 |
| Qwen3.6-27B W8A8, speculation off | 1P1D | MooncakeConnectorV1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 8/8 | 7/7 | — |
| Qwen3.6-27B W8A8, MTP | 1P1D | MooncakeConnectorV1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 9/9 | 8/8 | 51 |
| Qwen3.6-27B W8A8, DSpark | 1P1D | MooncakeConnectorV1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 9/9 | 9/9 | 56 |
| Qwen3.8-27B W8A8, speculation off | 1P1D | MooncakeConnectorV1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 8/8 | 7/7 | — |
| Qwen3.8-27B W8A8, MTP | 1P1D | MooncakeConnectorV1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 9/9 | 8/8 | 60 |
| Qwen3.8-27B BF16, DSpark | 1P1D | MooncakeConnectorV1 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 9/9 | 9/9 | 376 |
| Qwen4Exp W4A8, speculation off | 1P1D | MooncakeConnectorV2 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 8/8 | 7/7 | — |
| Qwen4Exp W4A8, MTP | 1P1D | MooncakeConnectorV2 | 2,688 / 2,688 | 0 / 0 | 0 / 0 | 9/9 | 9/9 | 51 |
| DeepSeek V4 Flash mini4 W8A8 | 1P1D | MooncakeHybridConnector | 16,384 / 16,384 | 0 / 0 | 0 / 0 | 8/8 | 8/8 | — |

Earlier **2P1D MooncakeConnectorV1** runs on Router `b65f4e5` covered Qwen3.5-4B and Qwen3.8-27B W8A8, each with MTP on and off. All four scenarios routed both warmed prefixes to their respective Prefill workers, matched local warm credit on each worker, and recorded eight transfer completions and nine equal handoffs per scenario. Those runs captured completion logs and did not retain the Decode external-token counter.

Full DeepSeek V4 Flash was tested in colocated mode. The measured model used all eight NPUs and approximately 39 GiB per NPU with a single TP8 worker. DeepSeek V4 Pro is outside this A2 experiment.

### Additional colocated deployment checks

| Case | Support and A2 validation status |
| --- | --- |
| Qwen4Exp W4A8 with DSpark | **Rejected by the runtime; no valid combination exists.** The Ascend build refuses the method during speculative-config validation, before any weights load: `NotImplementedError: Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear n-gram proposers` (raised both for a DSpark request without a draft and for one naming the target as the draft). The guard is model-level in `vllm/model_executor/models/config.py`: only `mtp`, `ngram` and `ngram_gpu` are accepted. The checkpoint matches that restriction: `Qwen3.8-Flash-Next-w4a8` declares `text_config.mtp_num_hidden_layers = 1`, carries 1,575 `mtp.*` tensors and no `dspark_*` configuration keys, and its weight description has no DSpark parameters. The native MTP arm remains as reported in the colocated and P/D matrices. [Ascend recipe](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-Flash-Next.html) |
| DeepSeek V4 Flash W8A8 with classic MTP | The v0.32.0rc1 image has a V4 MTP implementation. The older W8A8 MTP checkpoint contains the draft's `mtp.0.e_proj.weight` and `mtp.0.h_proj.weight`. The 0731 checkpoint used in the measured DSpark case contains a different draft, with `main_proj`, Markov and confidence weights. The dedicated 70-shard MTP checkpoint is being staged for a TP8 colocated run. [Pinned Ascend MTP fixture](https://github.com/vllm-project/vllm-ascend/blob/24487f4f48ee99cf45d86d1c504e645549b8c1ef/tests/e2e/cases/models/configs/DeepSeek/acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml) |
| DeepSeek V4.1 Flash W4A8 | **Not runnable on this cluster: one A2 node is too small.** The vllm-ascend 0.32.0rc1 recipe lists two A2 rows and both require two nodes even for colocated deployment (`DeepSeek-V4.1-Flash-1017-w4a8`: 2 nodes; `-w8a8`: 2 nodes colocated, 4 for P/D). Weight sizes confirm the capacity limit: the BF16 repository is 510.3 GB over 48 shards and a W8A8 mirror is 772.1 GB, while one A2 node provides 8 x 64 GB = 512 GB of HBM. The support matrix lists DSpark as supported for this model on A2, so the limitation is node count, not software; the pinned vLLM build also rejects classic MTP for V4.1. There is no Router prefix-accuracy result for V4.1 on A2. [Ascend deployment guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html#55-multi-node-colocated-deployment) |

The V4 Flash classic-MTP probe uses the same pinned Router and incremental event settings as the main matrix.

The mini4 colocated probes were re-run on the updated Router `1c61fc74` with the layout derived from the live events instead of a hand-written one. The engine exposed eight groups (sparse MLA pages at 512 and 16,384, four 128-token sliding-window MLA groups, then 8 and 32), and all four probes matched: warm 16,384 / 16,384, one token past the boundary 16,384 / 16,384, two tokens past it 16,384 / 16,384, and the divergent prompt 0 / 0. The engine's `get_computed_blocks` agreed with the Router credit and the worker response on every probe. It sets `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1,"enforce_eager":true}'`, derives the layout from the running engine, and checks warm, boundary and divergent prompts after replay. Acceptance requires matching Router credit, worker response and the engine lookup, plus a positive draft-token delta.

### API and text checks

The ordinary matrix exercised chat and plain completions with JSON and SSE, checked the SSE `[DONE]` marker, and sent a longer chat request with a 256-token output budget. The full Flash pair used explicit token-ID completion probes. Qwen3-4B and the colocated Qwen MTP runs returned 323 on the short chat prompt. Some Qwen P/D rows returned repeated `!` despite HTTP 200, including some samples with connector completions. The mini4 checkpoint also produced nonsensical text in colocated serving. These observations cover the recorded smoke prompts.

## Environment and model profiles

| Item | Value |
| --- | --- |
| Date and host | November 20–21, 2026, one Ascend A2 node, eight 64 GiB NPUs identified as 910B1 by `npu-smi` |
| Runtime image | `quay.io/ascend/vllm-ascend@sha256:ff6d1ea57df08dd9382c51d202d0dd2d9acc35d53a65bcb0dfd906886e674912` |
| Packages | `vllm 0.32.0+empty`, `vllm-ascend 0.32.0rc1`, `torch-npu 2.10.0.post7` |
| Router | [`1c61fc74a20b2f0bc3af389646d3a4666d51df55`](https://github.com/KaisennHu/router/commit/1c61fc74a20b2f0bc3af389646d3a4666d51df55) |
| Measured Router binary SHA-256 | `ade112e3a2611bba52d4db73e6e8fcef5dc396c4bc6fe0827e2b0b32b34fcad6` |
| Transport and event mode | HTTP, ZMQ live and replay, incremental reports, full 32-byte SHA-256 CBOR hashes |

Rows with an earlier Router build are identified where discussed. The experiment measures stable applied-event lookup. It has no concurrent-eviction, disconnect-replay, throughput or latency acceptance result.

Use the same model artifacts for each control. Mount the target at `/models/target` and the matching speculative model at `/models/draft` for Qwen DSpark.

| Profile | Checkpoint and launch overrides |
| --- | --- |
| Qwen3-4B | `Qwen3-4B`, TP1, `--block-size 16`. The published group size was 128. |
| Qwen3.5-4B | `Qwen3.5-4B`, TP1. MTP uses `mtp` with 3 tokens. DSpark uses the Qwen3.5 Ascend draft checkpoint with 8 tokens and context 4,096. |
| Qwen3.6-27B | `Qwen3.6-27B-w8a8`, TP1, `--quantization ascend`. MTP uses `mtp` with 3 tokens. DSpark uses `Qwen3.6-27B-DSpark` with 8 tokens and context 4,096. |
| Qwen3.8-27B | `Qwen3.8-27B-W8A8` uses TP1 and `--quantization ascend`, with `mtp` and 3 tokens in the MTP arm. DSpark uses the distinct `Qwen3.8-27B` BF16 target with `Qwen3.8-27B-speculator.dspark`, TP2, 8 tokens and context 4,096. |
| Qwen4Exp | `Qwen3.8-Flash-Next-w4a8` exposes architecture `qwen4_exp`. Use TP2, `--quantization ascend --trust-remote-code --enable-expert-parallel --mamba-cache-mode align`, and `--engram-config '{"cpu_offload":true,"dp_shared_memory":false}'`. Set `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_ASCEND_PLE_BACKEND=torch`, shared memory 128 GiB and memory utilization 0.93. MTP uses `qwen3_5_mtp` with 3 tokens. Both P/D workers use `--mamba-ssm-cache-dtype bfloat16`. |
| DeepSeek mini4 | `DeepSeek-V4-Flash-0731-W8A8-four-layer-test`, a four-layer checkpoint, TP1, speculation off. Use the full Flash worker options below with TP1, memory utilization 0.80, and omit the full-checkpoint loader override. |
| DeepSeek full | `DeepSeek-V4-Flash-0731-w8a8`, a 43-layer checkpoint with 75 shards and 291.84 GiB on disk. Use the full reproduction below. |

## Reproduce the matrix

### 1. Build and start the workers

On Linux AArch64, build the pinned Router with `cargo build --release --locked --bin vllm-router`. Mount it and its runtime libraries, including libzmq as needed, into the pinned Ascend image. Use the [Ascend A2 container setup](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/tutorials/models/DeepSeek-V4-Flash.md#41-docker-image). The base run uses host networking and 24 GiB shared memory. Use fresh processes, output directories and per-worker event topics per scenario. Reserve all HTTP, ZMQ and connector endpoints before launching workers.

This is the common worker command. Set `EXTRA` to the profile arguments above. For speculation off, omit `--speculative-config`. For MTP append the method and 3 tokens. For Qwen DSpark append `--speculative-config '{"method":"dspark","num_speculative_tokens":8,"model":"/models/draft"}'`.

~~~bash
# Set these for each worker. Allocate separate NPU sets for each P and D.
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_SERVER_DEV_MODE=1
MODEL=qwen35-a2
WORKER_PORT=28080 LIVE_PORT=28081 REPLAY_PORT=28082
TOPIC="a2-$(date +%s%N)"
TP=1 CONTEXT=10240 UTILIZATION=0.80
EXTRA=()

EVENTS="$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:%s","replay_endpoint":"tcp://127.0.0.1:%s","topic":"%s","buffer_steps":10000}' \
    "$LIVE_PORT" "$REPLAY_PORT" "$TOPIC")"

vllm serve /models/target --host 127.0.0.1 --port "$WORKER_PORT" \
    --served-model-name "$MODEL" --tensor-parallel-size "$TP" \
    --max-model-len "$CONTEXT" --max-num-batched-tokens 4096 --max-num-seqs 4 \
    --gpu-memory-utilization "$UTILIZATION" --enforce-eager --skip-mm-profiling \
    --enable-prefix-caching --enable-prompt-tokens-details --enable-scale-out \
    --prefix-caching-hash-algo sha256_cbor --kv-events-config "$EVENTS" \
    "${EXTRA[@]}"
~~~

For P/D, preserve the same model name and corresponding layout across workers. Append `--kv-transfer-config "$TRANSFER"` to the worker command. Set `CONNECTOR` from the results table and `ROLE` to `kv_producer` on each Prefill worker or `kv_consumer` on Decode. Use that worker's unique `KV_PORT` and `TOPIC`. `TP` is per worker.

~~~bash
CONNECTOR=MooncakeConnectorV1
ROLE=kv_producer
KV_PORT=28090
TRANSFER="$(printf '{"kv_connector":"%s","kv_buffer_device":"npu","kv_role":"%s","kv_port":%s,"engine_id":"%s","kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":%s},"decode":{"dp_size":1,"tp_size":%s}}}' \
    "$CONNECTOR" "$ROLE" "$KV_PORT" "$TOPIC" "$TP" "$TP")"
# Append before starting the vLLM process.
EXTRA+=(--kv-transfer-config "$TRANSFER")
~~~

Initialize the Ascend toolkit and network environment shown in the full Flash recipe for HCCL and Mooncake. Wait for `GET /health` on all engines. Check `/metrics` for the expected speculative method when enabled. At the end of each scenario, stop all of its worker and Router processes before reusing NPUs, ports or cache directories.

### 2. Declare the measured cache layout

Capture a replayed `BlockStored` from a directly warmed worker. Check its 32-byte hash against `sha256(cbor2.dumps([parent, token_ids, extra_keys], canonical=True))` using the actual parent and extra keys. The zero seed uses `sha256(cbor2.dumps("0", canonical=True))` as the first parent. Read group order and block sizes from the running engine's KV cache configuration and cross-check the emitted groups. On this A2 runtime the measured hash lookahead was **0**, including the enabled MTP and DSpark arms, with `tail_drop_groups: []` and `partial_hash_hits: false`.

| Profile | Hash unit | KV cache groups in group ID order |
| --- | ---: | --- |
| Qwen3-4B | 128 | G0 full attention |
| Qwen3.5, off or MTP | 640 | G0 full, G1–G3 aligned Mamba |
| Qwen3.5, DSpark | 640 | G0–G1 full, G2 sliding window 2,048, G3–G7 aligned Mamba |
| Qwen3.6, all three modes | 896 | G0 full, G1–G3 aligned Mamba |
| Qwen3.8 W8A8, off or MTP | 896 | G0 full, G1–G3 aligned Mamba |
| Qwen3.8 BF16, DSpark | 896 | G0–G3 full, G4 sliding window 2,048, G5–G14 aligned Mamba |
| Qwen4Exp, off / MTP | 1,664 colocated, 896 P/D | G0 scratch, G1 full, G2–G5 aligned Mamba, with G6 aligned Mamba added for MTP. Scratch is 4 tokens off and 8 with MTP. |
| DeepSeek mini4 | 8 | G0/G1 sparse MLA pages at 512/16,384, G2–G5 sliding-window MLA pages at 128 with window 128, G6 page/window 8, G7 page 32 with window 128 |
| DeepSeek full | 8 | Six groups specified in the full Flash recipe |

Qwen groups use the hash-unit page size except scratch. Use `FullAttentionSpec`, `SlidingWindowSpec`, `MambaSpec`, and `CircularBufferSpec` for full, sliding-window, Mamba and scratch, respectively. Set `prefix_cacheable: false` for scratch. Qwen3.5, Qwen3.6 and Qwen3.8 Mamba groups use `mamba_cache_mode: "align"` and `prefill_checkpoint_alignment` equal to the group page size. The Qwen4Exp native entries use aligned Mamba mode and leave the checkpoint alignment unset. Use the runtime's group IDs even if a group contributes no reusable tokens.

For each observed worker URL, use the same per-worker `endpoints`, `replay_endpoints` and `ranks` shape as the full Flash example below. Set rank 0, the actual served model, the matching topic, `hash_algorithm: "sha256_cbor"`, `hash_seed: "0"`, and `full_hashes: true`. Set `declared_groups` to the group count and `speculative_tokens` to the launch value. For P/D, place `compatible_decode_urls` beside the Prefill entry's `ranks` and list the measured Decode URL.

Start the Router after directly priming each Prefill or colocated worker. The verified Router runs supplied an explicit layout for every profile, including the full-attention control.

~~~bash
export RUST_LOG='info,vllm_router_rs::kv_index=debug,vllm_router_rs::policies::kv_aware=debug'
ROUTER=/opt/router/vllm-router
COMMON=(--host 127.0.0.1 --port 28083 --prometheus-port 28084 --kv-hybrid-layout layout.json)

# Colocated
"$ROUTER" "${COMMON[@]}" --worker-urls http://127.0.0.1:28080 --policy kv_aware

# P/D, in a separate run. For 2P1D repeat --prefill with a second Prefill URL.
"$ROUTER" "${COMMON[@]}" --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:28080 --decode http://127.0.0.1:28180 \
    --prefill-policy kv_aware --decode-policy round_robin --kv-connector mooncake
~~~

### 3. Run and score the probes

1. Generate token IDs with Python's `random.Random(seed)`, calling `randrange(700, 1200)` once for each token. Warm each observed worker directly with seed 7900 for the first worker and 7901 for the second, `max_tokens: 2`. Use the warm length in the next table. Send explicit token IDs to `POST /v1/completions` with `temperature: 0` and `vllm_xargs: {"kv_cache_report_mode":"incremental"}`.
2. Start the Router and wait for `replay sequence advanced` for each warmed source and rank. Send warm, boundary and divergent prompts through Router with `max_tokens: 8`. Use the same seed as the priming request for warm, seed 7900 for the boundary probe, and seed 8900 with the warm length for divergent. Run these in order. Verify the chosen worker and `credited_prefix_tokens` in Router's decision log.
3. Compare against the selected worker's `usage.prompt_tokens_details.cached_tokens` using the same request ID. On P/D, record the Prefill response before forwarding to Decode and inspect its cached-token count. Save the Decode request ID, transfer payload and connector logs. Compare the received and forwarded transfer objects before counting actual reads. Use connector completion without a pairing or read error for that ID. The scheduler's external-token count gives the requested amount. The scheduler's `Finished recving` line alone can also follow a failed connector operation.
4. Run short plain and chat prompts in JSON and SSE, asking for `17 * 19`. Confirm that SSE ends with `[DONE]`. The longer chat smoke probe repeated 430 numbered records stating that a green ticket showed 323, or 200 records in the shorter DSpark context, then asked for the number and supporting records. Record the response and finish reason.
5. For enabled speculation, capture `vllm:spec_decode_num_draft_tokens_total` before and after a new explicit-ID request, using seed 9700, a length of one lookup unit plus 19, `max_tokens: 48`, and `ignore_eos: true`. Read the colocated worker counter, or the Decode worker counter for P/D. Require a positive delta. The full Flash recipe supplies its own length.

| Profile | Warm prompt tokens | Boundary prompt tokens |
| --- | ---: | ---: |
| Qwen3-4B | 391 | 257 |
| Qwen3.5 | 1,927 | 1,281 |
| Qwen3.6 and Qwen3.8 | 2,695 | 1,793 |
| Qwen4Exp colocated | 4,999 | 3,329 |
| Qwen4Exp P/D | 2,695 | 1,793 |
| DeepSeek mini4 and full | 16,401 | 16,385 |

## P/D failure reproduction

### Ascend MooncakeHybridConnector can fail after a correct Prefill decision

For this comparison, keep the target checkpoint, context and group layout fixed, use `MooncakeHybridConnector` on both engines, and follow the P/D instructions above. The Router still uses its `mooncake` transfer mode. Run the warm and boundary probes, followed by short JSON/SSE probes and a long chat prompt. Capture all connector errors and transfer completions by Decode request ID. Some warm requests completed even when a later short or long request failed.

| A2 comparison | Router build | Prefill warm credit / cached | Equal payloads | Confirmed reads | Connector failure in Decode |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.5-4B, MTP, 1P1D | `b65f4e5` | 1,920 / 1,920 | 9/9 | 2 | `ValueError`, shapes `(0,) (3,)` and `(13,) (3,)` |
| Qwen3.8-27B W8A8, MTP, 1P1D | `b65f4e5` | 2,688 / 2,688 | 9/9 | 2 | `ValueError`, shapes `(0,) (3,)` and `(9,) (3,)`; one trace also hit `IndexError` |
| Qwen3.8-27B W8A8, MTP, 2P1D | `3ff5e6d` | 2,688 / 2,688 on each P | 10/10 | 3 | `ValueError`, shapes `(0,) (3,)` and `(9,) (3,)` |
| Qwen3.8-27B W8A8, speculation off, 1P1D | `b65f4e5` | 2,688 / 2,688 | 8/8 | 4 | `ValueError`, shapes `(3,) (0,)` and `(9,) (0,)` |

The traceback reaches `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`, where `group_concurrent_contiguous` compares adjacent positions in the Prefill and Decode block lists before issuing a read. The traced Qwen3.8 failures started at group **1**, the first Mamba group. Group 0 is full attention. Groups 2 and 3 are also Mamba.

~~~text
Qwen3.8 MTP, direct worker-to-worker control, group 1
  Prefill block IDs in the handoff   [54]
  Decode block IDs passed to pairing [35, 38, 39, 40]
  ValueError: operands could not be broadcast together with shapes (0,) (3,)

Qwen3.8 speculation off, routed warm probe, group 1
  Prefill block IDs in the handoff   [0, 0, 0, 15]
  Decode block IDs passed to pairing [5]
  ValueError: operands could not be broadcast together with shapes (3,) (0,)
~~~

The enabled trace shows a trimmed Prefill Mamba list and additional Decode allocations for the three draft slots. The disabled trace shows an independent mismatch between the Prefill list that still includes alignment padding and the Decode list with that padding removed. The diagnostic control sent a 21-token explicit-ID prompt to Prefill, then sent a Decode request with the returned `kv_transfer_params` unchanged and the returned `last_token_id` appended to the prompt. Use the same one-token Prefill and eight-token Decode budgets as the control. It reproduced the group-1 broadcast error while both workers returned HTTP 200 and no matching remote read completed. Instrument the connector immediately before grouping and in the exception handler to capture the two arrays and group IDs. Keep the diagnostic observer read-only.

A Prefill credit match validates the Router's prediction of what Prefill can reuse for that request. The matching handoff validates forwarding. The direct control localizes this failure to Ascend's connector pairing. `MooncakeConnectorV1` provided the successful Qwen comparison, and different models can have different compatible connectors, as the mini4 result shows.

### Earlier A2 controls

| Control | Observed result |
| --- | --- |
| Qwen3.5 MTP and Qwen3.8 BF16 DSpark with the earlier A2 image | Verified captured full-width hashes and positive speculation counters on Router `b9875d5`. A shorter prompt initially matched at 0 / 0, then reached 1,280 / 1,280 for Qwen3.5 or 1,792 / 1,792 for Qwen3.8 after directly warming that shorter shape. A fork after a shared page and an unrelated prefix both matched at 0 / 0. |
| Qwen3.5 MTP and Qwen3.8 DSpark, earlier 1P1D MooncakeConnectorV1 | Four transfer probes per model completed with unchanged payloads. The first one-token-over-boundary probe overestimated local Prefill reuse on `b9875d5` (1,280 / 0 and 1,792 / 0). The pinned runs in the matrix above matched at 0 / 0 on that first boundary probe. |
| Qwen3.5 MTP, earlier 1P1D MooncakeHybridConnector | Warm long prompt matched at 1,920 / 1,920 with a read. Warm short prompt matched at 640 / 640 but hit `IndexError` in `mooncake_hybrid_connector.py`. The distinction also appeared on the 0.32.0rc1 image. |
| Qwen3.5 and Qwen3.8 W8A8, earlier 2P1D | The warmed prefixes selected the correct Prefill workers. The Qwen3.5 and Qwen3.8 counts were 1,920 and 2,688 tokens per worker, respectively. The earliest matrix used Ascend `0.19.1rc2.dev2959+g450e55698` with vLLM `0.32.0+empty` and torch-npu `2.10.0.post4`. |
| Qwen3-4B, first two-worker attempt | Router credit was initially 0 for positive probes. After waiting for applied warm events, the repeated run matched at 384 / 384 for the warm prompt and 256 / 256 at the boundary. |
| DeepSeek mini4, startup and boundary controls | The earlier image failed worker initialization with `KeyError: 8`. Other startup attempts encountered an HCCL network error. On 0.32.0rc1, distinct fresh-run probes established 16,384 / 16,384 warm, 0 / 0 on the first shorter P/D boundary probe, and 16,384 / 16,384 after warming the shorter shape. Raising the context to 65,536 makes the 16,384-token positive lookup possible. |
| Qwen4Exp TP4 exploration | One no-MTP case failed the fused QSA norm/RoPE ABI. A TP4 no-MTP run with native MRoPE produced 2,688 / 2,688 at warm and 0 / 0 for a different prefix on `4c7bc5b`. TP4 with MTP hit the worker startup error `Qwen4 hybrid page mismatch: view=2048, region=2240`. The TP2 runs above use native fused MRoPE and retain their own measured layout. |
| Qwen3.6 DSpark 1P1D and full Flash startup | One Qwen3.6 attempt had 28.31 GiB free on a card at startup, below the 48.77 GiB requested. A clean retry completed. Full Flash initially provided 6.58 GiB of KV space against 9.36 GiB required at the chosen context. Both final arms used utilization 0.90 on freed cards. |

A separate 910C report used a different Ascend checkout. It declared one token of hash lookahead from scheduler settings even though the captured `BlockStored` preimage used zero. It also observed failures in block pairing. For the A2 profiles above, the layout follows observed group descriptors and event preimages. A scheduler lookahead or speculative-token setting alone does not determine the tokens encoded into a published block hash. [Independent 910C record](https://github.com/xuxiongchen/router/blob/52c389025d7279b84bc75c28cc1b48e01b0dfbf6/docs/benchmarks/pr-493-ascend-910c-validation.md)

## Full DeepSeek V4 Flash reproduction

This recipe checks both speculative modes with the complete checkpoint. The partial-page probes add one and two tokens after the 16,384-token common lookup boundary.

| Input | Prompt tokens | DSpark off, Router / engine | DSpark on, Router / engine |
| --- | ---: | ---: | ---: |
| Warm prefix with a short tail | 16,401 | 16,384 / 16,384 | 16,384 / 16,384 |
| One token past the reuse boundary | 16,385 | 16,384 / 16,384 | 16,384 / 16,384 |
| Two tokens past the reuse boundary | 16,386 | 16,384 / 16,384 | 16,384 / 16,384 |
| Different prefix | 16,401 | 0 / 0 | 0 / 0 |

For each row, the worker response and `KVCacheManager.get_computed_blocks` agreed. The Router applied replay sequence 0 before probing. With DSpark enabled, the separate 16,403-token activity probe matched 0 / 0 and produced 224 draft tokens.

Obtain the full W8A8 checkpoint from the [vLLM Ascend recipe](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/tutorials/models/DeepSeek-V4-Flash.md#31-model-weight). The measured `config.json` SHA-256 is `6c6cdc4a47e00e41137081429b95bd5c1cf91ba62e87309d1398431de7f4a8e1`. The `quant_model_weights.safetensors.index.json` SHA-256 is `fe34647052b65127b338d86a22c97381e8d7f002b76bb96d7f5cf717140d6570`. Mount the 75 shards at `/models/target`, Router at `/opt/router/vllm-router`, and a writable directory at `/run-a2`.

### 1. Record the engine lookup

Save the following as `/run-a2/probe/sitecustomize.py` before starting vLLM. The observer records the returned token count and leaves the engine result intact.

~~~python
import functools
import json
import os
from pathlib import Path

if path := os.environ.get("A2_LOOKUPS"):
    from vllm.v1.engine.core import EngineCore

    initialize = EngineCore.__init__

    @functools.wraps(initialize)
    def with_lookup_observer(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        manager = self.scheduler.kv_cache_manager
        lookup = manager.get_computed_blocks

        @functools.wraps(lookup)
        def observe(request, *args, **kwargs):
            result = lookup(request, *args, **kwargs)
            with Path(path).open("a") as log:
                log.write(json.dumps({
                    "request_id": request.request_id,
                    "prompt_tokens": request.num_prompt_tokens,
                    "computed_tokens": result[1],
                }) + "\n")
            return result

        manager.get_computed_blocks = observe

    EngineCore.__init__ = with_lookup_observer
~~~

### 2. Start the worker

Run each arm with a fresh engine, Router process, output directory and event topic. Set `DSPARK=0` for the control and `DSPARK=1` for the enabled arm. Use these same variables in the worker and probe terminals. Reserve ports 28080–28084 or change them consistently.

~~~bash
export DSPARK=0
export RUN="/run-a2/dspark-$DSPARK"
export MODEL="deepseek-v4-flash-w8a8"
export WORKER_PORT=28080 LIVE_PORT=28081 REPLAY_PORT=28082
export ROUTER_PORT=28083 METRICS_PORT=28084
mkdir -p "$RUN"
test -f "$RUN/topic" || printf 'a2-flash-%s\n' "$(date +%s%N)" > "$RUN/topic"
export TOPIC="$(cat "$RUN/topic")"
~~~

In the worker terminal, initialize the container's Ascend toolkit environment and run the following command.

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
for script in /usr/local/Ascend/cann-*/share/info/ascendnpu-ir/bin/set_env.sh; do
    [[ ! -f "$script" ]] || source "$script"
done
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
fi
NIC="$(ip -4 route show default | awk 'NR == 1 {print $5}')"
export HCCL_IF_IP="$(ip -4 -o addr show dev "$NIC" scope global | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}')"
export HCCL_SOCKET_IFNAME="$NIC" GLOO_SOCKET_IFNAME="$NIC" TP_SOCKET_IFNAME="$NIC"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384
export PYTHONHASHSEED=0 VLLM_SERVER_DEV_MODE=1 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
export PYTHONPATH="/run-a2/probe${PYTHONPATH:+:$PYTHONPATH}"
export A2_LOOKUPS="$RUN/lookup.jsonl"

EVENTS="$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:%s","replay_endpoint":"tcp://127.0.0.1:%s","topic":"%s","buffer_steps":10000}' \
    "$LIVE_PORT" "$REPLAY_PORT" "$TOPIC")"

spec=()
if [[ "$DSPARK" == 1 ]]; then
    spec=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}')
fi

vllm serve /models/target --host 127.0.0.1 --port "$WORKER_PORT" \
    --served-model-name "$MODEL" --tensor-parallel-size 8 --enable-expert-parallel \
    --quantization ascend --tokenizer-mode deepseek_v4 \
    --max-model-len 65536 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 --enforce-eager --skip-mm-profiling \
    --block-size 128 --attention_config.indexer_kv_dtype int8 \
    --no-disable-hybrid-kv-cache-manager --prefix-cache-retention-interval 16384 \
    --enable-prefix-caching --enable-prompt-tokens-details --enable-scale-out \
    --prefix-caching-hash-algo sha256_cbor --kv-events-config "$EVENTS" \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
    "${spec[@]}" > "$RUN/worker.log" 2>&1
~~~

Wait for `GET /health` to return 200 on the worker. Confirm that each assigned NPU is free before starting. During the first attempt, a 0.80 memory utilization limit left 6.58 GiB for KV cache, below the 9.36 GiB required for the configured 65,536-token context. Both successful arms used 0.90. Restarting before the previous container released its allocations also caused a startup memory error.

### 3. Declare the observed layout and run the probes

Group sizes are in prompt tokens. The hash unit was **8**, the common lookup alignment was **16,384**, and the runtime exposed the same six groups in both modes.

| Group | Cache group | Page size | Tokens per state | Sliding window |
| ---: | --- | ---: | ---: | ---: |
| 0 | Sparse compressed MLA | 512 | 4 | — |
| 1 | Sparse compressed MLA | 16,384 | 128 | — |
| 2 | Sliding-window MLA | 128 | 1 | 128 |
| 3 | Sliding-window MLA | 128 | 1 | 128 |
| 4 | Sliding-window MLA | 8 | 1 | 8 |
| 5 | Sliding-window MLA | 32 | 1 | 128 |

Save the following as `/run-a2/check.py` and run `python /run-a2/check.py` from the probe terminal with the same exported variables as the worker. It warms the engine directly, starts the Router and waits for the replayed Store before issuing scored requests. The measured runs used a transparent localhost recorder between the Router and vLLM to retain the worker response IDs. This reproduction uses the direct worker URL.

~~~python
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import requests

run = Path(os.environ["RUN"])
dspark = os.environ["DSPARK"] == "1"
worker = f"http://127.0.0.1:{os.environ['WORKER_PORT']}"
router = f"http://127.0.0.1:{os.environ['ROUTER_PORT']}"
model = os.environ["MODEL"]

def tokens(length, seed):
    rng = random.Random(seed)
    return [rng.randrange(700, 1200) for _ in range(length)]

def complete(server, prompt, max_tokens=8, **kwargs):
    response = requests.post(
        server + "/v1/completions",
        json=dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0,
                  vllm_xargs={"kv_cache_report_mode": "incremental"}, **kwargs),
        timeout=360,
    )
    response.raise_for_status()
    return response.json()

groups = []
for group, (size, ratio, window) in enumerate([
    (512, 4, None), (16384, 128, None), (128, 1, 128),
    (128, 1, 128), (8, 1, 8), (32, 1, 128),
]):
    entry = dict(group_idx=group, block_size=size, tokens_per_state=str(ratio),
                 prefix_cacheable=True, prefix_replay_tokens=0)
    if window is None:
        entry.update(kind="mla_attention", spec_type="MLAAttentionSpec", cache_role="sparse")
    else:
        entry.update(kind="sliding_window_mla", spec_type="SlidingWindowMLASpec",
                     sliding_window=window)
    groups.append(entry)
layout = {
    worker: dict(
        schema_version=1,
        endpoints=[f"tcp://127.0.0.1:{os.environ['LIVE_PORT']}"],
        replay_endpoints=[f"tcp://127.0.0.1:{os.environ['REPLAY_PORT']}"],
        ranks=[dict(
            rank=0, model=model, weight_version="default",
            publisher_topic=os.environ["TOPIC"], instance_topic=True,
            hash_algorithm="sha256_cbor", hash_seed="0", full_hashes=True,
            hash_block_size=8, hash_lookahead_tokens=0, partial_hash_hits=False,
            speculative_tokens=7 if dspark else 0, tail_drop_groups=[],
            prefix_caching=True, declared_groups=len(groups), groups=groups,
        )],
    )
}
(run / "layout.json").write_text(json.dumps(layout, indent=2))
complete(worker, tokens(16401, 7900), max_tokens=2)

log_path = run / "router.log"
with log_path.open("w") as stdout:
    process = subprocess.Popen([
        "/opt/router/vllm-router", "--host", "127.0.0.1",
        "--port", os.environ["ROUTER_PORT"],
        "--prometheus-port", os.environ["METRICS_PORT"],
        "--worker-urls", worker, "--policy", "kv_aware",
        "--kv-hybrid-layout", str(run / "layout.json"),
    ], env=dict(os.environ, RUST_LOG="info,vllm_router_rs::kv_index=debug,"
                                      "vllm_router_rs::policies::kv_aware=debug"),
       stdout=stdout, stderr=subprocess.STDOUT)

def log_text():
    return re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="replace"))

def check(label, length, seed, expected, **kwargs):
    offset = len(log_text())
    reply = complete(router, tokens(length, seed), **kwargs)
    log = log_text()[offset:]
    credits = re.findall(r"credited_prefix_tokens=(\d+)", log)
    assert len(credits) == 1, (label, log)
    reported = reply["usage"]["prompt_tokens_details"]["cached_tokens"]
    request_id = reply["id"]
    lookups = [json.loads(line) for line in (run / "lookup.jsonl").read_text().splitlines()]
    matching = [row for row in lookups if row["request_id"].startswith(request_id)]
    assert matching and matching[0]["prompt_tokens"] == length, (label, request_id)
    actual = matching[0]["computed_tokens"]
    assert int(credits[0]) == reported == actual == expected, (label, credits, reported, actual)
    result = dict(case=label, request_id=request_id, prompt_tokens=length,
                  router_credit=int(credits[0]), worker_cached=reported, engine_lookup=actual)
    with (run / "samples.jsonl").open("a") as output:
        output.write(json.dumps(result) + "\n")
    print(result)

def drafted():
    response = requests.get(worker + "/metrics", timeout=10)
    response.raise_for_status()
    return sum(map(float, re.findall(
        r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
        response.text, re.M,
    )))

try:
    for _ in range(200):
        assert process.poll() is None, log_text()
        if re.search(r"replay sequence advanced[^\n]*rank=0 seq=0\b", log_text()):
            break
        time.sleep(0.25)
    else:
        raise TimeoutError("Router did not apply the rank-0 warm-up event")

    check("warm", 16401, 7900, 16384)
    check("boundary+1", 16385, 7900, 16384)
    check("boundary+2", 16386, 7900, 16384)
    check("divergent", 16401, 8900, 0)
    if dspark:
        before = drafted()
        check("dspark-activity", 16403, 9700, 0, max_tokens=48, ignore_eos=True)
        delta = drafted() - before
        assert delta > 0, delta
        print(f"DSpark drafted {delta:g} tokens")
finally:
    process.terminate()
    process.wait(timeout=30)
~~~

Stop the worker after each arm. The acceptance signal is equality between the Router estimate and both engine observations on every scored request. For the enabled arm, also require a positive draft-token delta. The exact count can vary with token acceptance.

### 4. Recorded request IDs

These request IDs join the original worker response to the engine lookup. The result hashes identify the corresponding captured runs.

| Case | DSpark off request ID | DSpark on request ID |
| --- | --- | --- |
| Warm | `cmpl-95bd3fba2a959687` | `cmpl-b0e150c878feb68e` |
| Boundary + 1 | `cmpl-a913e27ebf8d7631` | `cmpl-bd17cfa4f84668b9` |
| Boundary + 2 | `cmpl-83700d9dd38f07fe` | `cmpl-99738e930cc11a7c` |
| Divergent | `cmpl-a78379b130355972` | `cmpl-aee84f23eb870517` |
| DSpark activity | — | `cmpl-89db3ab5060874c9` |

- DSpark off `results.json` SHA-256. `c97252fe09629bfb303baaa8b0bd0b0e18dab027cc8c4fdb5fcdc853e8b88042`
- DSpark on `results.json` SHA-256. `065ccaa35cf56501e2a07843f8f8ca65e629248a767589a4a82cffe549716207`
- Both native layouts SHA-256. `4d6064bcef930ac13183d3ec1edaaef37680d26e85b2cbc3e80f08e048468227`
