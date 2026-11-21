# Reproducing KV-aware prefix reuse for DeepSeek V4 Flash on Ascend A2

On 2026-11-21, the Router matched the vLLM engine's local prefix-reuse count in all eight controlled requests on the **full** DeepSeek V4 Flash W8A8 checkpoint. The four cases ran once with DSpark enabled and once with DSpark disabled. A separate request confirmed that the enabled engine drafted tokens. Both runs used HTTP, one colocated worker and incremental KV events.

| Input | Prompt tokens | DSpark off, Router / engine | DSpark on, Router / engine |
| --- | ---: | ---: | ---: |
| Warm prefix with a short tail | 16,401 | 16,384 / 16,384 | 16,384 / 16,384 |
| Warm prefix, one token past the reuse boundary | 16,385 | 16,384 / 16,384 | 16,384 / 16,384 |
| Warm prefix, two tokens past the reuse boundary | 16,386 | 16,384 / 16,384 | 16,384 / 16,384 |
| Different prefix | 16,401 | 0 / 0 | 0 / 0 |

“Engine” is both the worker response's `usage.prompt_tokens_details.cached_tokens` and the same request's `KVCacheManager.get_computed_blocks` result. Every row had HTTP 200 and an exact match between those two engine observations. The Router applied replay sequence 0 for rank 0 before the first probe in each run. The DSpark activity probe used 16,403 prompt tokens, matched at 0 / 0 and increased the draft-token counter by **224**.

This experiment covers stable local prefix credit on the two colocated deployments. It leaves P/D transfer, concurrent eviction, performance and generation quality outside the measured scope.

## 1. Pin the environment

| Item | Tested value |
| --- | --- |
| Hardware | One Ascend A2 node, eight 64 GiB NPUs, reported as 910B1 by `npu-smi` |
| Model | `DeepSeek-V4-Flash-0731-w8a8`, 43 layers, 75 weight shards, 291.84 GiB on disk |
| Runtime image | `quay.io/ascend/vllm-ascend@sha256:ff6d1ea57df08dd9382c51d202d0dd2d9acc35d53a65bcb0dfd906886e674912` |
| Packages | `vllm 0.32.0+empty`, `vllm-ascend 0.32.0rc1`, `torch-npu 2.10.0.post7` |
| Router | [`3ff5e6d987c77bce0afcfe2b57d79ea8adf869d8`](https://github.com/KaisennHu/router/commit/3ff5e6d987c77bce0afcfe2b57d79ea8adf869d8) |
| Tested Router binary SHA-256 | `4abf9536aa17f8cbd5c2b2bdea7eb4e88100b36c1f76100a01d3756447cf75c0` |

Use the full W8A8 checkpoint from the [vLLM Ascend model recipe](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/tutorials/models/DeepSeek-V4-Flash.md#31-model-weight). The measured `config.json` SHA-256 is `6c6cdc4a47e00e41137081429b95bd5c1cf91ba62e87309d1398431de7f4a8e1`. The `quant_model_weights.safetensors.index.json` SHA-256 is `fe34647052b65127b338d86a22c97381e8d7f002b76bb96d7f5cf717140d6570`. Verify that all 75 referenced shards are present.

Build `vllm-router` from the pinned commit for Linux AArch64 with `cargo build --release --locked --bin vllm-router`. Record your own binary digest. Set up the Ascend container runtime and access to all eight NPUs following the [A2 setup](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/tutorials/models/DeepSeek-V4-Flash.md#41-docker-image). Mount the checkpoint read-only at `/models/target`, the Router binary at `/opt/router/vllm-router` and a writable run directory at `/run-a2`. Supply the Router's dynamic libraries, including libzmq, in the container's library path if required by your build. The measured container used host networking and 24 GiB of shared memory.

## 2. Record the engine lookup

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

## 3. Start the worker

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

## 4. Declare the observed layout and run the probes

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

## 5. Recorded evidence

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
