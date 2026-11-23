"""Start the eight fixed P/D engines (4 Prefill + 4 Decode) for the #519 slice.

Same vLLM 0.33.0 pairing as the B300 run; the only deviation from the #519
description is ``--max-model-len 163840`` instead of 131072: one replayed request
has a planned context of 132,674 tokens, which the engine truncated to a
max_tokens below the request's min_tokens and rejected with HTTP 400. 163,840
keeps the 26 GiB KV budget check satisfied (163,840 x 144 KiB = 22.5 GiB).
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(os.environ.get("ROUTER_POLICY_ROOT", "/home/david_cwq/hhc/pr519-h20-20261123"))
VLLM = "/home/david_cwq/hhc/kv-unified-vllm033-runtime/bin/vllm"
MODEL = "/home/david_cwq/hhc/models/Qwen3-4B-Instruct-2507"
# vLLM 0.33 on this node imports the NIXL 1.5.2 wheel from this prefix; without it
# NixlConnector is unavailable (ModuleNotFoundError) exactly as in the PR #493
# H20 acceptance run.
NIXL_PYTHONPATH = "/home/david_cwq/.cache/codex-hybrid-validation/nixl152"
CUDA_HOME = "/usr/local/cuda"
PORTS = [int(p) for p in os.environ.get("PORTS", "").split(",") if p]
if len(PORTS) != 8:
    raise SystemExit(f"PORTS must list 8 engine ports, got {PORTS}")
PREFILL = 4
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "163840")
KV_BYTES = os.environ.get("KV_BYTES", "27917287424")
procs, logs = [], []
worker_root = ROOT / "workers" / os.environ.get("RUN_LABEL", "manual")
worker_root.mkdir(parents=True, exist_ok=True)
for directory in ["tmp", "hf", "xdg", "vllm", "triton", "cuda", "torchinductor", "flashinfer"]:
    (ROOT / directory).mkdir(exist_ok=True)


def terminate(_signum, _frame):
    raise SystemExit(143)


signal.signal(signal.SIGTERM, terminate)
try:
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if not gpus:
        gpus = [str(i) for i in range(len(PORTS))]
    if len(gpus) < len(PORTS):
        raise RuntimeError(f"need {len(PORTS)} GPUs, got {gpus}")
    for port in [p + offset for p in PORTS for offset in range(4)] + [PORTS[0] + 200, PORTS[0] + 201]:
        deadline = time.monotonic() + 120
        while True:
            try:
                with socket.socket() as sock:
                    sock.bind(("0.0.0.0", port))
                break
            except OSError as error:
                if error.errno != 98 or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    for i, port in enumerate(PORTS):
        work = worker_root / f"worker-{i}"
        (work / "tmp").mkdir(parents=True, exist_ok=True)
        role = "kv_producer" if i < PREFILL else "kv_consumer"
        events = {"enable_kv_cache_events": True, "publisher": "zmq",
                  "endpoint": f"tcp://*:{port+1}", "replay_endpoint": f"tcp://127.0.0.1:{port+2}",
                  "buffer_steps": 10000, "topic": f"pr519-{os.getpid()}-{i}"}
        transfer = {"kv_connector": "NixlConnector", "kv_role": role,
                    "kv_connector_extra_config": {"backends": ["UCX"]}}
        cmd = [VLLM, "serve", MODEL,
               "--served-model-name", "qwen3", "--host", "127.0.0.1", "--port", str(port),
               "--max-model-len", MAX_MODEL_LEN, "--max-num-seqs", "8",
               "--max-num-batched-tokens", "32768", "--dtype", "bfloat16",
               "--gpu-memory-utilization", "0.80", "--kv-cache-memory-bytes", KV_BYTES,
               "--enable-prefix-caching", "--enable-prompt-tokens-details",
               "--prefix-caching-hash-algo", "sha256_cbor", "--block-size", "16",
               "--enable-request-id-headers", "--kv-events-config", json.dumps(events),
               "--kv-transfer-config", json.dumps(transfer)]
        env = os.environ.copy()
        env.pop("PYTHONHASHSEED", None)
        env["PYTHONPATH"] = NIXL_PYTHONPATH
        env.update(CUDA_VISIBLE_DEVICES=gpus[i], VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES="0",
                   VLLM_SERVER_DEV_MODE="1", HF_HUB_OFFLINE="1", CUDA_HOME=CUDA_HOME,
                   PATH=f"{Path(VLLM).parent}:{CUDA_HOME}/bin:" + env.get("PATH", "/usr/bin:/bin"),
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                   VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1", VLLM_NIXL_SIDE_CHANNEL_PORT=str(port+3),
                   VLLM_RPC_BASE_PATH=f"@pr519-{os.getpid()}-{i}", TMPDIR=str(work / "tmp"),
                   HF_HOME=str(ROOT / "hf"), XDG_CACHE_HOME=str(ROOT / "xdg"),
                   VLLM_CACHE_ROOT=str(ROOT / "vllm"), TRITON_CACHE_DIR=str(ROOT / "triton"),
                   CUDA_CACHE_PATH=str(ROOT / "cuda"), FLASHINFER_WORKSPACE_BASE=str(ROOT / "flashinfer"),
                   TORCHINDUCTOR_CACHE_DIR=str(ROOT / "torchinductor"))
        log = (work / "engine.log").open("w")
        logs.append(log)
        (work / "command.json").write_text(json.dumps({"argv": cmd, "gpu": gpus[i], "env": {
            k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES",
                                "VLLM_NIXL_SIDE_CHANNEL_PORT", "VLLM_RPC_BASE_PATH"]}}, indent=2))
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True))
    (worker_root / "manifest.json").write_text(json.dumps({
        "gpus": gpus, "ports": PORTS, "prefill": PREFILL, "max_model_len": MAX_MODEL_LEN,
        "kv_cache_memory_bytes": KV_BYTES, "pids": [p.pid for p in procs],
        "model": MODEL, "vllm": "0.33.0"}, indent=2))
    with httpx.Client(timeout=3) as client:
        deadline = time.monotonic() + 2400
        pending = set(range(len(PORTS)))
        while pending and time.monotonic() < deadline:
            for i in list(pending):
                if procs[i].poll() is not None:
                    raise RuntimeError(f"Worker {i} exited {procs[i].returncode}; "
                                       f"see {worker_root}/worker-{i}/engine.log")
                try:
                    if client.get(f"http://127.0.0.1:{PORTS[i]}/health").status_code == 200:
                        pending.remove(i)
                        print(f"READY worker={i} gpu={gpus[i]} port={PORTS[i]}", flush=True)
                except httpx.HTTPError:
                    pass
            time.sleep(3)
        if pending:
            raise TimeoutError(f"Workers not healthy: {sorted(pending)}")
    print("ALL_WORKERS_READY", flush=True)
    while all(p.poll() is None for p in procs):
        time.sleep(5)
    raise RuntimeError("A worker exited during the experiment")
finally:
    for proc in reversed(procs):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    for proc in reversed(procs):
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    for log in logs:
        log.close()
