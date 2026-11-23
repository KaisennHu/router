"""Run one AgentInfer AgentX replay arm of the #519 slice on fixed P/D workers."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(os.environ.get("ROUTER_POLICY_ROOT", "/home/david_cwq/hhc/pr519-h20-20261123"))
PORTS = [int(p) for p in os.environ.get("PORTS", "").split(",") if p]
PREFILL = 4
ROUTER_PORT = int(os.environ["ROUTER_PORT"])
METRICS_PORT = int(os.environ["METRICS_PORT"])
AGENT_INFER = Path(os.environ.get("AGENT_INFER_SRC", "/home/david_cwq/hhc/e2e-9f35426-2026-10-13/agent-infer-main"))
ARM_BIN = {"cache": "router-pr493-6db7970", "main_cache": "router-main-e8ce7ea",
           "current_kv": "router-pr493-6db7970",
           "tiebreak": "router-pr493-tiebreak",
           "release_kv": "router-v0.1.17", "release_fixed": "router-v0.1.17-event-fix"}
P = argparse.ArgumentParser()
P.add_argument("arm", choices=ARM_BIN)
P.add_argument("label")
args = P.parse_args()
out = ROOT / "runs" / args.label
out.mkdir(parents=True, exist_ok=False)
trace = Path(os.environ.get("TRACE_PATH", str(ROOT / "data" / "agentx-pr519-12x8.jsonl")))
binary = ROOT / "bin" / ARM_BIN[args.arm]
policy = "cache_aware" if args.arm in ("cache", "main_cache") else "kv_aware"
router_cmd = [str(binary), "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
              "--prometheus-host", "127.0.0.1", "--prometheus-port", str(METRICS_PORT),
              "--vllm-pd-disaggregation", "--kv-connector", "nixl", "--policy", policy,
              "--prefill-policy", policy, "--decode-policy", policy,
              "--worker-startup-check-interval", "1", "--request-timeout-secs", "3600",
              "--router-temperature", "0", "--cache-threshold", "0.3",
              "--balance-abs-threshold", "2", "--balance-rel-threshold", "1.5",
              "--kv-penalty-host", "0.8", "--kv-penalty-external", "0.6", "--kv-penalty-disk", "0.3"]
for port in PORTS[:PREFILL]:
    router_cmd += ["--prefill", f"http://127.0.0.1:{port}"]
for port in PORTS[PREFILL:]:
    router_cmd += ["--decode", f"http://127.0.0.1:{port}"]
config = {
    "experiment": {"task_num": 12, "max_concurrency": 12,
                   "task_timeout_seconds": 3600, "run_timeout_seconds": 10800},
    "backend": {"type": "vllm", "base_url": f"http://127.0.0.1:{ROUTER_PORT}",
                "metrics_url": f"http://127.0.0.1:{METRICS_PORT}/metrics",
                "tokenizer_base_url": f"http://127.0.0.1:{PORTS[0]}",
                "model": "qwen3", "endpoint": "/v1/completions"},
    "replay": {"trace_type": "agentX", "trace_path": str(trace),
               "trace_same_agent_gap_scale": 0.002, "trace_same_agent_gap_offset_seconds": 0,
               "interval_mode": "trace", "sample_seed": 0, "max_inflight_requests": 1,
               "request_timeout_seconds": 3600},
}
(out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
meta = {"arm": args.arm, "trace": str(trace), "router_command": router_cmd,
        "router_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "router_binary": str(binary),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "worker_layout": {"prefill": PREFILL, "decode": len(PORTS) - PREFILL, "ports": PORTS}}
(out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
env = os.environ.copy()
env.update(PYTHONPATH=str(AGENT_INFER), XDG_CACHE_HOME=str(ROOT / "xdg"),
           HF_HOME=str(ROOT / "hf"), TMPDIR=str(ROOT / "tmp"), HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
           RUST_LOG=os.environ.get("ARM_RUST_LOG", "info"))
env.pop("PYTHONHASHSEED", None)
log = (out / "router.log").open("w")
router = subprocess.Popen(router_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop():
    if router.poll() is None:
        os.killpg(router.pid, signal.SIGTERM)
        try:
            router.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(router.pid, signal.SIGKILL)
            router.wait()
    log.close()


def terminate(_sig, _frame):
    raise SystemExit(143)


signal.signal(signal.SIGTERM, terminate)
try:
    with httpx.Client(timeout=10) as client:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if router.poll() is not None:
                raise RuntimeError(f"Router exited {router.returncode}; see {out}/router.log")
            try:
                if client.get(f"http://127.0.0.1:{ROUTER_PORT}/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Router not healthy")
        if args.arm not in ("cache", "main_cache"):
            expected = PORTS[:PREFILL] if args.arm.startswith("release") else PORTS
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                text = (out / "router.log").read_text(errors="replace")
                if all(f"kv_index http://127.0.0.1:{port}: spawned subscriber" in text for port in expected):
                    break
                if router.poll() is not None:
                    raise RuntimeError("Router exited while waiting for KV subscribers")
                time.sleep(1)
            else:
                raise TimeoutError(f"Missing expected KV subscriptions for {expected}")
        for port in PORTS:
            for _attempt in range(30):
                response = client.post(f"http://127.0.0.1:{port}/reset_prefix_cache")
                response.raise_for_status()
                if response.json().get("success"):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"Prefix cache did not reset on {port}")
        time.sleep(2)
        before = out / "before"
        before.mkdir()
        for port in PORTS:
            (before / f"{port}.prom").write_text(client.get(f"http://127.0.0.1:{port}/metrics").text)
        (before / "router.prom").write_text(client.get(f"http://127.0.0.1:{METRICS_PORT}/metrics").text)
    command = ["/home/david_cwq/hhc/kv-unified-vllm033-runtime/bin/python", "-u", "-c",
               "import sys; from agentinfer.agentcache.entrypoints.bench import main; "
               "sys.exit(main(['vllm', *sys.argv[1:]]))",
               "bench", "serve", "--agentinfer", "replay", "--config", str(out / "config.yaml"),
               "--result-dir", str(out / "replay")]
    (out / "replay-command.json").write_text(json.dumps(command, indent=2))
    started = time.time()
    with (out / "replay.log").open("w") as replay_log:
        replay = subprocess.Popen(command, env=env, stdout=replay_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            replay.wait(timeout=11000)
        finally:
            if replay.poll() is None:
                os.killpg(replay.pid, signal.SIGTERM)
                try:
                    replay.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(replay.pid, signal.SIGKILL)
                    replay.wait()
    ended = time.time()
    time.sleep(5)
    with httpx.Client(timeout=10) as client:
        after = out / "after"
        after.mkdir()
        for port in PORTS:
            (after / f"{port}.prom").write_text(client.get(f"http://127.0.0.1:{port}/metrics").text)
        (after / "router.prom").write_text(client.get(f"http://127.0.0.1:{METRICS_PORT}/metrics").text)
    meta.update(replay_returncode=replay.returncode, started_unix=started, ended_unix=ended,
                wall_seconds=ended - started)
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    rows = [json.loads(x) for x in (out / "replay/requests.jsonl").read_text().splitlines() if x.strip()]
    failures = [r for r in rows if r.get("status") != "success"]
    if replay.returncode or failures:
        raise RuntimeError(f"Replay returned {replay.returncode} with {len(failures)} failures: "
                           f"{[(f.get('status_code'), (f.get('error') or '')[:120]) for f in failures]}")
    print(f"ARM_COMPLETE {out} requests={len(rows)} wall_seconds={ended - started:.2f}", flush=True)
finally:
    stop()
