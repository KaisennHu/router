"""Per-arm trace series: run an explicit (arm, label, trace) schedule.

SCHEDULE_SPEC="arm:label:trace,arm:label:trace,..." where trace is a filename
under data/ (or an absolute path). Workload-identity verification is done per
trace file, so different slices can be compared inside one worker lifetime.
Workers start once; each arm gets a fresh Router process and a prefix-cache
reset on all engines.
"""
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/david_cwq/hhc/pr519-h20-20261123")
PYTHON = "/home/david_cwq/hhc/kv-unified-vllm033-runtime/bin/python"
TAG = os.environ.get("TAG", "few")
WORKERS = int(os.environ.get("WORKERS", "8"))
SPEC = os.environ["SCHEDULE_SPEC"]
SCHEDULE = []
# sbatch --export splits on commas, so ';' and '|' are accepted as item separators.
for item in SPEC.replace("|", ";").replace(",", ";").split(";"):
    if not item.strip():
        continue
    parts = item.strip().split(":")
    arm, label = parts[0], parts[1]
    trace = parts[2] if len(parts) > 2 and parts[2] else None
    SCHEDULE.append((arm, f"{TAG}-{label}", trace))

worker = arm_proc = log = None
expected = {}


def terminate(*_):
    raise SystemExit(143)


def stop(p):
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def port_free(port):
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


PORT_SEARCH_START = int(os.environ.get("PORT_SEARCH_START", "50000"))


def allocate_ports():
    # Start from a per-job base so two of our own jobs never race for the same
    # block, and stale engines from a crashed run cannot overlap a new one.
    for base in range(PORT_SEARCH_START, min(PORT_SEARCH_START + 6000, 65000), 100):
        ports = [base + i * 20 for i in range(WORKERS)]
        router, metrics = base + 170, base + 171
        wanted = []
        for port in ports:
            wanted.extend([port, port + 1, port + 2, port + 3])
        wanted.extend([router, metrics])
        if all(port_free(port) for port in wanted):
            return ports, router, metrics
    raise RuntimeError("no free port block")


signal.signal(signal.SIGTERM, terminate)
try:
    ports, router_port, metrics_port = allocate_ports()
    os.environ["PORTS"] = ",".join(str(p) for p in ports)
    os.environ["ROUTER_PORT"] = str(router_port)
    os.environ["METRICS_PORT"] = str(metrics_port)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"{TAG}-ports.json").write_text(json.dumps(
        {"ports": ports, "router": router_port, "metrics": metrics_port}, indent=2))
    print(f"PORTS {os.environ['PORTS']} router={router_port} metrics={metrics_port}", flush=True)
    for name, label, trace in SCHEDULE:
        print("START", label, trace or "(default trace)", flush=True)
        if worker is None:
            logpath = ROOT / "runs" / f"{TAG}-workers-{label}.log"
            log = logpath.open("w")
            worker = subprocess.Popen([PYTHON, "-u", str(ROOT / "scripts" / "serve_workers.py")],
                                      env={**os.environ, "RUN_LABEL": f"{TAG}-workers"},
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                log.flush()
                if "ALL_WORKERS_READY" in logpath.read_text(errors="replace"):
                    break
                if worker.poll() is not None:
                    raise RuntimeError(f"worker exit {worker.returncode}; see {logpath}")
                time.sleep(3)
            else:
                raise TimeoutError(f"workers not ready; see {logpath}")
            print("WORKERS_READY", flush=True)
        env = dict(os.environ)
        if trace:
            env["TRACE_PATH"] = trace if trace.startswith("/") else str(ROOT / "data" / trace)
        arm_proc = subprocess.Popen([PYTHON, "-u", str(ROOT / "scripts" / "run_arm.py"), name, label],
                                    env=env, start_new_session=True)
        rc = arm_proc.wait()
        arm_proc = None
        if rc:
            raise RuntimeError(f"{label}: run failed {rc}")
        result = subprocess.run([PYTHON, str(ROOT / "scripts" / "summarize.py"), label],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{label}: invalid run: {result.stderr[-2000:]}")
        data = json.loads(result.stdout)
        key = str(env.get("TRACE_PATH", ""))
        if key not in expected:
            expected[key] = (data["trace_sha256"], data["plan_namespace"])
        # The plan namespace embeds per-run identifiers, so only the trace digest
        # identifies the workload across arms.
        if data["trace_sha256"] != expected[key][0]:
            raise RuntimeError(f"{label}: workload mismatch for {key}")
        (ROOT / "runs" / label / "metrics-summary.json").write_text(json.dumps(data, indent=2) + "\n")
        print(f"VERIFIED {label} requests={data['requests']} "
              f"tok_s={data['throughput_output_tokens_per_second']:.2f} "
              f"session_jct_mean={data['session_jct_seconds']['mean']:.2f}", flush=True)
    print("SERIES_COMPLETE", flush=True)
finally:
    stop(arm_proc)
    stop(worker)
    if log:
        log.close()

