#!/bin/bash
#SBATCH --job-name=main960-4p4d
#SBATCH --partition=h20
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/home/david_cwq/hhc/pr519-h20-20261123/logs/main960-run-%j.log
set -u
root=/home/david_cwq/hhc/pr519-h20-20261123
export PATH=/home/david_cwq/hhc/kv-unified-vllm033-runtime/bin:/usr/local/cuda/bin:/home/david_cwq/.local/bin:$PATH
export LD_LIBRARY_PATH=/home/david_cwq/hhc/e2e-9f35426-2026-10-13/envs/openssl3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/home/david_cwq/.cache/codex-hybrid-validation/nixl152
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache/xdg" HF_HOME="$root/hf"
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-163840}
export KV_BYTES=${KV_BYTES:-27917287424}
export PREFILL=${PREFILL:-4}
export WORKERS=${WORKERS:-8}
export TRACE_PATH="${TRACE_PATH-$root/data/agentx-hetero12x12.jsonl}"
export PORT_SEARCH_START="${PORT_SEARCH_START:-$((50000 + (${SLURM_JOB_ID:-0} % 30) * 400))}"
mkdir -p "$root"/{logs,runs,tmp,cache/xdg,hf,xdg,vllm,triton,cuda,torchinductor,flashinfer}
echo "node=$(hostname) $(date -Is) PREFILL=$PREFILL WORKERS=$WORKERS"
echo "trace sha=$(sha256sum "$TRACE_PATH" | cut -c1-16)"
B=/home/david_cwq/hhc/bin/router-main-9605956f
[ -x "$B" ] || { echo "MISSING BINARY $B"; exit 1; }
"$B" --help > /dev/null 2>&1 || { echo "BINARY NOT RUNNABLE $B"; exit 1; }
echo "binary ok $(sha256sum "$B" | cut -c1-16)"

# A previous job of ours can still be tearing engines down. Engine cmdlines are
# `.../bin/vllm serve ...`, which a workdir-path match misses, so match the
# interpreter path and then wait until every port in the block is actually free.
pkill -u "$USER" -f "kv-unified-vllm033-runtime/bin/vllm serve" 2>/dev/null || true
pkill -u "$USER" -f "scripts/serve_workers.py" 2>/dev/null || true
sleep 3
pkill -9 -u "$USER" -f "kv-unified-vllm033-runtime/bin/vllm serve" 2>/dev/null || true
free_ports() {
  local p o
  for p in $(seq 50000 20 50400); do
    for o in 0 1 2 3; do
      if ss -ltn 2>/dev/null | grep -q ":$((p + o)) "; then return 1; fi
    done
  done
  if ss -ltn 2>/dev/null | grep -qE ":(50170|50171) "; then return 1; fi
  return 0
}
for i in $(seq 1 90); do
  if free_ports; then echo "ports free after $((i * 2))s"; break; fi
  if [ "$i" -eq 90 ]; then echo "PORTS_STILL_BUSY"; ss -ltn | grep -E ":(500|501)" | head -10; exit 1; fi
  sleep 2
done
# A failed earlier attempt leaves its output directory behind and run_arm.py
# refuses to reuse one, so clear this tag's directories before starting.
rm -rf "$root"/runs/"${TAG:-main960b}"-x1-* "$root"/runs/"${TAG:-main960b}"-x2-* "$root"/runs/"${TAG:-main960b}"-x3-*
exec /home/david_cwq/hhc/kv-unified-vllm033-runtime/bin/python -u "$root/scripts/series3.py"
