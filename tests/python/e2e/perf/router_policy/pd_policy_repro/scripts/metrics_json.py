"""Emit compact per-arm metrics as JSON (rounded) for downstream report generation."""
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/david_cwq/hhc/pr519-h20-20261123')
GROUPS = {
    'tracelab_2p2d': 'tlg-*',
    'tracelab_4p4d': 'tlf-*',
    'swebench_4p4d': 'swe-*',
    'agentx_2p2d': 'p2d-*',
    'agentx_4p4d_pr493': 'het3-*',
}
ARM_LABEL = {'rr': 'round_robin', 'cache': 'cache_aware', 'kv': 'kv_aware', 'rel': 'v0.1.17'}


def metric(path, name):
    if not Path(path).is_file():
        return None
    pat = re.compile(re.escape(name) + r'(?:\{[^}]*\})?\s+([0-9.eE+-]+)$')
    vals = [float(m.group(1)) for line in Path(path).read_text(errors='replace').splitlines()
            if (m := pat.match(line))]
    return sum(vals) if vals else None


def stats(vals):
    if not vals:
        return None
    s = sorted(vals)

    def q(p):
        k = (len(s) - 1) * p / 100.0
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {'mean': round(statistics.mean(s), 4), 'p50': round(q(50), 4),
            'p95': round(q(95), 4), 'p99': round(q(99), 4)}


out = {}
for group, pat in GROUPS.items():
    rounds = {}
    for d in sorted(ROOT.glob('runs/' + pat)):
        f = d / 'replay/requests.jsonl'
        if not f.exists() or not (d / 'after').is_dir():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        ok = [r for r in rows if r.get('status') == 'success']
        if not ok:
            continue
        span = (max(datetime.fromisoformat(r['finished_at']) for r in ok)
                - min(datetime.fromisoformat(r['started_at']) for r in ok)).total_seconds()
        tin = sum(r.get('input_tokens') or 0 for r in ok)
        tout = sum(r.get('output_tokens') or 0 for r in ok)
        ttft = [r['ttft_seconds'] for r in ok if r.get('ttft_seconds') is not None]
        tpot = []
        for r in ok:
            o = r.get('output_tokens') or 0
            if o > 1 and r.get('ttft_seconds') is not None and r.get('latency_seconds') is not None:
                tpot.append((r['latency_seconds'] - r['ttft_seconds']) / (o - 1) * 1000.0)
        lat = [r['latency_seconds'] for r in ok if r.get('latency_seconds') is not None]
        ses = {}
        for r in ok:
            s = r.get('session_id')
            a, b = datetime.fromisoformat(r['started_at']), datetime.fromisoformat(r['finished_at'])
            lo, hi = ses.get(s, (a, b))
            ses[s] = (min(lo, a), max(hi, b))
        jct = [(b - a).total_seconds() for a, b in ses.values()]

        ports = sorted(x.stem for x in (d / 'before').glob('*.prom'))
        new, hits, qrys = [], 0.0, 0.0
        for port in ports:
            pa = metric(d/'after'/f'{port}.prom', 'vllm:prompt_tokens_total')
            pb = metric(d/'before'/f'{port}.prom', 'vllm:prompt_tokens_total')
            ha = metric(d/'after'/f'{port}.prom', 'vllm:prefix_cache_hits_total')
            hb = metric(d/'before'/f'{port}.prom', 'vllm:prefix_cache_hits_total')
            qa = metric(d/'after'/f'{port}.prom', 'vllm:prefix_cache_queries_total')
            qb = metric(d/'before'/f'{port}.prom', 'vllm:prefix_cache_queries_total')
            if None not in (pa, pb, ha, hb):
                new.append(int((pa - pb) - (ha - hb)))
            if None not in (qa, qb, ha, hb):
                hits += ha - hb
                qrys += qa - qb

        round_id, arm_id = d.name.rsplit('-', 1)
        out.setdefault(group, {}).setdefault(round_id, {})[ARM_LABEL.get(arm_id, arm_id)] = {
            'requests': len(ok),
            'span_s': round(span, 1),
            'out_tps': round(tout / span, 1),
            'total_tps': round((tin + tout) / span, 1),
            'hit_pct': round(100 * hits / qrys, 2) if qrys else None,
            'new_prefill_tokens': int(sum(new)) if new else None,
            'skew': round(max(new) / statistics.mean(new), 2) if new and statistics.mean(new) else None,
            'ttft_s': stats(ttft),
            'tpot_ms': stats(tpot),
            'req_latency_s': stats(lat),
            'jct_s': stats(jct),
        }

print(json.dumps(out, separators=(',', ':'), ensure_ascii=False))
