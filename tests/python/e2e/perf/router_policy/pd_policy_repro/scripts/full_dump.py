"""Dump every arm of every H20 experiment as markdown tables with full percentiles."""
import json
import re
import statistics
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path('/home/david_cwq/hhc/pr519-h20-20261123')
GROUPS = [
    ('TraceLab 2P2D (6 sessions)', 'tlg-*'),
    ('TraceLab 4P4D (12 sessions)', 'tlf-*'),
    ('SWE-bench Pro 4P4D (12 tasks)', 'swe-*'),
    ('agentX 2P2D (6 sessions x 12 turns)', 'p2d-*'),
    ('agentX 4P4D (12 sessions x 12 turns)', 'het3-*'),
]


def metric(path, name):
    if not Path(path).is_file():
        return None
    pat = re.compile(re.escape(name) + r'(?:\{[^}]*\})?\s+([0-9.eE+-]+)$')
    vals = [float(m.group(1)) for line in Path(path).read_text(errors='replace').splitlines()
            if (m := pat.match(line))]
    return sum(vals) if vals else None


def q(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def parse(ts):
    return datetime.fromisoformat(ts)


def arm_stats(d):
    f = d / 'replay/requests.jsonl'
    if not f.exists() or not (d / 'after').is_dir():
        return None
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    ok = [r for r in rows if r.get('status') == 'success']
    if not ok:
        return None
    span = (max(parse(r['finished_at']) for r in ok) - min(parse(r['started_at']) for r in ok)).total_seconds()
    tin = sum(r.get('input_tokens') or 0 for r in ok)
    tout = sum(r.get('output_tokens') or 0 for r in ok)
    ttft = [r['ttft_seconds'] for r in ok if r.get('ttft_seconds') is not None]
    tpot = []
    for r in ok:
        o = r.get('output_tokens') or 0
        if o > 1 and r.get('ttft_seconds') is not None and r.get('latency_seconds') is not None:
            tpot.append((r['latency_seconds'] - r['ttft_seconds']) / (o - 1) * 1000.0)
    ses = {}
    for r in ok:
        s = r.get('session_id')
        a, b = parse(r['started_at']), parse(r['finished_at'])
        lo, hi = ses.get(s, (a, b))
        ses[s] = (min(lo, a), max(hi, b))
    jct = [(b - a).total_seconds() for a, b in ses.values()]
    ports = sorted(x.stem for x in (d / 'before').glob('*.prom'))
    new = []
    hits = qrys = 0.0
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
    return dict(
        tag=d.name, n=len(ok), span=span, out=tout / span, tot=(tin + tout) / span,
        hit=100 * hits / qrys if qrys else float('nan'),
        newp=sum(new) if new else None,
        skew=(max(new) / statistics.mean(new)) if new and statistics.mean(new) else float('nan'),
        ttft=ttft, tpot=tpot, jct=jct,
    )


def table(rows):
    out = ['| 臂 | 请求 | span (s) | 输出 tok/s | 总 tok/s | 命中率 | 新算 Prefill | 倾斜 max/avg |',
           '|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['tag'].rsplit('-', 1)[-1]} | {r['n']} | {r['span']:.0f} | {r['out']:.1f} | {r['tot']:.0f} | "
                   f"{r['hit']:.2f}% | {r['newp']:,} | {r['skew']:.2f} |")
    return '\n'.join(out)


def lat_table(rows):
    out = ['| 臂 | TTFT mean / p50 / p95 / p99 (s) | TPOT mean / p50 / p95 / p99 (ms) | JCT mean / p50 / p95 / p99 (s) |',
           '|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['tag'].rsplit('-', 1)[-1]} | "
                   f"{statistics.mean(r['ttft']):.2f} / {q(r['ttft'],50):.2f} / {q(r['ttft'],95):.2f} / {q(r['ttft'],99):.2f} | "
                   f"{statistics.mean(r['tpot']):.1f} / {q(r['tpot'],50):.1f} / {q(r['tpot'],95):.1f} / {q(r['tpot'],99):.1f} | "
                   f"{statistics.mean(r['jct']):.1f} / {q(r['jct'],50):.1f} / {q(r['jct'],95):.1f} / {q(r['jct'],99):.1f} |")
    return '\n'.join(out)


for title, pat in GROUPS:
    print(f"\n\n### {title}\n")
    rounds = {}
    for d in sorted(ROOT.glob('runs/' + pat)):
        st = arm_stats(d)
        if st:
            rounds.setdefault(d.name.rsplit('-', 1)[0], []).append(st)
    for rnd in sorted(rounds):
        rows = rounds[rnd]
        print(f"\n**{rnd}**\n")
        print(table(rows))
        print()
        print(lat_table(rows))
        if len(rows) >= 2:
            base = rows[0]
            print('\n相对基线：')
            for r in rows[1:]:
                print(f"- {r['tag'].rsplit('-',1)[-1]}: 吞吐 {r['out']/base['out']*100-100:+.1f}%, "
                      f"命中率 {r['hit']-base['hit']:+.2f} pt, 新算 Prefill {r['newp']/base['newp']*100-100:+.1f}%, "
                      f"TTFT mean {statistics.mean(r['ttft'])/statistics.mean(base['ttft'])*100-100:+.1f}%, "
                      f"TPOT mean {statistics.mean(r['tpot'])/statistics.mean(base['tpot'])*100-100:+.1f}%, "
                      f"JCT mean {statistics.mean(r['jct'])/statistics.mean(base['jct'])*100-100:+.1f}%, "
                      f"JCT p95 {q(r['jct'],95)/q(base['jct'],95)*100-100:+.1f}%")

