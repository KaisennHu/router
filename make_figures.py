"""Generate the two PR figures from the verified formal-run CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'data' / 'repeats-metrics.csv').open()))
CAPS = ('4', '8', '16', '32')
ARMS = [('cache_aware', 'Cache', '#64748b', 'o'),
        ('kv_aware', 'KV', '#2563eb', 's'),
        ('kv_aware_progress', 'KV + Progress', '#9333ea', 'D'),
        ('kv_aware_aff', 'KV + Affinity', '#ea580c', '^'),
        ('kv_aware_aff_progress', 'KV + Affinity + Progress', '#059669', 'P')]


def record(cap, arm, rep):
    found = [r for r in rows if r['concurrency'] == cap and r['arm'] == arm and r['repeat'] == str(rep)]
    if len(found) != 1:
        raise ValueError((cap, arm, rep, len(found)))
    return found[0]


def f(row, key):
    return float(row[key])


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5,
    'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
x = np.arange(len(CAPS))

fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), layout='constrained')
fig.suptitle('Prefill progress: mean effect with decode affinity on', fontsize=21, fontweight='bold')
panels = [
    ('P queue: change in mean', 'prefill_queue_s_mean_ms', 'absolute', 'ms / request', 'Lower is better'),
    ('P prefill: change in mean', 'prefill_prefill_s_mean_ms', 'percent', '% vs progress off', 'Lower is better'),
    ('Client TTFT: change in mean', 'ttft_ms_mean', 'percent', '% vs progress off', 'Lower is better'),
    ('Output throughput: change', 'output_tok_s', 'percent', '% vs progress off', 'Higher is better'),
]
for ax, (title, key, scale, ylabel, direction) in zip(axes.flat, panels):
    vals = []
    for cap in CAPS:
        changes = []
        for rep in (1, 2):
            a = f(record(cap, 'kv_aware_aff', rep), key)
            b = f(record(cap, 'kv_aware_aff_progress', rep), key)
            changes.append((b - a) if scale == 'absolute' else (b / a - 1) * 100)
        vals.append(sum(changes) / len(changes))
    bars = ax.bar(x, vals, width=.46, color='#2563eb', zorder=3)
    ax.bar_label(bars, labels=[f'{v:+.1f}' for v in vals], padding=3, fontsize=11)
    ax.axhline(0, color='#334155', linewidth=1)
    ax.grid(axis='y', color='#e2e8f0', linewidth=.8, zorder=0)
    ax.set_title(title, loc='left')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, ['C4', 'C8', 'C16', 'C32'])
    ax.text(.99, .97, direction, transform=ax.transAxes, ha='right', va='top',
            fontsize=10.5, color='#475569')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * .12, hi + (hi - lo) * .17)
fig.text(.5, -.005, 'Each bar is the arithmetic mean of two paired-run changes; '
         '32 sessions / 730 requests per run.', ha='center', fontsize=11, color='#475569')
fig.savefig(OUT / 'progress-impact.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), layout='constrained')
fig.suptitle('Five routing configurations: end-to-end comparison\n'
             'KV: mean of two runs; Cache: one run', fontsize=19, fontweight='bold')
charts = [('TTFT mean', 'ttft_ms_mean', 'ms / request', 'Lower is better'),
          ('TPOT mean', 'tpot_ms_mean', 'ms / output token', 'Lower is better'),
          ('Task duration (JCT) mean', 'jct_task_duration_s_mean', 's / task', 'Lower is better'),
          ('Output throughput', 'output_tok_s', 'token / s', 'Higher is better')]
for ax, (title, key, ylabel, direction) in zip(axes.flat, charts):
    for arm, label, color, marker in ARMS:
        runs = (1,) if arm == 'cache_aware' else (1, 2)
        vals = [sum(f(record(cap, arm, rep), key) for rep in runs) / len(runs) for cap in CAPS]
        ax.plot(x, vals, marker=marker, markersize=7, linewidth=2.3,
                color=color, label=label, zorder=3)
    ax.set_title(title, loc='left')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, ['C4', 'C8', 'C16', 'C32'])
    ax.grid(axis='y', color='#e2e8f0', linewidth=.8, zorder=0)
    ax.text(.99, .97, direction, transform=ax.transAxes, ha='right', va='top',
            fontsize=10.5, color='#475569')
    ax.margins(x=.09, y=.15)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncols=5, frameon=False, fontsize=10.5)
fig.savefig(OUT / 'five-policy-end-to-end.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('generated', *(str(p) for p in sorted(OUT.glob('*.png'))))
