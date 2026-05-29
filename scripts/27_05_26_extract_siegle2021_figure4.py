"""Reproduce the data underlying Siegle et al. 2021, Figure 4 (e, f, h)
and ExtData Figure 9k-p.

Source CSVs (in ../data/figure4/) come straight from the published repo:
    https://github.com/AllenInstitute/neuropixels_platform_paper/tree/master/data

Outputs (in ../data/figure4/summaries/):
    fig4e_time_to_first_spike.csv       region, n, mean, ci_lo, ci_hi
    fig4f_change_modulation.csv         region, n, mean_active, ci_lo_a, ci_hi_a,
                                                  mean_passive, ci_lo_p, ci_hi_p
    fig4h_decoder_corr.csv              region, n, mean, sem
    ed9k_change_modulation_hit_miss.csv region, n, mean_hit, ci_lo_h, ci_hi_h,
                                                  mean_miss, ci_lo_m, ci_hi_m
    ed9lmn_response_rates.csv           region, rate_kind, state, n, mean, ci_lo, ci_hi
    ed9p_decoder_accuracy.csv           region, n, mean, sem
    fig4_reproduction.png               sanity-check plot of 4e/4f/4h
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'figure4'
OUT = DATA / 'summaries'
OUT.mkdir(parents=True, exist_ok=True)

# Hard-coded from Figure4.py (regions are listed in hierarchy order).
REGIONS = ['LGd', 'V1', 'LM', 'RL', 'LP', 'AL', 'PM', 'AM']
HIER_SCORE = np.array([
    -0.5150279628298357, -0.35733209934482374, -0.09388855125761343,
    -0.05987132463908328,  0.10524780962600731,  0.15221797920142832,
     0.32766807486511995,  0.440986074378801,
])
HIER_COLORS = np.array([
    [217, 141, 194], [129, 116, 177], [78, 115, 174], [101, 178, 201],
    [88, 167, 106], [202, 183, 120], [219, 132, 87], [194, 79, 84],
], dtype=float) / 255.0

RNG = np.random.default_rng(0)
N_BOOT = 5000


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Match the published code: mean of n_boot resamples, 2.5/97.5 percentile."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = np.array([
        np.nanmean(RNG.choice(values, len(values), replace=True))
        for _ in range(n_boot)
    ])
    return tuple(np.percentile(boots, (2.5, 97.5)))


def per_region(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        d = df.loc[df['Region'] == region, col].to_numpy(dtype=float)
        d = d[~np.isnan(d)]
        if len(d):
            lo, hi = bootstrap_ci(d)
        else:
            lo = hi = np.nan
        rows.append({'region': region, 'n': len(d),
                     'mean': np.nanmean(d) if len(d) else np.nan,
                     'ci_lo': lo, 'ci_hi': hi})
    return pd.DataFrame(rows)


def main() -> None:
    cm = pd.read_csv(DATA / 'change_modulation_data.csv')
    dec = pd.read_csv(DATA / 'decoding_data.csv')

    # ------- Figure 4e: Time to First Spike -------
    f4e = per_region(cm, 'Time To First Spike')
    f4e['hierarchy_score'] = HIER_SCORE
    f4e.to_csv(OUT / 'fig4e_time_to_first_spike.csv', index=False)

    # ------- Figure 4f: Change Modulation, Active vs Passive -------
    a = per_region(cm, 'Change Modulation Active').rename(columns={
        'mean': 'mean_active', 'ci_lo': 'ci_lo_a', 'ci_hi': 'ci_hi_a', 'n': 'n'})
    p = per_region(cm, 'Change Modulation Passive').rename(columns={
        'mean': 'mean_passive', 'ci_lo': 'ci_lo_p', 'ci_hi': 'ci_hi_p'})
    f4f = a.merge(p[['region', 'mean_passive', 'ci_lo_p', 'ci_hi_p']], on='region')
    f4f['hierarchy_score'] = HIER_SCORE
    f4f.to_csv(OUT / 'fig4f_change_modulation.csv', index=False)

    # ------- Figure 4h: Decoder correlation with behavior -------
    col = 'Correleation of decoder prediction and mouse behavior'
    rows = []
    for region in REGIONS:
        d = dec.loc[dec['Region'] == region, col].to_numpy(dtype=float)
        d = d[~np.isnan(d)]
        rows.append({'region': region, 'n': len(d),
                     'mean': np.nanmean(d) if len(d) else np.nan,
                     'sem': np.nanstd(d, ddof=1) / np.sqrt(len(d)) if len(d) else np.nan})
    f4h = pd.DataFrame(rows)
    f4h['hierarchy_score'] = HIER_SCORE
    f4h.to_csv(OUT / 'fig4h_decoder_corr.csv', index=False)

    # ------- ExtData 9k: Hit vs Miss change modulation -------
    h = per_region(cm, 'Change Modulation Hit').rename(columns={
        'mean': 'mean_hit', 'ci_lo': 'ci_lo_h', 'ci_hi': 'ci_hi_h'})
    m = per_region(cm, 'Change Modulation Miss').rename(columns={
        'mean': 'mean_miss', 'ci_lo': 'ci_lo_m', 'ci_hi': 'ci_hi_m'})
    ed9k = h.merge(m[['region', 'mean_miss', 'ci_lo_m', 'ci_hi_m']], on='region')
    ed9k['hierarchy_score'] = HIER_SCORE
    ed9k.to_csv(OUT / 'ed9k_change_modulation_hit_miss.csv', index=False)

    # ------- ExtData 9l-n: Pre-change/Change/Baseline rates, Active vs Passive
    rows = []
    for rate_kind in ('Pre-change Response', 'Change Response', 'Baseline Rate'):
        for state in ('Active', 'Passive'):
            sub = per_region(cm, f'{rate_kind} {state}')
            sub['rate_kind'] = rate_kind
            sub['state'] = state
            rows.append(sub)
    ed9lmn = pd.concat(rows, ignore_index=True)
    ed9lmn = ed9lmn[['region', 'rate_kind', 'state', 'n',
                     'mean', 'ci_lo', 'ci_hi']]
    ed9lmn.to_csv(OUT / 'ed9lmn_response_rates.csv', index=False)

    # ------- ExtData 9p: Decoder accuracy -------
    col = 'Decoder accuracy'
    rows = []
    for region in REGIONS:
        d = dec.loc[dec['Region'] == region, col].to_numpy(dtype=float)
        d = d[~np.isnan(d)]
        rows.append({'region': region, 'n': len(d),
                     'mean': np.nanmean(d) if len(d) else np.nan,
                     'sem': np.nanstd(d, ddof=1) / np.sqrt(len(d)) if len(d) else np.nan})
    ed9p = pd.DataFrame(rows)
    ed9p['hierarchy_score'] = HIER_SCORE
    ed9p.to_csv(OUT / 'ed9p_decoder_accuracy.csv', index=False)

    # ------- Verification plot mirroring Figure4.py -------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 4e
    ax = axes[0]
    for h_, m_, lo, hi, c in zip(HIER_SCORE, f4e['mean'], f4e['ci_lo'], f4e['ci_hi'], HIER_COLORS):
        ax.plot(h_, m_, 'o', mec=c, mfc=c, ms=6)
        ax.plot([h_, h_], [lo, hi], color=c)
    slope, yint, *_ = scipy.stats.linregress(HIER_SCORE, f4e['mean'])
    x = np.array([HIER_SCORE.min(), HIER_SCORE.max()])
    ax.plot(x, slope * x + yint, '--k')
    r, p_ = scipy.stats.pearsonr(HIER_SCORE, f4e['mean'])
    ax.set_title(f'Fig 4e: TTFS\nPearson r={r:.2f}, p={p_:.3f}')
    ax.set_xlabel('Hierarchy score'); ax.set_ylabel('Time to first spike (ms)')

    # 4f
    ax = axes[1]
    for state, fill, fitclr, means, los, his in [
        ('Active', True, 'k', f4f['mean_active'], f4f['ci_lo_a'], f4f['ci_hi_a']),
        ('Passive', False, '0.5', f4f['mean_passive'], f4f['ci_lo_p'], f4f['ci_hi_p']),
    ]:
        for i, (h_, m_, lo, hi, c) in enumerate(zip(HIER_SCORE, means, los, his, HIER_COLORS)):
            mfc = c if fill else 'none'
            ax.plot(h_, m_, 'o', mec=c, mfc=mfc, ms=6,
                    label=state if i == 0 else None)
            ax.plot([h_, h_], [lo, hi], color=c)
        slope, yint, *_ = scipy.stats.linregress(HIER_SCORE, means)
        ax.plot(x, slope * x + yint, '--', color=fitclr)
    ax.set_title('Fig 4f: Change modulation'); ax.legend()
    ax.set_xlabel('Hierarchy score'); ax.set_ylabel('Change modulation index')

    # 4h
    ax = axes[2]
    for h_, m_, s_, c in zip(HIER_SCORE, f4h['mean'], f4h['sem'], HIER_COLORS):
        ax.plot(h_, m_, 'o', mec=c, mfc=c, ms=6)
        ax.plot([h_, h_], [m_ - s_, m_ + s_], color=c)
    slope, yint, *_ = scipy.stats.linregress(HIER_SCORE, f4h['mean'])
    ax.plot(x, slope * x + yint, '--', color='0.5')
    r, p_ = scipy.stats.pearsonr(HIER_SCORE, f4h['mean'])
    ax.set_title(f'Fig 4h: Decoder vs behavior\nPearson r={r:.2f}, p={p_:.3f}')
    ax.set_xlabel('Hierarchy score'); ax.set_ylabel('Decoder/behavior corr.')

    for a in axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    plt.tight_layout()
    out_png = OUT / 'fig4_reproduction.png'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f'wrote {out_png}')

    print('\nFig 4e (region, n, mean ms):')
    print(f4e[['region', 'n', 'mean']].to_string(index=False))


if __name__ == '__main__':
    main()
