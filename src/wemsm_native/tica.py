"""
tica.py
=======

tICA trained on the weighted-ensemble data itself.

Time-lagged pairs (X_t, X_t+lag) are the endpoints of lineage windows, built
per run and only then pooled: both endpoints of every pair always come from
the same run. The default lag is 1 WE cycle (two-snapshot window, one
resampling step).

deeptime's TICA estimates the covariances from unweighted samples, so the
walker weights do not enter the tICA fit. With lag > 1 each pair spans several
resampling steps of WE datapoints and a warning is issued.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np


class WETICALagWarning(UserWarning):
    """tICA lag > 1 on unweighted WE datapoints."""


def lag_warning(lag: int) -> None:
    if lag > 1:
        warnings.warn(
            f'tICA lag = {lag} WE cycles (sliding window of {lag + 1} snapshots). '
            f'These are weighted-ensemble datapoints joined through {lag} '
            f'resampling steps, and tICA training is UNWEIGHTED (walker weights '
            f'are not used in the covariance estimates). The recommended '
            f'setting is tica.lag = 1.', WETICALagWarning, stacklevel=2)


def lagged_pairs(feats_by_run: Sequence[np.ndarray],
                 windows_by_run: Sequence[np.ndarray]):
    """Pool within-run lagged pairs.

    feats_by_run[r] : (n_cycles_r, n_walkers_r, n_features)
    windows_by_run[r] : (n_windows_r, lag + 1, 2) of (walker, cycle) for run r
    """
    if len(feats_by_run) != len(windows_by_run):
        raise ValueError(f'features for {len(feats_by_run)} runs but windows '
                         f'for {len(windows_by_run)} runs')
    X0, X1 = [], []
    for r, (feats, wins) in enumerate(zip(feats_by_run, windows_by_run)):
        if feats.ndim != 3:
            raise ValueError(f'run {r}: features must be (cycle, walker, feature); '
                             f'got {feats.shape}')
        if len(wins) == 0:
            continue
        w1, c1 = wins[:, 0, 0], wins[:, 0, 1]
        w2, c2 = wins[:, -1, 0], wins[:, -1, 1]
        X0.append(feats[c1, w1])
        X1.append(feats[c2, w2])
    if not X0:
        raise RuntimeError('no lagged pairs: every run is shorter than the tICA lag')
    return np.concatenate(X0).astype(float), np.concatenate(X1).astype(float)


def train(feats_by_run, windows_by_run, lag: int, dim: int, scaling='kinetic_map'):
    """Fit deeptime TICA on lineage lagged pairs; returns the fitted model."""
    from deeptime.decomposition import TICA
    from deeptime.util.data import TimeLaggedDataset
    lag_warning(lag)
    X0, X1 = lagged_pairs(feats_by_run, windows_by_run)
    print(f'  [tica] fitting dim={dim} lag={lag} on {X0.shape[0]} within-run '
          f'lineage pairs x {X0.shape[1]} features (unweighted)')
    return TICA(lagtime=lag, dim=dim, scaling=scaling).fit(
        TimeLaggedDataset(X0, X1)).fetch_model()


def project(model, feats_by_run: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Per-run projections, each (n_cycles, n_walkers, dim)."""
    out = []
    for f in feats_by_run:
        nc, nw, nf = f.shape
        out.append(np.asarray(model.transform(f.reshape(nc * nw, nf).astype(float)))
                   .reshape(nc, nw, -1))
    return out


def timescales(model, lag: int, n: int = 5):
    try:
        return np.asarray(model.timescales(lagtime=lag))[:n]
    except Exception:                                             # noqa: BLE001
        return None
