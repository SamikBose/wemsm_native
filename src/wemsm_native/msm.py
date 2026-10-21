"""
msm.py
======

Weighted transition counts, transition matrix and CSN stationary weights.

Conventions (identical to the reference msm.py)
-----------------------------------------------
* ``counts[to, from]``: the destination is the row, so the transition matrix
  is COLUMN-stochastic, which is what CSNAnalysis expects.
* For a lineage window (w1, c1) -> (w2, c2) within one run:
    'w2'   : counts[x2, x1] += weight(w2, c2)
    'w2w1' : counts[x2, x1] += weight(w2, c2) / weight(w1, c1)
* Stationary weights come from CSNAnalysis: trim to the main connected
  component, then ``calc_mult_weights``. There is no fallback: any failure
  raises ``MSMError`` explaining what went wrong.

Frame-weight mapping
--------------------
'cluster'    : frame weight = MSM weight of its state (msm_weights[labels]),
               exactly as the reference msm.py.
'normalized' : MSM weight of its state / number of frames in the state, so
               the per-frame weights of a state sum to the state's weight.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KB_KCAL = 0.0019872041          # kcal/mol/K


class MSMError(RuntimeError):
    """The stationary MSM weights could not be computed."""


# --------------------------------------------------------------------------- #
def ensemble_cluster_weights(labels_by_run, weights_by_run, n_clusters: int) -> np.ndarray:
    """Summed raw walker weights per state over all runs, normalised to 1."""
    lab = np.concatenate([np.asarray(a).ravel() for a in labels_by_run])
    w = np.concatenate([np.asarray(a, dtype=float).ravel() for a in weights_by_run])
    cw = np.bincount(lab, weights=w, minlength=n_clusters)
    return cw / cw.sum() if cw.sum() > 0 else cw


def build_counts(labels_by_run: Sequence[np.ndarray],
                 weights_by_run: Sequence[np.ndarray],
                 windows_by_run: Sequence[np.ndarray],
                 n_clusters: int, scheme: str = 'w2') -> np.ndarray:
    """Weighted counts[to, from] from per-run lineage windows.

    Each run's windows index only that run's labels and weights; the per-run
    count matrices are then summed.
    """
    if scheme not in ('w2', 'w2w1'):
        raise ValueError(f"unknown counts scheme {scheme!r}; use 'w2' or 'w2w1'")
    if not (len(labels_by_run) == len(weights_by_run) == len(windows_by_run)):
        raise ValueError(f'runs mismatch: {len(labels_by_run)} label arrays, '
                         f'{len(weights_by_run)} weight arrays, '
                         f'{len(windows_by_run)} window sets')
    counts = np.zeros((n_clusters, n_clusters), dtype=float)
    n_used = n_skipped = 0
    for r, (lab, wt, wins) in enumerate(zip(labels_by_run, weights_by_run,
                                            windows_by_run)):
        if lab.shape != wt.shape:
            raise ValueError(f'run {r}: labels {lab.shape} vs weights {wt.shape}')
        if len(wins) == 0:
            continue
        w1, c1 = wins[:, 0, 0], wins[:, 0, 1]
        w2, c2 = wins[:, -1, 0], wins[:, -1, 1]
        if max(c1.max(), c2.max()) >= lab.shape[0] or max(w1.max(), w2.max()) >= lab.shape[1]:
            raise IndexError(f'run {r}: lineage window outside its arrays {lab.shape}')
        x1, x2 = lab[c1, w1], lab[c2, w2]
        wt1, wt2 = wt[c1, w1], wt[c2, w2]
        if scheme == 'w2':
            val = wt2
        else:
            ok = wt1 > 0                       # the wt2/wt1 ratio needs wt1 > 0
            n_skipped += int((~ok).sum())
            x1, x2, val = x1[ok], x2[ok], wt2[ok] / wt1[ok]
        np.add.at(counts, (x2, x1), val)
        n_used += len(val)
    print(f'  [msm] {n_used} lineage windows counted ({scheme})'
          + (f', {n_skipped} skipped with zero source weight' if n_skipped else '')
          + f'; {int(np.sum(counts > 0))} non-zero entries')
    return counts


def transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Column-stochastic T from counts[to, from]."""
    col = counts.sum(axis=0, keepdims=True)
    return np.divide(counts, col, out=np.zeros_like(counts), where=col > 0)


# --------------------------------------------------------------------------- #
def make_csn(counts: np.ndarray):
    """CSN object for ``counts``, trimmed to the main connected component."""
    try:
        from csnanalysis.csn import CSN
    except ImportError as exc:
        raise MSMError(
            'CSNAnalysis is required for the MSM weights and is not installed in '
            'this Python environment. Install it with:\n'
            '    pip install "git+https://github.com/ADicksonLab/CSNAnalysis.git"'
        ) from exc
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise MSMError('count matrix contains negative or non-finite entries')
    if counts.sum() <= 0:
        raise MSMError('count matrix is empty: no lineage transitions were counted. '
                       'Check that the runs have more cycles than msm.lag and that '
                       'the resampling records were read.')
    csn = CSN(counts)
    try:
        csn.trim()
    except AssertionError as exc:
        raise MSMError(
            f'CSN trimming failed ({exc}). A state in the main connected '
            f'component has no outgoing transitions; this usually means too '
            f'many clusters for the amount of data. Reduce cluster.n_clusters '
            f'or add data.') from exc
    except Exception as exc:                                      # noqa: BLE001
        raise MSMError(f'CSN trimming failed: {type(exc).__name__}: {exc}') from exc
    return csn


def stationary_weights(counts: np.ndarray, tol: float = 1e-10):
    """CSN multiplication weights on the trimmed network.

    Returns (weights (n_clusters,), csn). States outside the main component
    get weight 0.
    """
    csn = make_csn(counts)
    try:
        w = np.asarray(csn.calc_mult_weights(label='msm_weight', tol=tol),
                       dtype=float).reshape(-1)
    except Exception as exc:                                      # noqa: BLE001
        raise MSMError(f'CSNAnalysis calc_mult_weights failed: '
                       f'{type(exc).__name__}: {exc}') from exc
    if w.shape != (counts.shape[0],):
        raise MSMError(f'CSN returned {w.shape} weights for {counts.shape[0]} states')
    if not np.all(np.isfinite(w)):
        raise MSMError('CSN weights contain NaN/inf; the trimmed transition matrix '
                       'is probably ill-conditioned (try fewer clusters)')
    if np.any(w < -1e-12):
        raise MSMError(f'CSN weights contain negative values (min {w.min():.3e})')
    total = w.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise MSMError(f'CSN weights sum to {total:.8f}, not 1; calc_mult_weights '
                       f'did not converge (tol={tol})')
    n_trim = len(csn.trim_indices)
    print(f'  [msm] main component {n_trim}/{counts.shape[0]} states; '
          f'max weight {w.max():.3e}')
    return np.clip(w, 0.0, None), csn


def free_energy(w: np.ndarray, temperature: Optional[float]) -> np.ndarray:
    """-kT ln w (kcal/mol, or -ln w if temperature is None), min shifted to 0.

    Zero-weight states get 100, as in the reference msm.py.
    """
    fe = np.full(w.shape, 100.0)
    pos = w > 0
    if pos.any():
        fe[pos] = -np.log(w[pos]) * (KB_KCAL * temperature if temperature else 1.0)
        fe[pos] -= fe[pos].min()
    return fe


# --------------------------------------------------------------------------- #
def frame_weights(labels_by_run: Sequence[np.ndarray], state_weights: np.ndarray,
                  mapping: str = 'cluster') -> List[np.ndarray]:
    """Per-run (n_cycles, n_walkers) frame weights from state weights."""
    sw = np.asarray(state_weights, dtype=float).reshape(-1)
    if mapping == 'cluster':
        per = sw
    elif mapping == 'normalized':
        occ = np.bincount(np.concatenate([np.asarray(l).ravel() for l in labels_by_run]),
                          minlength=sw.size).astype(float)
        per = np.divide(sw, occ, out=np.zeros_like(sw), where=occ > 0)
    else:
        raise ValueError(f"mapping must be 'cluster' or 'normalized'; got {mapping!r}")
    return [per[np.asarray(l)] for l in labels_by_run]
