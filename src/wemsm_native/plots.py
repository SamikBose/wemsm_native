"""
plots.py
========

Weighted 2-D free-energy surfaces, F = -kT ln P.

P is the weight-normalised 2-D histogram over all runs. Bins with P < pmin
are treated as empty and unsampled bins are NaN (drawn blank), so empty
regions stay distinct from sampled-but-unlikely ones. F is shifted to a
minimum of 0.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .msm import KB_KCAL


def free_energy_2d(x, y, weights=None, bins=100, ranges=None,
                   temperature: Optional[float] = None, pmin: float = 1e-20):
    """Returns (F (nx, ny), P (nx, ny), xedges, yedges)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    if w is not None:
        if w.shape != x.shape:
            raise ValueError(f'weights shape {w.shape} != data shape {x.shape}')
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError('weights must be non-negative with a positive sum')
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=ranges, weights=w)
    P = H / H.sum()
    P[P < pmin] = 0.0
    F = np.full_like(P, np.nan)
    occ = P > 0
    F[occ] = -np.log(P[occ]) * (KB_KCAL * temperature if temperature else 1.0)
    if occ.any():
        F[occ] -= F[occ].min()
    return F, P, xe, ye


def plot_fe(F, xe, ye, out_png: str, xlabel: str, ylabel: str, title: str,
            temperature: Optional[float], dpi: int = 300) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(F.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=(xe[0], xe[-1], ye[0], ye[-1]), interpolation='nearest')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Free energy [kcal/mol]' if temperature else r'$-\ln P$')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
