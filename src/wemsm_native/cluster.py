"""
cluster.py
==========

k-means discretisation of the pooled tICA space. Frames from all runs are
concatenated for fitting and the labels are split back per run.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def split_like(flat: np.ndarray, like: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Split a flat per-frame array into per-run (n_cycles, n_walkers, ...) arrays."""
    out, i = [], 0
    for a in like:
        nc, nw = a.shape[:2]
        out.append(flat[i:i + nc * nw].reshape(nc, nw, *flat.shape[1:]))
        i += nc * nw
    if i != len(flat):
        raise ValueError(f'split_like: {len(flat)} values for {i} frames')
    return out


def kmeans(tics_by_run: Sequence[np.ndarray], n_clusters: int,
           fixed_seed: bool = True, max_iter: int = 500):
    """Returns (centers (n_clusters, dim), labels_by_run [(n_cycles, n_walkers)])."""
    from deeptime.clustering import KMeans
    flat = np.concatenate([t.reshape(-1, t.shape[-1]) for t in tics_by_run])
    if len(flat) < n_clusters:
        raise ValueError(f'{len(flat)} frames cannot fill {n_clusters} clusters')
    print(f'  [cluster] k-means k={n_clusters} on {len(flat)} frames x '
          f'{flat.shape[1]} TICs')
    km = KMeans(n_clusters=n_clusters, fixed_seed=fixed_seed, max_iter=max_iter,
                n_jobs=None)
    labels = np.asarray(km.fit_transform(flat), dtype=np.int64)
    centers = np.asarray(km.fetch_model().cluster_centers)
    occupied = len(np.unique(labels))
    print(f'  [cluster] {occupied}/{n_clusters} clusters occupied')
    return centers, split_like(labels, tics_by_run)
