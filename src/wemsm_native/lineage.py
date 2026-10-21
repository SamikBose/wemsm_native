"""
lineage.py
==========

Lineage-following sliding windows for one WE run.

In REVO, walkers are cloned and merged between cycles, so walker w at cycle
c + lag is generally NOT a descendant of walker w at cycle c. wepy's
``sliding_window`` over the run's net parent table returns traces that follow
the true parent lineage. These windows are used both for tICA lagged pairs and
for MSM transition counts.

Windows are always built from ONE run's own resampling records and are only
ever used to index that same run's arrays: a window can never join frames from
two different runs. ``check_windows`` enforces the structure explicitly.

Each run's windows are returned as an int array of shape
(n_windows, lag + 1, 2) whose last axis is (walker, cycle).
"""

from __future__ import annotations

import numpy as np

from .config import RunSpec
from .io import check_h5


class LineageError(RuntimeError):
    """Resampling records could not be turned into lineage windows."""


def _net_parent_table(spec: RunSpec):
    from wepy.analysis.parents import (net_parent_table, parent_panel,
                                       resampling_panel)
    from wepy.hdf5 import WepyHDF5
    from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
    check_h5(spec)
    wh = WepyHDF5(spec.h5_path, mode='r')
    wh.open()
    try:
        records = wh.resampling_records([spec.run_idx])
    finally:
        wh.close()
    if not records:
        raise LineageError(f'[{spec.label}] run {spec.run_idx} has no resampling records')
    return net_parent_table(parent_panel(MultiCloneMergeDecision,
                                         resampling_panel(records)))


def lineage_windows(spec: RunSpec, lag: int, n_cycles: int, n_walkers: int
                    ) -> np.ndarray:
    """wepy sliding windows of length lag + 1 within cycles [0, n_cycles)."""
    from wepy.analysis.parents import sliding_window
    if lag < 1:
        raise ValueError(f'lag must be >= 1; got {lag}')
    table = [list(row) for row in _net_parent_table(spec)[:n_cycles]]
    if len(table) == n_cycles - 1:
        # No resampling after the last stored cycle. Row c only supplies the
        # parents of cycle c + 1, so the missing last row is never read; it is
        # padded because sliding_window takes its final cycle from the table
        # length and would otherwise drop every window ending at that cycle.
        table.append(list(range(n_walkers)))
    if len(table) < n_cycles:
        raise LineageError(
            f'[{spec.label}] resampling records cover {len(table)} cycles but '
            f'{n_cycles} cycles are analysed')
    wins = sliding_window(table, lag + 1)
    return (np.asarray(wins, dtype=np.int64).reshape(-1, lag + 1, 2)
            if len(wins) else np.zeros((0, lag + 1, 2), dtype=np.int64))


def consecutive_windows(lag: int, n_cycles: int, n_walkers: int) -> np.ndarray:
    """Per-walker windows (w, c) -> (w, c + lag), ignoring cloning/merging."""
    c = np.arange(n_cycles - lag)
    w = np.arange(n_walkers)
    cc, ww = np.meshgrid(c, w, indexing='ij')
    steps = np.arange(lag + 1)
    out = np.empty((cc.size, lag + 1, 2), dtype=np.int64)
    out[:, :, 0] = ww.reshape(-1, 1)
    out[:, :, 1] = cc.reshape(-1, 1) + steps
    return out


def check_windows(windows: np.ndarray, lag: int, n_cycles: int, n_walkers: int,
                  label: str) -> None:
    """Every window spans exactly cycles c, c+1, ..., c+lag inside this run."""
    if windows.ndim != 3 or windows.shape[1:] != (lag + 1, 2):
        raise LineageError(f'[{label}] windows have shape {windows.shape}; '
                           f'expected (n, {lag + 1}, 2)')
    if len(windows) == 0:
        return
    w, c = windows[..., 0], windows[..., 1]
    if not np.all(np.diff(c, axis=1) == 1):
        raise LineageError(f'[{label}] a lineage window does not span consecutive cycles')
    if c.min() < 0 or c.max() >= n_cycles or w.min() < 0 or w.max() >= n_walkers:
        raise LineageError(
            f'[{label}] lineage window outside this run '
            f'({n_cycles} cycles x {n_walkers} walkers): cycles '
            f'{c.min()}..{c.max()}, walkers {w.min()}..{w.max()}')


def run_windows(spec: RunSpec, lag: int, n_cycles: int, n_walkers: int,
                allow_fallback: bool) -> np.ndarray:
    """Lineage windows for one run; error (or opt-in fallback) if unavailable."""
    try:
        wins = lineage_windows(spec, lag, n_cycles, n_walkers)
    except Exception as exc:                                      # noqa: BLE001
        if not allow_fallback:
            raise LineageError(
                f'[{spec.label}] could not build lineage windows from the '
                f'resampling records of {spec.h5_path} (run {spec.run_idx}): '
                f'{type(exc).__name__}: {exc}\n'
                f'Check that the wepy version can read this file. Only if this '
                f'run had NO cloning/merging, set '
                f'`lineage.allow_contiguous_fallback=True` to use per-walker '
                f'consecutive windows instead.') from exc
        print(f'  [{spec.label}] WARNING: lineage windows unavailable ({exc}); '
              f'using per-walker consecutive windows as requested by '
              f'lineage.allow_contiguous_fallback. These ignore cloning/merging.')
        wins = consecutive_windows(lag, n_cycles, n_walkers)
    # outside the try: a malformed window is a bug, never a reason to fall back
    check_windows(wins, lag, n_cycles, n_walkers, spec.label)
    print(f'  [{spec.label}] {len(wins)} lineage windows (lag {lag})')
    return wins
