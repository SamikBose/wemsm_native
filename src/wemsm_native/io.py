"""
io.py
=====

The only module that reads wepy HDF5 files.

wepy stores per-walker arrays as ``[walker][cycle]``; everything returned here
is ``(cycle, walker, ...)``. Each run is read on its own, so runs may differ in
their numbers of cycles and walkers.

Positions are in nm. Only orthorhombic boxes are supported (the minimum-image
convention used for distances assumes a rectangular cell).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, RunSpec


# --------------------------------------------------------------------------- #
def box_lengths(bv: np.ndarray, check_orthorhombic: bool = True) -> np.ndarray:
    """Box lengths (nm) from (3, 3) or (n_frames, 3, 3) box vectors.

    The norm is taken along the last axis so batched input is handled
    correctly (wepy's ``box_vectors_to_lengths_angles`` is single-frame only).
    """
    bv = np.asarray(bv, dtype=float)
    lengths = np.linalg.norm(bv, axis=-1)
    if check_orthorhombic:
        idx = np.arange(3)
        diag = np.zeros_like(bv)
        diag[..., idx, idx] = bv[..., idx, idx]
        off = np.abs(bv - diag).max()
        if off / max(np.abs(bv).max(), 1e-12) > 1e-4:
            raise ValueError(
                f'box vectors are not orthorhombic (largest off-diagonal {off:.4g} '
                f'nm); the minimum-image convention used here needs a '
                f'rectangular box. Re-image the trajectory upstream.')
    return lengths


def minimum_image(disp: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    """Minimum-image displacements; ``disp`` is (n_frames, ..., 3) and ``box``
    is (n_frames, 3), (3,) or None."""
    if box is None:
        return disp
    L = np.asarray(box, dtype=float)
    if L.ndim == 1:
        L = np.broadcast_to(L, (disp.shape[0], 3))
    L = L.reshape(disp.shape[0], *([1] * (disp.ndim - 2)), 3)
    return disp - L * np.round(disp / L)


def unwrap_atoms(xyz: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    """Make a compact molecule whole by minimum-imaging every atom against atom 0."""
    if box is None:
        return xyz
    ref = xyz[:, :1, :]
    return (ref + minimum_image(xyz - ref, box)).astype(xyz.dtype)


# --------------------------------------------------------------------------- #
def load_topology(cfg: Config):
    """mdtraj topology: ``cfg.topology_pdb`` or the JSON stored in the first HDF5."""
    import mdtraj as mdj
    if cfg.topology_pdb:
        if not os.path.isfile(cfg.topology_pdb):
            raise FileNotFoundError(f'topology PDB not found: {cfg.topology_pdb}')
        return mdj.load_pdb(cfg.topology_pdb).topology
    import h5py
    from wepy.util.mdtraj import json_to_mdtraj_topology
    first = cfg.run_specs()[0].h5_path
    with h5py.File(first, 'r') as f:
        if 'topology' not in f:
            raise KeyError(f'{first} stores no topology; set `topology_pdb` in the config')
        top = f['topology'][()]
    return json_to_mdtraj_topology(top.decode() if isinstance(top, bytes) else top)


def check_h5(spec: RunSpec) -> None:
    if not os.path.isfile(spec.h5_path):
        raise FileNotFoundError(f'[{spec.label}] wepy HDF5 not found: {spec.h5_path}')


def run_shape(spec: RunSpec) -> Tuple[int, int]:
    """(n_cycles used, n_walkers) for one run, honouring ``max_cycles``."""
    import h5py
    check_h5(spec)
    with h5py.File(spec.h5_path, 'r') as f:
        trajs = f[f'/runs/{spec.run_idx}/trajectories']
        walkers = sorted(int(k) for k in trajs.keys())
        if walkers != list(range(len(walkers))):
            raise ValueError(f'[{spec.label}] walker indices are not contiguous: '
                             f'{walkers[:5]}...')
        lengths = {trajs[str(w)]['positions'].shape[0] for w in walkers}
    if len(lengths) != 1:
        raise ValueError(f'[{spec.label}] walkers have different numbers of '
                         f'cycles {sorted(lengths)}; a constant walker count per '
                         f'cycle is required within a run')
    n_cycles = lengths.pop()
    if spec.max_cycles is not None:
        n_cycles = min(n_cycles, int(spec.max_cycles))
    return int(n_cycles), len(walkers)


def read_weights(spec: RunSpec) -> np.ndarray:
    """Walker weights, shape (n_cycles, n_walkers)."""
    import h5py
    nc, nw = run_shape(spec)
    out = np.zeros((nc, nw), dtype=float)
    with h5py.File(spec.h5_path, 'r') as f:
        base = f[f'/runs/{spec.run_idx}/trajectories']
        for w in range(nw):
            wt = np.asarray(base[str(w)]['weights'][:nc])
            out[:, w] = wt[:, 0] if wt.ndim == 2 else wt
    if out.sum() <= 0:
        raise ValueError(f'[{spec.label}] walker weights sum to {out.sum()}')
    return out


def read_frame(spec: RunSpec, cycle: int = 0, walker: int = 0):
    """(positions (n_atoms, 3), box_lengths (3,)) of one stored frame."""
    import h5py
    check_h5(spec)
    with h5py.File(spec.h5_path, 'r') as f:
        g = f[f'/runs/{spec.run_idx}/trajectories/{walker}']
        return (np.asarray(g['positions'][cycle]),
                box_lengths(np.asarray(g['box_vectors'][cycle])))


# --------------------------------------------------------------------------- #
@dataclass
class FrameComputer:
    """A per-frame quantity computed from a subset of atoms.

    ``fn(xyz, box)`` receives xyz of shape (n_frames, len(atoms), 3) in the
    order of ``atoms`` and box lengths (n_frames, 3), and returns an array of
    shape (n_frames,) or (n_frames, k).
    """
    name: str
    atoms: np.ndarray
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    unwrap: bool = False


def extract_run(spec: RunSpec, computers: Sequence[FrameComputer]
                ) -> Dict[str, np.ndarray]:
    """One pass over a run's positions evaluating several FrameComputers.

    Only the contiguous atom span covering every requested atom is read, which
    avoids loading solvent for protein-only quantities.

    Returns {computer.name: (n_cycles, n_walkers, ...) array}.
    """
    import h5py
    nc, nw = run_shape(spec)
    atoms_all = np.concatenate([np.asarray(c.atoms, dtype=int) for c in computers])
    lo, hi = int(atoms_all.min()), int(atoms_all.max())
    out: Dict[str, List[np.ndarray]] = {c.name: [None] * nw for c in computers}
    with h5py.File(spec.h5_path, 'r') as f:
        base = f[f'/runs/{spec.run_idx}/trajectories']
        for w in range(nw):
            g = base[str(w)]
            box = box_lengths(np.asarray(g['box_vectors'][:nc]))
            span = np.asarray(g['positions'][:nc, lo:hi + 1, :])
            for c in computers:
                xyz = np.ascontiguousarray(span[:, np.asarray(c.atoms) - lo, :])
                if c.unwrap:
                    xyz = unwrap_atoms(xyz, box)
                val = np.asarray(c.fn(xyz, box))
                if val.shape[0] != nc:
                    raise ValueError(f'{c.name}: returned {val.shape[0]} frames '
                                     f'for {nc} cycles')
                out[c.name][w] = val
    result = {name: np.stack(vals, axis=1) for name, vals in out.items()}
    print(f'  [{spec.label}] {nc} cycles x {nw} walkers: '
          f'{", ".join(f"{k} {v.shape[2:] or (1,)}" for k, v in result.items())}')
    return result
