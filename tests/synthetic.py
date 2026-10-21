"""Synthetic wepy-like HDF5 runs for tests.

Each file has /runs/0/trajectories/{w}/{positions,box_vectors,weights} plus a
``/test_parent_table`` dataset holding a clone/merge parent table, which the
tests patch into ``lineage._net_parent_table`` in place of wepy's resampling
record parsing.
"""

import os

import numpy as np


def make_topology_pdb(path, n_res=12):
    import mdtraj as mdj
    top = mdj.Topology()
    ch = top.add_chain()
    for i in range(n_res):
        r = top.add_residue('ALA', ch, resSeq=i + 1)
        top.add_atom('CA', mdj.element.carbon, r)
    xyz = np.zeros((1, n_res, 3), dtype=np.float32)
    xyz[0, :, 0] = 0.38 * np.arange(n_res)                       # extended chain
    xyz[0, :, 1] = 0.3 * np.sin(np.arange(n_res))
    t = mdj.Trajectory(xyz, top, unitcell_lengths=np.array([[5.0, 5.0, 5.0]]),
                       unitcell_angles=np.array([[90.0, 90.0, 90.0]]))
    t.save_pdb(path)
    return t


def random_parent_table(rng, n_cycles, n_walkers):
    """Parent table with clones/merges: row c maps walker at c+1 -> parent at c."""
    table = []
    for _ in range(n_cycles):
        parents = np.arange(n_walkers)
        # clone a few walkers over others (merge victims)
        for _ in range(max(1, n_walkers // 4)):
            src, dst = rng.choice(n_walkers, 2, replace=False)
            parents[dst] = parents[src]
        table.append(parents.tolist())
    return table


def make_run(path, ref_xyz, n_cycles, n_walkers, seed):
    import h5py
    rng = np.random.default_rng(seed)
    n_atoms = ref_xyz.shape[0]
    table = random_parent_table(rng, n_cycles, n_walkers)
    pos = np.zeros((n_cycles, n_walkers, n_atoms, 3), dtype=np.float32)
    wts = np.zeros((n_cycles, n_walkers))
    cur = np.repeat(ref_xyz[None], n_walkers, axis=0).astype(np.float64)
    w = np.full(n_walkers, 1.0 / n_walkers)
    for c in range(n_cycles):
        if c > 0:
            par = np.asarray(table[c - 1])
            cur = cur[par].copy()
            # weights follow parents, then renormalise (clone/merge bookkeeping)
            w = w[par] * rng.uniform(0.5, 1.5, n_walkers)
            w /= w.sum()
        hinge = rng.normal(0, 0.05, n_walkers)
        cur[:, n_atoms // 2:, 1] += hinge[:, None]              # slow hinge mode
        cur += rng.normal(0, 0.01, cur.shape)
        pos[c] = cur
        wts[c] = w
    box = np.diag([5.0, 5.0, 5.0])
    with h5py.File(path, 'w') as f:
        for wk in range(n_walkers):
            g = f.create_group(f'runs/0/trajectories/{wk}')
            g['positions'] = pos[:, wk]
            g['box_vectors'] = np.repeat(box[None], n_cycles, axis=0)
            g['weights'] = wts[:, wk][:, None]
        f['test_parent_table'] = np.asarray(table, dtype=np.int64)
    return pos, wts, table


def patch_lineage():
    """Make lineage read the synthetic parent table instead of wepy records.

    Returns the original function so the caller can restore it.
    """
    import h5py
    from wemsm_native import lineage
    original = lineage._net_parent_table

    def _table(spec):
        with h5py.File(spec.h5_path, 'r') as f:
            return np.asarray(f['test_parent_table']).tolist()
    lineage._net_parent_table = _table
    return original
