"""
observables.py
==============

Per-frame properties ("observables") evaluated on every (run, cycle, walker)
frame, later averaged per MSM state and used as free-energy coordinates.

Built-in kind 'ca_rmsd'
    CA RMSD after a Kabsch fit to a reference PDB. The simulation CA sequence
    is aligned to the chosen reference chain (Biopython if available, else a
    built-in Needleman-Wunsch), so renumbered, gapped or homologous references
    work. With ``properties.common_core`` every RMSD uses only CA matched to
    ALL references. Optional ``metric_resid_ranges`` / ``alignment_resid_ranges``
    (0-based inclusive simulation residue indices) separate the fitted atoms
    from the measured ones.

Kind 'custom'
    ``func(xyz, box_lengths, topology)`` on the atoms of ``selection``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import ca_table, load_callable
from .io import FrameComputer

ANGSTROM_PER_NM = 10.0


# --------------------------------------------------------------------------- #
def _resolve_chain_index(topology, chain) -> int:
    if isinstance(chain, (int, np.integer)):
        return int(chain)
    for ch in topology.chains:
        if getattr(ch, 'chain_id', None) == chain:
            return ch.index
    available = sorted({str(getattr(c, 'chain_id', c.index)) for c in topology.chains})
    raise ValueError(f'chain {chain!r} not found; available chains: {available}')


def _align_sequences(sim_seq: str, ref_seq: str) -> List[Tuple[int, int]]:
    """Global alignment -> [(sim_pos, ref_pos)] for aligned columns."""
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError:
        return _needleman_wunsch_pairs(sim_seq, ref_seq)
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = 'global'
    best = aligner.align(sim_seq, ref_seq)[0]
    pairs = []
    for (s0, s1), (r0, _) in zip(*best.aligned):
        pairs.extend((s0 + k, r0 + k) for k in range(s1 - s0))
    return pairs


def _needleman_wunsch_pairs(a: str, b: str) -> List[Tuple[int, int]]:
    na, nb = len(a), len(b)
    score = np.zeros((na + 1, nb + 1), dtype=np.int32)
    trace = np.zeros((na + 1, nb + 1), dtype=np.int8)      # 0 diag, 1 up, 2 left
    score[:, 0] = -2 * np.arange(na + 1)
    score[0, :] = -2 * np.arange(nb + 1)
    trace[1:, 0], trace[0, 1:] = 1, 2
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            choices = (score[i - 1, j - 1] + (2 if a[i - 1] == b[j - 1] else -1),
                       score[i - 1, j] - 2, score[i, j - 1] - 2)
            trace[i, j] = int(np.argmax(choices))
            score[i, j] = choices[trace[i, j]]
    pairs, i, j = [], na, nb
    while i or j:
        d = trace[i, j]
        if i and j and d == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and (j == 0 or d == 1):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


@dataclass
class RefMatch:
    """Matched CA atoms between the simulation topology and one reference."""
    label: str
    sim_ca: np.ndarray            # atom indices in the simulation topology
    ref_xyz: np.ndarray           # (n_matched, 3) reference CA coordinates, nm
    report: str

    def subset(self, keep: np.ndarray, note: str) -> 'RefMatch':
        return RefMatch(self.label, self.sim_ca[keep], self.ref_xyz[keep],
                        self.report + note)


def match_reference(sim_topology, ref, require_identity: bool = False) -> RefMatch:
    """Sequence-align the simulation CA to the reference chain's CA."""
    import mdtraj as mdj
    if not os.path.isfile(ref.path):
        raise FileNotFoundError(f'reference {ref.label!r}: PDB not found: {ref.path}')
    rt = mdj.load_pdb(ref.path)
    chain_idx = _resolve_chain_index(rt.topology, ref.chain)
    sim_seq, sim_ca, _ = ca_table(sim_topology)
    ref_seq, ref_ca, _ = ca_table(rt.topology, chain_index=chain_idx)
    if ref_ca.size == 0:
        raise RuntimeError(f'reference {ref.label!r}: chain {ref.chain!r} has no CA')
    pairs = _align_sequences(sim_seq, ref_seq)
    n_mis = sum(sim_seq[s] != ref_seq[r] for s, r in pairs)
    if require_identity:
        pairs = [(s, r) for s, r in pairs if sim_seq[s] == ref_seq[r]]
    if not pairs:
        raise RuntimeError(f'reference {ref.label!r}: no CA matched; is chain '
                           f'{ref.chain!r} correct?')
    sim_idx = np.asarray([sim_ca[s] for s, _ in pairs], dtype=int)
    ref_idx = np.asarray([ref_ca[r] for _, r in pairs], dtype=int)
    report = (f'reference {ref.label!r} chain {ref.chain!r}: sim CA {sim_ca.size}, '
              f'ref CA {ref_ca.size}, matched {len(pairs)} '
              f'(mismatches {n_mis}, {"dropped" if require_identity else "kept"})')
    return RefMatch(ref.label, sim_idx, rt.xyz[0][ref_idx], report)


def common_core(matches: Dict[str, RefMatch]) -> Dict[str, RefMatch]:
    """Restrict every match to the simulation CA shared by all references."""
    common = set.intersection(*(set(m.sim_ca.tolist()) for m in matches.values()))
    if not common:
        raise RuntimeError('references share no common CA with the simulation')
    out = {}
    for lab, m in matches.items():
        keep = np.isin(m.sim_ca, sorted(common))
        out[lab] = m.subset(keep, f' [common core {int(keep.sum())}]')
    return out


def restrict_ranges(m: RefMatch, topology, ranges) -> RefMatch:
    if ranges is None:
        return m
    ranges = [(int(lo), int(hi)) for lo, hi in ranges]
    keep = np.asarray([any(lo <= topology.atom(int(a)).residue.index <= hi
                           for lo, hi in ranges) for a in m.sim_ca])
    if not keep.any():
        raise ValueError(f'reference {m.label!r}: residue ranges {ranges} select no CA')
    return m.subset(keep, f' [ranges {ranges}: {int(keep.sum())} CA]')


def kabsch_rmsd(mobile_fit: np.ndarray, target_fit: np.ndarray,
                mobile_metric: np.ndarray, target_metric: np.ndarray) -> np.ndarray:
    """Per-frame RMSD on metric atoms after the optimal fit on fit atoms.

    mobile_* : (n_frames, n, 3); target_* : (n, 3).
    """
    mobile_fit = np.asarray(mobile_fit, dtype=np.float64)
    mc = mobile_fit.mean(axis=1, keepdims=True)
    tc = target_fit.mean(axis=0, keepdims=True)
    cov = np.einsum('fai,aj->fij', mobile_fit - mc, target_fit - tc)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(np.einsum('fij,fjk->fik', u, vt)))
    u[:, :, -1] *= d[:, None]
    rot = np.einsum('fij,fjk->fik', u, vt)
    fitted = np.einsum('fai,fij->faj',
                       np.asarray(mobile_metric, dtype=np.float64) - mc, rot) + tc
    return np.sqrt(np.mean(np.sum((fitted - target_metric) ** 2, axis=2), axis=1))


# --------------------------------------------------------------------------- #
def computers(cfg, topology) -> List[FrameComputer]:
    """One FrameComputer per configured observable (named by observable)."""
    p = cfg.properties
    out: List[FrameComputer] = []
    rmsd_obs = [o for o in p.observables if o.kind == 'ca_rmsd']
    matches: Dict[str, RefMatch] = {}
    if rmsd_obs:
        matches = {r.label: match_reference(topology, r, p.require_identity)
                   for r in p.references}
        if p.common_core and len(matches) > 1:
            matches = common_core(matches)
        for m in matches.values():
            print('  ' + m.report)
    scale = ANGSTROM_PER_NM if p.units == 'angstrom' else 1.0

    for o in p.observables:
        if o.kind == 'ca_rmsd':
            base = matches[o.reference]
            metric = restrict_ranges(base, topology, o.metric_resid_ranges)
            align = restrict_ranges(base, topology, o.alignment_resid_ranges)
            atoms = np.union1d(metric.sim_ca, align.sim_ca)
            fit_pos = np.searchsorted(atoms, align.sim_ca)
            met_pos = np.searchsorted(atoms, metric.sim_ca)

            def _rmsd(xyz, box, fit_pos=fit_pos, met_pos=met_pos,
                      align=align, metric=metric):
                return scale * kabsch_rmsd(xyz[:, fit_pos], align.ref_xyz,
                                           xyz[:, met_pos], metric.ref_xyz)
            out.append(FrameComputer(o.name, atoms, _rmsd, unwrap=cfg.unwrap_protein))
        else:
            func = load_callable(o.function)
            atoms = topology.select(o.selection)
            if atoms.size == 0:
                raise ValueError(f'observable {o.name!r}: selection '
                                 f'{o.selection!r} selects no atoms')
            sub = topology.subset(atoms)
            out.append(FrameComputer(
                o.name, atoms,
                lambda xyz, box, func=func, sub=sub: np.asarray(func(xyz, box, sub)),
                unwrap=cfg.unwrap_protein))
    return out


def cluster_averages(values_by_run: Sequence[np.ndarray],
                     labels_by_run: Sequence[np.ndarray],
                     weights_by_run: Optional[Sequence[np.ndarray]],
                     n_clusters: int) -> np.ndarray:
    """Per-state mean of a scalar property (weighted if weights are given).

    Empty states are NaN.
    """
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in values_by_run])
    lab = np.concatenate([np.asarray(a).ravel() for a in labels_by_run])
    w = (np.ones_like(v) if weights_by_run is None else
         np.concatenate([np.asarray(a, dtype=float).ravel() for a in weights_by_run]))
    num = np.bincount(lab, weights=v * w, minlength=n_clusters)
    den = np.bincount(lab, weights=w, minlength=n_clusters)
    return np.divide(num, den, out=np.full(n_clusters, np.nan), where=den > 0)
