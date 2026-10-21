"""
features.py
===========

Features used to train tICA.

  ca_table            -- protein CA atom indices / resSeq / 1-letter sequence
  build_union_pairs   -- CA-CA pairs with |i-j| > min_seq_sep and d < cutoff in
                         at least one structure (union over structures)
  distance_features   -- vectorised minimum-image distances for atom pairs
  feature_computer    -- the FrameComputer for the configured feature kind

A custom feature function is called as ``func(xyz, box_lengths, topology)``
with xyz holding only the atoms of ``features.custom_selection``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .io import FrameComputer, minimum_image

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


# --------------------------------------------------------------------------- #
def load_callable(spec: str) -> Callable:
    """Import ``'path/to/file.py:func'`` or ``'package.module:func'``."""
    if ':' not in spec:
        raise ValueError(f"callable spec must be 'module:function'; got {spec!r}")
    mod_part, func = spec.rsplit(':', 1)
    if mod_part.endswith('.py'):
        if not os.path.isfile(mod_part):
            raise FileNotFoundError(f'custom function file not found: {mod_part}')
        name = 'wemsm_user_' + os.path.splitext(os.path.basename(mod_part))[0]
        s = importlib.util.spec_from_file_location(name, mod_part)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)
    if not hasattr(mod, func):
        raise AttributeError(f'{mod_part} has no function {func!r}')
    return getattr(mod, func)


# --------------------------------------------------------------------------- #
def ca_table(topology, chain_index: Optional[int] = None,
             resid_range: Optional[Tuple[int, int]] = None):
    """(sequence, ca_atom_indices, resseq) for protein CA atoms.

    ``resid_range`` is a 0-based inclusive residue-index window.
    """
    seq, idxs, resseq = [], [], []
    for res in topology.residues:
        if not res.is_protein:
            continue
        if chain_index is not None and res.chain.index != chain_index:
            continue
        if resid_range is not None and not (resid_range[0] <= res.index <= resid_range[1]):
            continue
        ca = [a for a in res.atoms if a.name == 'CA']
        if not ca:
            continue
        seq.append(THREE2ONE.get(res.name, 'X'))
        idxs.append(ca[0].index)
        resseq.append(res.resSeq)
    return ''.join(seq), np.asarray(idxs, dtype=int), np.asarray(resseq, dtype=int)


def distance_features(pos: np.ndarray, box: Optional[np.ndarray],
                      pairs: np.ndarray) -> np.ndarray:
    """Minimum-image distances (nm).

    pos : (n_atoms, 3) or (n_frames, n_atoms, 3); box : (3,), (n_frames, 3) or
    None; pairs : (n_pairs, 2) atom indices into ``pos``.
    Returns (n_pairs,) or (n_frames, n_pairs).
    """
    pos = np.asarray(pos)
    pairs = np.asarray(pairs, dtype=int)
    single = pos.ndim == 2
    if single:
        pos = pos[None]
    disp = minimum_image(pos[:, pairs[:, 0], :] - pos[:, pairs[:, 1], :], box)
    d = np.sqrt(np.sum(disp ** 2, axis=-1))
    return d[0] if single else d


def ca_pairs_within_cutoff(pos, box, ca_idxs, ca_resid, cutoff_nm: float,
                           min_seq_sep: int, label: str = '') -> np.ndarray:
    """CA-CA atom pairs with |resid_i - resid_j| > min_seq_sep closer than cutoff."""
    n = len(ca_idxs)
    if n == 0:
        raise RuntimeError(f'{label}: no CA atoms selected')
    iu, ju = np.triu_indices(n, k=1)
    keep = np.abs(ca_resid[iu] - ca_resid[ju]) > min_seq_sep
    cand = np.column_stack([ca_idxs[iu[keep]], ca_idxs[ju[keep]]]).astype(np.int64)
    if len(cand) == 0:
        raise RuntimeError(f'{label}: no CA pairs with sequence separation > {min_seq_sep}')
    sel = cand[distance_features(pos, box, cand) < cutoff_nm]
    print(f'    {label}: {n} CA | {len(cand)} seq-filtered pairs | '
          f'{len(sel)} within {cutoff_nm} nm')
    return sel


def build_union_pairs(structures: Sequence[dict], cutoff_nm: float,
                      min_seq_sep: int) -> np.ndarray:
    """Union of per-structure CA-CA pairs, sorted for deterministic ordering.

    structures : dicts with 'label', 'pos' (n_atoms, 3), 'box' ((3,) or None),
        'ca_idxs', 'ca_resid'. The returned pairs are ATOM indices, so every
        run featurised with them must share the same atom ordering.
    """
    pair_set = set()
    for s in structures:
        for p in ca_pairs_within_cutoff(s['pos'], s.get('box'), s['ca_idxs'],
                                        s['ca_resid'], cutoff_nm, min_seq_sep,
                                        label=s.get('label', '')):
            pair_set.add((int(p[0]), int(p[1])))
    if not pair_set:
        raise RuntimeError(f'no CA-CA pairs within {cutoff_nm} nm in any structure')
    union = np.asarray(sorted(pair_set), dtype=np.int64)
    print(f'  union over {len(structures)} structure(s): {len(union)} CA-CA pairs')
    return union


# --------------------------------------------------------------------------- #
def feature_computer(cfg, topology, pairs: Optional[np.ndarray]) -> FrameComputer:
    """FrameComputer named 'features' for the configured feature kind."""
    f = cfg.features
    if f.kind == 'ca_distances':
        pairs = np.asarray(pairs, dtype=int)
        atoms = np.unique(pairs)
        local = np.searchsorted(atoms, pairs)
        return FrameComputer(
            'features', atoms,
            lambda xyz, box: distance_features(xyz, box, local).astype(np.float32))
    func = load_callable(f.custom_function)
    atoms = topology.select(f.custom_selection)
    if atoms.size == 0:
        raise ValueError(f'features.custom_selection {f.custom_selection!r} '
                         f'selects no atoms')
    sub_top = topology.subset(atoms)

    def _custom(xyz, box):
        out = np.asarray(func(xyz, box, sub_top), dtype=np.float32)
        return out[:, None] if out.ndim == 1 else out
    return FrameComputer('features', atoms, _custom, unwrap=cfg.unwrap_protein)
