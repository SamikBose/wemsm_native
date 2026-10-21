#!/usr/bin/env python3
"""Make small example wepy HDF5 files by stripping solvent from real runs.

Everything wepy needs is copied unchanged (weights, box vectors, resampling and
resampler records, decision enum, settings), except that positions keep only
the atoms of ``--selection`` and the stored topology / atom-count settings are
updated to match. The clone/merge history is therefore the real one.

    python make_example_data.py SRC.h5 DEST.h5 [--selection protein]
"""

import argparse

import h5py
import numpy as np


def strip(src: str, dest: str, selection: str = 'protein') -> None:
    from wepy.util.mdtraj import json_to_mdtraj_topology, mdtraj_to_json_topology
    with h5py.File(src, 'r') as fin, h5py.File(dest, 'w') as fout:
        top_json = fin['topology'][()]
        top = json_to_mdtraj_topology(top_json.decode() if isinstance(top_json, bytes)
                                      else top_json)
        keep = top.select(selection)
        if keep.size == 0:
            raise ValueError(f'selection {selection!r} selects no atoms')
        n_all = top.n_atoms

        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                g = fout.require_group(name)
                g.attrs.update(obj.attrs)
                return
            if name == 'topology':
                return
            if name.endswith('/positions') and obj.ndim == 3 and obj.shape[1] == n_all:
                data = np.asarray(obj[()])[:, keep, :]
                fout.create_dataset(name, data=data, compression='gzip',
                                    compression_opts=4, chunks=(1,) + data.shape[1:])
            elif name == '_settings/main_rep_idxs':
                fout[name] = np.arange(keep.size, dtype=obj.dtype)
            elif name == '_settings/n_atoms':
                fout[name] = np.asarray(keep.size, dtype=obj.dtype)
            else:
                fin.copy(obj, fout, name=name)
            fout[name].attrs.update(obj.attrs)

        fout.attrs.update(fin.attrs)
        fin.visititems(copy)
        fout.create_dataset('topology', data=mdtraj_to_json_topology(top.subset(keep)),
                            dtype=h5py.string_dtype())
    print(f'{dest}: kept {keep.size}/{n_all} atoms ({selection!r})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src')
    ap.add_argument('dest')
    ap.add_argument('--selection', default='protein')
    a = ap.parse_args()
    strip(a.src, a.dest, a.selection)
