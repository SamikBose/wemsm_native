"""
pipeline.py
===========

Stages, in order:

    features    per-run features (+ walker weights) from the HDF5 files
    tica        tICA trained on within-run lineage pairs, projections
    cluster     k-means on the pooled TICs
    properties  per-run per-frame observables
    msm         weighted lineage counts -> T -> CSN stationary weights,
                frame weights and GEXF networks
    fe          reweighted free-energy surfaces

``Pipeline`` first builds a ``store.Plan`` covering every artifact, scans the
output directory and decides REUSE / BUILD / REBUILD for each one. Conflicts
stop the run before anything is computed. Data are then loaded lazily: a
reused downstream artifact never forces its inputs to be read.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from . import cluster as clustering
from . import features, io, lineage, msm, observables, plots, store, tica
from .config import Config, as_jsonable
from .store import Item, Plan

STAGES = ['features', 'tica', 'cluster', 'properties', 'msm', 'fe']


class Pipeline:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.specs = cfg.run_specs()
        self.labels = [s.label for s in self.specs]
        self._data: Dict[str, object] = {}
        self._shapes: Dict[str, tuple] = {}
        self._windows: Dict[tuple, np.ndarray] = {}
        self._topology = None
        self.plan = self._build_plan()
        self.plan.resolve()

    # ------------------------------------------------------------------ #
    # plan
    # ------------------------------------------------------------------ #
    def _run_params(self, spec) -> dict:
        return {'h5_path': os.path.abspath(spec.h5_path), 'run_idx': spec.run_idx,
                'max_cycles': spec.max_cycles}

    def _build_plan(self) -> Plan:
        cfg, p = self.cfg, Plan()
        f = cfg.features
        topo = os.path.abspath(cfg.topology_pdb) if cfg.topology_pdb else 'hdf5'

        uses_pair_item = f.kind == 'ca_distances' and not f.pairs_path
        if uses_pair_item:
            p.add(Item('pairs', 'features',
                       os.path.join(cfg.stage_dir('features'), f'pairs_{cfg.feat_tag}.pkl'),
                       {'cutoff_nm': f.cutoff_nm, 'min_seq_sep': f.min_seq_sep,
                        'resid_range': f.resid_range, 'topology': topo,
                        'selected_from': [self._run_params(s) for s in self.specs]},
                       'features.recompute', f.recompute))
        feat_params = {'kind': f.kind, 'topology': topo, 'unwrap': cfg.unwrap_protein}
        if f.kind == 'custom':
            feat_params.update(function=f.custom_function, selection=f.custom_selection)
        elif f.pairs_path:
            feat_params['pairs_path'] = os.path.abspath(f.pairs_path)

        for s in self.specs:
            p.add(Item(f'weights:{s.label}', 'features',
                       os.path.join(cfg.stage_dir('features'), f'{s.label}__weights.pkl'),
                       self._run_params(s), 'features.recompute', f.recompute))
            p.add(Item(f'features:{s.label}', 'features',
                       os.path.join(cfg.stage_dir('features'), f'{s.label}__{cfg.feat_tag}.pkl'),
                       {**self._run_params(s), **feat_params}, 'features.recompute',
                       f.recompute, upstream=['pairs'] if uses_pair_item else []))

        fb = cfg.lineage.allow_contiguous_fallback
        p.add(Item('tica', 'tica',
                   os.path.join(cfg.stage_dir('tica'), f'tica_{cfg.tica_tag}.pkl'),
                   {'lag': cfg.tica.lag, 'dim': cfg.tica.dim, 'scaling': cfg.tica.scaling,
                    'runs': self.labels, 'allow_contiguous_fallback': fb},
                   'tica.retrain', cfg.tica.retrain,
                   upstream=[f'features:{l}' for l in self.labels]))
        c = cfg.cluster
        p.add(Item('cluster', 'cluster',
                   os.path.join(cfg.stage_dir('cluster'), f'kmeans_{cfg.cluster_tag}.pkl'),
                   {'n_clusters': c.n_clusters, 'fixed_seed': c.fixed_seed,
                    'max_iter': c.max_iter}, 'cluster.recluster', c.recluster,
                   upstream=['tica']))

        pr = cfg.properties
        refs = {r.label: {'path': os.path.abspath(r.path), 'chain': r.chain}
                for r in pr.references}
        for o in pr.observables:
            op = as_jsonable(o)
            if o.kind == 'ca_rmsd':
                op.update(references=refs, common_core=pr.common_core,
                          require_identity=pr.require_identity, units=pr.units)
            for s in self.specs:
                p.add(Item(f'properties:{s.label}:{o.name}', 'properties',
                           os.path.join(cfg.stage_dir('properties'),
                                        f'{s.label}__{o.name}.pkl'),
                           {**self._run_params(s), **op, 'topology': topo,
                            'unwrap': cfg.unwrap_protein},
                           'properties.recompute', pr.recompute))

        m = cfg.msm
        p.add(Item('msm', 'msm',
                   os.path.join(cfg.stage_dir('msm'), f'msm_{cfg.msm_tag}.pkl'),
                   {'lag': m.lag, 'scheme': m.scheme, 'tol': m.tol,
                    'runs': self.labels, 'allow_contiguous_fallback': fb},
                   'msm.rebuild', m.rebuild,
                   upstream=['cluster'] + [f'weights:{l}' for l in self.labels]))
        return p

    def needed_items(self, stage: str) -> List[str]:
        """Items a stage depends on (transitively), plus its own."""
        roots = {'features': [n for n in self.plan.items if n.startswith(('features:', 'weights:'))],
                 'tica': ['tica'], 'cluster': ['cluster'],
                 'properties': [n for n in self.plan.items if n.startswith('properties:')],
                 'msm': ['msm'], 'fe': ['msm', 'cluster', 'tica']}[stage]
        seen, todo = [], list(roots)
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.append(n)
                todo.extend(self.plan.items[n].upstream)
        return seen

    def check(self, stages: List[str]) -> None:
        """Print the plan for these stages and stop on conflicts."""
        names = set()
        for st in stages:
            names.update(self.needed_items(st))
        # properties are optional inputs to msm/fe (GEXF attributes, FE axes):
        # they are reported, and a conflict there only excludes them.
        optional = set()
        if 'msm' in stages or 'fe' in stages:
            optional = {n for n in self.plan.items
                        if n.startswith('properties:') and n not in names}
        print(f'system : {self.cfg.system_name}')
        print(f'runs   : {self.labels}')
        print(f'output : {os.path.abspath(self.cfg.output_dir)}')
        print('plan:')
        for n, it in self.plan.items.items():
            if n in names or n in optional:
                opt = '  (optional)' if n in optional else ''
                print(f'  {it.action.upper():8s} {n:40s} {it.path}{opt}')
        print()
        required = Plan()
        required.items = {k: v for k, v in self.plan.items.items() if k in names}
        required.raise_on_conflict()
        for n in sorted(optional):
            it = self.plan.items[n]
            if it.action == 'conflict':
                print(f'  NOTE: {n} is out of date and will not be used '
                      f'({"; ".join(it.reasons)}). Set properties.recompute=True '
                      f'and run the properties stage to refresh it.')

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def topology(self):
        if self._topology is None:
            self._topology = io.load_topology(self.cfg)
        return self._topology

    def shape(self, spec) -> tuple:
        if spec.label not in self._shapes:
            self._shapes[spec.label] = io.run_shape(spec)
        return self._shapes[spec.label]

    def windows(self, spec, lag: int) -> np.ndarray:
        key = (spec.label, lag)
        if key not in self._windows:
            nc, nw = self.shape(spec)
            self._windows[key] = lineage.run_windows(
                spec, lag, nc, nw, self.cfg.lineage.allow_contiguous_fallback)
        return self._windows[key]

    def get(self, name: str):
        if name not in self._data:
            data, _ = self.plan.run(name, lambda: self._build(name))
            self._data[name] = data
        return self._data[name]

    def _spec(self, label: str):
        return self.specs[self.labels.index(label)]

    def _check_run_array(self, label: str, arr: np.ndarray, what: str) -> None:
        nc, nw = self.shape(self._spec(label))
        if arr.shape[:2] != (nc, nw):
            raise store.StaleArtifactError(
                f'[{label}] cached {what} has shape {arr.shape[:2]} but the run now '
                f'has {(nc, nw)} (cycles, walkers). Set the rebuild flag for {what}.')

    # ------------------------------------------------------------------ #
    # builders
    # ------------------------------------------------------------------ #
    def _build(self, name: str):
        kind, _, rest = name.partition(':')
        if name == 'pairs':
            return self._build_pairs()
        if kind == 'weights':
            return io.read_weights(self._spec(rest))
        if kind == 'features':
            return self._extract_run(rest, name)
        if kind == 'properties':
            return self._extract_run(rest.split(':', 1)[0], name)
        return getattr(self, f'_build_{name}')()

    def _build_pairs(self):
        f = self.cfg.features
        _, ca, _ = features.ca_table(self.topology, resid_range=f.resid_range)
        ca_resid = np.asarray([self.topology.atom(int(i)).residue.index for i in ca])
        structs = []
        for s in self.specs:
            pos, box = io.read_frame(s, 0, 0)
            structs.append({'label': s.label, 'pos': pos, 'box': box,
                            'ca_idxs': ca, 'ca_resid': ca_resid})
        return features.build_union_pairs(structs, f.cutoff_nm, f.min_seq_sep)

    def _pairs(self):
        f = self.cfg.features
        if f.kind != 'ca_distances':
            return None
        if f.pairs_path:
            return np.asarray(store.load(f.pairs_path), dtype=int)
        return self.get('pairs')

    def _extract_run(self, label: str, requested: str) -> np.ndarray:
        """One HDF5 pass computing every item of this run that must be built.

        ``requested`` is returned (and saved by the caller's Plan.run); the
        other features/properties items of this run that are planned for a
        build are saved here, so a run's positions are read only once.
        """
        spec = self._spec(label)
        todo = [n for n, it in self.plan.items.items()
                if it.action in ('build', 'rebuild') and n not in self._data
                and (n == f'features:{label}' or n.startswith(f'properties:{label}:'))]
        if requested not in todo:
            todo.append(requested)
        comps = []
        if f'features:{label}' in todo:
            comps.append(features.feature_computer(self.cfg, self.topology, self._pairs()))
        obs_needed = {n.split(':', 2)[2] for n in todo if n.startswith('properties:')}
        if obs_needed:
            comps.extend(c for c in observables.computers(self.cfg, self.topology)
                         if c.name in obs_needed)
        result = io.extract_run(spec, comps)
        for c in comps:
            item = (f'features:{label}' if c.name == 'features'
                    else f'properties:{label}:{c.name}')
            if item == requested:
                continue
            it = self.plan.items[item]
            print(f'  [{it.action}] {it.path}')
            it.meta = store.save(it.path, result[c.name], it.stage, it.params,
                                 self.plan.upstream_ids(item))
            self._data[item] = result[c.name]
        return result['features' if requested.startswith('features:')
                      else requested.split(':', 2)[2]]

    def _features_by_run(self) -> List[np.ndarray]:
        out = []
        for l in self.labels:
            a = self.get(f'features:{l}')
            self._check_run_array(l, a, 'features')
            out.append(a)
        return out

    def _weights_by_run(self) -> List[np.ndarray]:
        out = []
        for l in self.labels:
            a = self.get(f'weights:{l}')
            self._check_run_array(l, a, 'weights')
            out.append(a)
        return out

    def _build_tica(self):
        t = self.cfg.tica
        feats = self._features_by_run()
        wins = [self.windows(s, t.lag) for s in self.specs]
        model = tica.train(feats, wins, t.lag, t.dim, t.scaling)
        tics = tica.project(model, feats)
        ts = tica.timescales(model, t.lag)
        if ts is not None:
            print(f'  [tica] implied timescales (cycles): {np.array2string(ts, precision=2)}')
        return {'model': model, 'tics': tics, 'timescales': ts}

    def _build_cluster(self):
        c = self.cfg.cluster
        centers, labels = clustering.kmeans(self.get('tica')['tics'], c.n_clusters,
                                            c.fixed_seed, c.max_iter)
        return {'centers': centers, 'labels': labels}

    def _build_msm(self):
        m, k = self.cfg.msm, self.cfg.cluster.n_clusters
        labels = self.get('cluster')['labels']
        weights = self._weights_by_run()
        for l, lab in zip(self.labels, labels):
            self._check_run_array(l, lab, 'cluster labels')
        wins = [self.windows(s, m.lag) for s in self.specs]
        counts = msm.build_counts(labels, weights, wins, k, m.scheme)
        w, csn = msm.stationary_weights(counts, m.tol)
        return {'counts': counts, 'transmat': msm.transition_matrix(counts),
                'revo_cluster_weights': msm.ensemble_cluster_weights(labels, weights, k),
                'msm_weights': w, 'trim_indices': np.asarray(csn.trim_indices)}

    # ------------------------------------------------------------------ #
    # stage entry points
    # ------------------------------------------------------------------ #
    def stage_features(self):
        self._features_by_run()
        self._weights_by_run()

    def stage_tica(self):
        self.get('tica')

    def stage_cluster(self):
        self.get('cluster')

    def stage_properties(self):
        for o in self.cfg.properties.observables:
            for l in self.labels:
                self._check_run_array(l, self.get(f'properties:{l}:{o.name}'),
                                      f'property {o.name}')
        if not self.cfg.properties.observables:
            print('  no properties configured')

    def _properties_available(self) -> Dict[str, List[np.ndarray]]:
        """Scalar properties whose every run is cached (or built) and consistent."""
        out = {}
        for o in self.cfg.properties.observables:
            names = [f'properties:{l}:{o.name}' for l in self.labels]
            if all(self.plan.items[n].action == 'reuse' or n in self._data for n in names):
                vals = [self.get(n) for n in names]
                if all(v.ndim == 2 for v in vals):
                    out[o.name] = vals
        return out

    def stage_msm(self):
        cfg, m = self.cfg, self.cfg.msm
        res = self.get('msm')
        labels = self.get('cluster')['labels']
        fw = msm.frame_weights(labels, res['msm_weights'], m.frame_weight_mapping)
        fw_path = os.path.join(cfg.stage_dir('msm'),
                               f'frame_weights_{cfg.msm_tag}_{m.frame_weight_mapping}.pkl')
        store.save(fw_path, {'run_labels': self.labels, 'frame_weights': fw,
                             'mapping': m.frame_weight_mapping},
                   'msm', {'mapping': m.frame_weight_mapping},
                   {'msm': self.plan.items['msm'].meta['id']})
        print(f'  wrote {fw_path}')
        if m.write_gexf:
            self._write_gexf(res, labels)

    def _write_gexf(self, res, labels):
        import networkx as nx
        cfg, k = self.cfg, self.cfg.cluster.n_clusters
        csn = msm.make_csn(res['counts'])
        w = res['msm_weights']
        weights = self._weights_by_run()
        occ = np.bincount(np.concatenate([l.ravel() for l in labels]), minlength=k)
        csn.add_attr('occupancy', [int(x) for x in occ])
        csn.add_attr('cluster_weight', [float(x) for x in res['revo_cluster_weights']])
        csn.add_attr('msm_weight', [float(x) for x in w])
        csn.add_attr('msm_fe', [float(x) for x in msm.free_energy(w, None)])
        csn.add_attr('msm_free_energy_kcal_mol',
                     [float(x) for x in msm.free_energy(w, cfg.msm.temperature)])
        centers = self.get('cluster')['centers']
        for j in range(centers.shape[1]):
            csn.add_attr(f'tic{j + 1}_center', [float(x) for x in centers[:, j]])
        for name, vals in self._properties_available().items():
            safe = ''.join(ch if ch.isalnum() else '_' for ch in name)
            csn.add_attr(f'avg_{safe}', [float(x) for x in observables.cluster_averages(
                vals, labels, None, k)])
            csn.add_attr(f'revo_avg_{safe}', [float(x) for x in observables.cluster_averages(
                vals, labels, weights, k)])
        for kind, g in (('full', csn.graph), ('trimmed', csn.trim_graph)):
            path = os.path.join(cfg.stage_dir('msm'), f'csn_{kind}_{cfg.msm_tag}.gexf')
            nx.write_gexf(g, path)
            print(f'  wrote {path}')

    def _coordinate(self, key: str) -> List[np.ndarray]:
        if key.lower().startswith('tic') and key[3:].isdigit():
            j = int(key[3:]) - 1
            tics = self.get('tica')['tics']
            if not 0 <= j < tics[0].shape[-1]:
                raise ValueError(f'{key}: tICA has {tics[0].shape[-1]} components')
            return [t[..., j] for t in tics]
        props = self._properties_available()
        if key not in props:
            raise ValueError(f'FE coordinate {key!r} is neither ticN nor a computed '
                             f'scalar property (available: {sorted(props)}). Run the '
                             f'properties stage first.')
        return props[key]

    def stage_fe(self):
        cfg, fe = self.cfg, self.cfg.fe
        out_dir = cfg.stage_dir('fe')
        for weighting in fe.weightings:
            if weighting == 'msm':
                res = self.get('msm')
                w = msm.frame_weights(self.get('cluster')['labels'], res['msm_weights'],
                                      cfg.msm.frame_weight_mapping)
                note = f'MSM ({cfg.cluster.n_clusters} states, {cfg.msm.frame_weight_mapping} mapping)'
            elif weighting == 'revo':
                w, note = self._weights_by_run(), 'REVO walker weights'
            else:
                w, note = None, 'UNWEIGHTED (diagnostic only)'
            wflat = None if w is None else np.concatenate([a.ravel() for a in w])
            for xk, yk in fe.surfaces:
                x = np.concatenate([a.ravel() for a in self._coordinate(xk)])
                y = np.concatenate([a.ravel() for a in self._coordinate(yk)])
                F, P, xe, ye = plots.free_energy_2d(x, y, wflat, fe.bins,
                                                    temperature=fe.temperature,
                                                    pmin=fe.pmin)
                stem = os.path.join(out_dir, f'{cfg.system_name}_{xk}_{yk}_{weighting}'
                                    + (f'_{cfg.msm_tag}_{cfg.msm.frame_weight_mapping}'
                                       if weighting == 'msm' else ''))
                np.savez(stem + '.npz', F=F, P=P, xedges=xe, yedges=ye)
                plots.plot_fe(F, xe, ye, stem + '.png', xk, yk,
                              f'{cfg.system_name}: {note}', fe.temperature, fe.dpi)
                print(f'  wrote {stem}.png/.npz')

    # ------------------------------------------------------------------ #
    def run(self, stages: List[str]) -> None:
        self.check(stages)
        for st in STAGES:
            if st in stages:
                print(f'===== {st} =====')
                getattr(self, f'stage_{st}')()
        print('done.')
