"""End-to-end tests on the bundled real SiaP REVO data (examples/data).

Unlike test_wemsm.py, nothing is patched here: lineage windows come from
wepy's own resampling records. Every result is checked against an independent
transcription of the reference msm.py
(adkexample/ticaWE_MSM/build_MarkovModels/msm.py, segments 3, 5, 6 and 7).

Run with:  python -m unittest discover -s tests -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SRC = os.path.join(ROOT, 'src')
EXAMPLE_CFG = os.path.join(ROOT, 'examples', 'siap_example_config.py')
DATA = os.path.join(ROOT, 'examples', 'data')
sys.path.insert(0, SRC)

try:
    import csnanalysis                                             # noqa: F401
    import deeptime                                                # noqa: F401
    import wepy                                                    # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
HAVE_DATA = os.path.isfile(os.path.join(DATA, 'SiaP_clust2_nmerge2.h5'))

if HAVE_DEPS:
    from wemsm_native import load_config, lineage, msm, store, tica
    from wemsm_native.pipeline import STAGES, Pipeline


# --------------------------------------------------------------------------- #
# Reference implementation: msm.py, transcribed with the minimum changes
# needed for runs capped at max_cycles (windows ending past the analysed
# cycles are skipped).
# --------------------------------------------------------------------------- #
def msmpy_windows(h5_path, n_cycles):
    from wepy.analysis.parents import (net_parent_table, parent_panel,
                                       resampling_panel, sliding_window)
    from wepy.hdf5 import WepyHDF5
    from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
    wepy_h5 = WepyHDF5(h5_path, mode='r')
    with wepy_h5:
        resampling_rec = wepy_h5.resampling_records([0])
        resamp_panel = resampling_panel(resampling_rec)
        par_panel = parent_panel(MultiCloneMergeDecision, resamp_panel)
        net_par_table = net_parent_table(par_panel)
        sw_1step = sliding_window(net_par_table, window_length=2)
    return [s for s in sw_1step if s[-1][1] < n_cycles]


def msmpy_run_weights(h5_path, n_walkers):
    """load_run_weights from msm.py: shape (n_walkers, n_cycles)."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        out = []
        for walker in range(n_walkers):
            wt = np.array(f[f'/runs/0/trajectories/{walker}/weights'])
            out.append(wt[:, 0] if wt.ndim == 2 else wt)
    return np.array(out)


def msmpy_counts(cluster_labels, specs, n_clusters, scheme_str):
    """Segment 5 of msm.py."""
    counts_lag1 = np.zeros((n_clusters, n_clusters))
    for run_idx, spec in enumerate(specs):
        labels = cluster_labels[run_idx]
        n_cycles, n_walkers = labels.shape
        run_weights = msmpy_run_weights(spec.h5_path, n_walkers)
        for s in msmpy_windows(spec.h5_path, n_cycles):
            w1, c1 = s[0]
            w2, c2 = s[-1]
            x1 = int(labels[c1, w1])
            x2 = int(labels[c2, w2])
            wt2 = run_weights[w2, c2]
            wt1 = run_weights[w1, c1]
            if scheme_str == 'w2':
                counts_lag1[x2, x1] += wt2
            elif scheme_str == 'w2w1':
                counts_lag1[x2, x1] += wt2 / wt1
    return counts_lag1


def msmpy_cluster_weights(cluster_labels, all_weights, n_clusters):
    """Segment 3 of msm.py (triple loop)."""
    cluster_weights = np.zeros(n_clusters)
    for run_idx in range(len(cluster_labels)):
        n_cycles, n_walkers = cluster_labels[run_idx].shape
        for cyc in range(n_cycles):
            for walker in range(n_walkers):
                cl = cluster_labels[run_idx][cyc, walker]
                cluster_weights[cl] += all_weights[run_idx][cyc, walker]
    return cluster_weights / cluster_weights.sum()


def msmpy_transmat(counts_lag1):
    """Segment 6 of msm.py."""
    row_sums = counts_lag1.sum(axis=0, keepdims=True)
    return np.divide(counts_lag1, row_sums, out=np.zeros_like(counts_lag1),
                     where=row_sums > 0)


def msmpy_csn_weights(counts_lag1):
    """Segment 7 of msm.py."""
    from csnanalysis.csn import CSN
    csn = CSN(counts_lag1)
    csn.trim()
    return csn.calc_mult_weights(label='msm_weight', tol=1e-10)


# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_DEPS, 'needs wepy, deeptime and csnanalysis')
@unittest.skipUnless(HAVE_DATA, 'examples/data not present')
class TestExampleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='wemsm_example_')
        cls.cfg = cls.make_cfg()
        cls.pipe = Pipeline(cls.cfg)
        cls.pipe.run(STAGES)
        cls.specs = cls.cfg.run_specs()
        cls.labels = cls.pipe.get('cluster')['labels']
        cls.result = cls.pipe.get('msm')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def make_cfg(cls, **overrides):
        from wemsm_native.config import apply_override
        cfg = load_config(EXAMPLE_CFG)
        cfg.output_dir = os.path.join(cls.tmp, 'out')
        for k, v in overrides.items():
            apply_override(cfg, f'{k}={v}')
        return cfg

    def weights(self):
        return [self.pipe.get(f'weights:{s.label}') for s in self.specs]

    # ---- lineage ---------------------------------------------------------- #
    def test_lineage_windows_identical_to_wepy_sliding_window(self):
        for spec, lab in zip(self.specs, self.labels):
            nc, nw = lab.shape
            ours = self.pipe.windows(spec, 1)
            ref = msmpy_windows(spec.h5_path, nc)
            self.assertEqual({tuple(map(tuple, w)) for w in ours},
                             {tuple(map(tuple, w)) for w in ref}, spec.label)
            self.assertEqual(len(ours), (nc - 1) * nw)

    def test_lineage_has_real_cloning_and_stays_in_run(self):
        for spec, lab in zip(self.specs, self.labels):
            wins = self.pipe.windows(spec, 1)
            self.assertTrue(np.any(wins[:, 0, 0] != wins[:, 1, 0]),
                            f'{spec.label}: no clone/merge lineage found')
            np.testing.assert_array_equal(wins[:, 1, 1] - wins[:, 0, 1], 1)
            self.assertLess(wins[..., 1].max(), lab.shape[0])
            self.assertLess(wins[..., 0].max(), lab.shape[1])

    # ---- data ------------------------------------------------------------- #
    def test_runs_of_different_lengths(self):
        self.assertEqual([l.shape for l in self.labels], [(10, 48), (7, 48), (5, 48)])
        tics = self.pipe.get('tica')['tics']
        self.assertEqual([t.shape for t in tics], [(10, 48, 2), (7, 48, 2), (5, 48, 2)])

    def test_features_match_mdtraj_periodic_distances(self):
        import h5py
        import mdtraj as mdj
        from wepy.util.mdtraj import json_to_mdtraj_topology
        pairs = self.pipe.get('pairs')
        spec = self.specs[1]
        feats = self.pipe.get(f'features:{spec.label}')
        with h5py.File(spec.h5_path, 'r') as f:
            top = json_to_mdtraj_topology(f['topology'][()])
            for cyc, w in [(0, 0), (3, 7), (6, 47)]:
                g = f[f'runs/0/trajectories/{w}']
                t = mdj.Trajectory(g['positions'][cyc][None], top,
                                   unitcell_lengths=np.diag(g['box_vectors'][cyc])[None],
                                   unitcell_angles=np.array([[90.0, 90.0, 90.0]]))
                d = mdj.compute_distances(t, pairs, periodic=True)[0]
                np.testing.assert_allclose(feats[cyc, w], d, atol=1e-5)

    def test_rmsd_zero_for_the_reference_frame(self):
        # the reference PDB is cycle 0, walker 0 of the nmerge2 run
        rmsd = self.pipe.get('properties:nmerge2:rmsd_start')
        self.assertLess(rmsd[0, 0], 1e-3)
        self.assertGreater(rmsd.max(), rmsd[0, 0])

    def test_weights_are_the_hdf5_walker_weights(self):
        for spec, w, lab in zip(self.specs, self.weights(), self.labels):
            ref = msmpy_run_weights(spec.h5_path, lab.shape[1]).T[:lab.shape[0]]
            np.testing.assert_array_equal(w, ref)

    # ---- MSM against msm.py ----------------------------------------------- #
    def test_counts_identical_to_msm_py_w2(self):
        ref = msmpy_counts(self.labels, self.specs, self.cfg.cluster.n_clusters, 'w2')
        np.testing.assert_array_equal(self.result['counts'], ref)

    def test_counts_identical_to_msm_py_w2w1(self):
        wins = [self.pipe.windows(s, 1) for s in self.specs]
        ours = msm.build_counts(self.labels, self.weights(), wins,
                                self.cfg.cluster.n_clusters, 'w2w1')
        ref = msmpy_counts(self.labels, self.specs, self.cfg.cluster.n_clusters, 'w2w1')
        np.testing.assert_allclose(ours, ref, rtol=1e-12)

    def test_transition_matrix_identical_to_msm_py(self):
        T = self.result['transmat']
        np.testing.assert_array_equal(T, msmpy_transmat(self.result['counts']))
        cols = T.sum(axis=0)
        np.testing.assert_allclose(cols[cols > 0], 1.0)

    def test_csn_weights_identical_to_msm_py(self):
        ref = msmpy_csn_weights(self.result['counts'])
        np.testing.assert_allclose(self.result['msm_weights'], ref, rtol=1e-12, atol=0)
        self.assertAlmostEqual(self.result['msm_weights'].sum(), 1.0, places=8)

    def test_cluster_weights_identical_to_msm_py(self):
        ref = msmpy_cluster_weights(self.labels, self.weights(),
                                    self.cfg.cluster.n_clusters)
        np.testing.assert_allclose(self.result['revo_cluster_weights'], ref, rtol=1e-12)

    def test_default_frame_weights_are_msm_py_mapping(self):
        path = os.path.join(self.cfg.output_dir, 'msm',
                            f'frame_weights_{self.cfg.msm_tag}_cluster.pkl')
        saved = store.load(path)
        self.assertEqual(sorted(saved), ['frame_weights', 'mapping', 'run_labels'])
        self.assertEqual(saved['run_labels'], [s.label for s in self.specs])
        self.assertEqual(saved['mapping'], 'cluster')
        fw = saved['frame_weights']
        w = self.result['msm_weights']
        for f, lab in zip(fw, self.labels):
            np.testing.assert_array_equal(f, w[lab])          # msm_weights[cluster_labels]

    def test_normalized_frame_weights_preserve_state_weights(self):
        fw = msm.frame_weights(self.labels, self.result['msm_weights'], 'normalized')
        lab = np.concatenate([l.ravel() for l in self.labels])
        per_state = np.bincount(lab, weights=np.concatenate([f.ravel() for f in fw]),
                                minlength=self.cfg.cluster.n_clusters)
        np.testing.assert_allclose(per_state, self.result['msm_weights'], atol=1e-15)

    # ---- outputs ---------------------------------------------------------- #
    def test_outputs_written(self):
        out = self.cfg.output_dir
        for kind in ('full', 'trimmed'):
            self.assertTrue(os.path.isfile(os.path.join(
                out, 'msm', f'csn_{kind}_{self.cfg.msm_tag}.gexf')))
        npz = os.path.join(out, 'fe', f'SiaP_example_tic1_tic2_msm_{self.cfg.msm_tag}_cluster.npz')
        d = np.load(npz)
        self.assertAlmostEqual(float(d['P'].sum()), 1.0, places=10)
        self.assertEqual(float(np.nanmin(d['F'])), 0.0)

    def test_gexf_attributes(self):
        import networkx as nx
        g = nx.read_gexf(os.path.join(self.cfg.output_dir, 'msm',
                                      f'csn_full_{self.cfg.msm_tag}.gexf'))
        attrs = next(iter(g.nodes(data=True)))[1]
        for key in ('occupancy', 'cluster_weight', 'msm_weight', 'msm_fe',
                    'msm_free_energy_kcal_mol', 'tic1_center', 'tic2_center',
                    'avg_rmsd_start', 'revo_avg_rmsd_start'):
            self.assertIn(key, attrs)

    # ---- reuse, warnings, CLI --------------------------------------------- #
    def test_second_run_reuses_everything(self):
        p = Pipeline(self.make_cfg())
        self.assertEqual({it.action for it in p.plan.items.values()}, {'reuse'})

    def test_tica_lag_above_one_warns(self):
        feats = [self.pipe.get(f'features:{s.label}') for s in self.specs]
        wins = [self.pipe.windows(s, 2) for s in self.specs]
        for w in wins:
            np.testing.assert_array_equal(w[:, 2, 1] - w[:, 0, 1], 2)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            tica.train(feats, wins, lag=2, dim=2)
        msgs = [str(r.message) for r in rec if issubclass(r.category, tica.WETICALagWarning)]
        self.assertTrue(msgs)
        self.assertIn('weighted-ensemble datapoints', msgs[0])
        self.assertIn('UNWEIGHTED', msgs[0])

    def _cli(self, *args):
        env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get('PYTHONPATH', ''))
        return subprocess.run(
            [sys.executable, '-W', 'ignore', '-m', 'wemsm_native', *args,
             '--config', EXAMPLE_CFG, '--set', f'output_dir="{self.cfg.output_dir}"'],
            capture_output=True, text=True, env=env)

    def test_cli_status(self):
        r = self._cli('status')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('REUSE', r.stdout)
        self.assertNotIn('CONFLICT', r.stdout)

    def test_cli_refuses_implicit_rebuild(self):
        r = self._cli('msm', '--set', 'tica.retrain=true')
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('cluster.recluster', r.stderr)
        self.assertIn('msm.rebuild', r.stderr)

    def test_unreadable_records_error_without_fallback(self):
        from wemsm_native.config import RunSpec
        bad = RunSpec('bad', os.path.join(self.tmp, 'no_such_file.h5'))
        with self.assertRaises(lineage.LineageError) as cm:
            lineage.run_windows(bad, 1, 5, 48, allow_fallback=False)
        self.assertIn('allow_contiguous_fallback', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
