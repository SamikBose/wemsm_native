"""Tests for wemsm_native.  Run with:  python -m unittest discover -s tests -v"""

import os
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

import synthetic                                                   # noqa: E402
from wemsm_native import config as C                               # noqa: E402
from wemsm_native import lineage, msm, store, tica                 # noqa: E402
from wemsm_native.pipeline import Pipeline                         # noqa: E402

_ORIGINAL_TABLE = None


def setUpModule():
    # synthetic runs carry their parent table in a test dataset; patch only for
    # this module so other test modules use wepy's real resampling records
    global _ORIGINAL_TABLE
    _ORIGINAL_TABLE = synthetic.patch_lineage()


def tearDownModule():
    lineage._net_parent_table = _ORIGINAL_TABLE


def reference_counts(labels, weights, tables, n_clusters, scheme, lag=1):
    """The counting loop of the reference msm.py, over wepy sliding windows."""
    from wepy.analysis.parents import sliding_window
    counts = np.zeros((n_clusters, n_clusters))
    for r, table in enumerate(tables):
        for s in sliding_window(table, lag + 1):
            w1, c1 = s[0]
            w2, c2 = s[-1]
            x1, x2 = int(labels[r][c1, w1]), int(labels[r][c2, w2])
            wt1, wt2 = weights[r][c1, w1], weights[r][c2, w2]
            counts[x2, x1] += wt2 if scheme == 'w2' else wt2 / wt1
    return counts


# --------------------------------------------------------------------------- #
class TestLineage(unittest.TestCase):
    def test_windows_follow_parents(self):
        table = [[0, 0, 2], [1, 1, 0], [0, 1, 2]]   # row c: parents at c of walkers at c+1
        spec = C.RunSpec('r', 'unused.h5')
        lineage._net_parent_table, saved = (lambda s: table), lineage._net_parent_table
        try:
            wins = lineage.lineage_windows(spec, 1, 3, 3)
        finally:
            lineage._net_parent_table = saved
        pairs = {(tuple(w[0]), tuple(w[1])) for w in wins}
        # walker 1 at cycle 1 descends from walker 0 at cycle 0 (cloned)
        self.assertIn(((0, 0), (1, 1)), pairs)
        self.assertIn(((1, 1), (0, 2)), pairs)
        self.assertIn(((0, 1), (2, 2)), pairs)
        self.assertEqual(len(wins), 6)

    def test_check_rejects_non_consecutive_and_out_of_run(self):
        bad = np.array([[[0, 0], [0, 2]]])
        with self.assertRaises(lineage.LineageError):
            lineage.check_windows(bad, 1, 5, 2, 'x')
        out = np.array([[[0, 3], [0, 4]]])
        with self.assertRaises(lineage.LineageError):
            lineage.check_windows(out, 1, 4, 2, 'x')

    def test_missing_records_is_an_error_unless_fallback(self):
        spec = C.RunSpec('r', 'unused.h5')

        def boom(s):
            raise KeyError('no resampling')
        saved = lineage._net_parent_table
        lineage._net_parent_table = boom
        try:
            with self.assertRaises(lineage.LineageError) as cm:
                lineage.run_windows(spec, 1, 4, 2, allow_fallback=False)
            self.assertIn('allow_contiguous_fallback', str(cm.exception))
            wins = lineage.run_windows(spec, 1, 4, 2, allow_fallback=True)
            self.assertEqual(len(wins), 6)
        finally:
            lineage._net_parent_table = saved


class TestCountsAndWeights(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.k = 6
        self.shapes = [(9, 4), (5, 4), (7, 3)]                   # ragged runs
        self.tables = [synthetic.random_parent_table(rng, nc, nw) for nc, nw in self.shapes]
        self.labels = [rng.integers(0, self.k, s) for s in self.shapes]
        self.weights = [rng.uniform(0.01, 1, s) for s in self.shapes]
        self.wins = []
        for (nc, nw), t in zip(self.shapes, self.tables):
            lineage._net_parent_table, saved = (lambda s, t=t: t), lineage._net_parent_table
            try:
                self.wins.append(lineage.lineage_windows(C.RunSpec('r', 'x'), 1, nc, nw))
            finally:
                lineage._net_parent_table = saved

    def test_counts_match_reference_msm_py(self):
        for scheme in ('w2', 'w2w1'):
            got = msm.build_counts(self.labels, self.weights, self.wins, self.k, scheme)
            ref = reference_counts(self.labels, self.weights,
                                   [t[:nc] for t, (nc, _) in zip(self.tables, self.shapes)],
                                   self.k, scheme)
            np.testing.assert_allclose(got, ref)

    def test_transition_matrix_column_stochastic(self):
        T = msm.transition_matrix(msm.build_counts(self.labels, self.weights,
                                                   self.wins, self.k))
        cols = T.sum(axis=0)
        np.testing.assert_allclose(cols[cols > 0], 1.0)

    def test_frame_weight_mappings(self):
        sw = np.array([0.5, 0.3, 0.2, 0, 0, 0])
        fw = msm.frame_weights(self.labels, sw, 'cluster')
        for l, f in zip(self.labels, fw):
            np.testing.assert_array_equal(f, sw[l])               # msm.py mapping
        fn = msm.frame_weights(self.labels, sw, 'normalized')
        lab = np.concatenate([l.ravel() for l in self.labels])
        tot = np.bincount(lab, weights=np.concatenate([f.ravel() for f in fn]),
                          minlength=6)
        np.testing.assert_allclose(tot, sw)


class TestCSN(unittest.TestCase):
    def test_stationary_matches_eigenvector(self):
        T = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.3], [0.0, 0.1, 0.7]])
        counts = T * 1000
        w, _ = msm.stationary_weights(counts, 1e-12)
        vals, vecs = np.linalg.eig(T)
        v = np.real(vecs[:, np.argmax(np.real(vals))])
        np.testing.assert_allclose(w, v / v.sum(), atol=1e-8)

    def test_empty_counts_raise(self):
        with self.assertRaises(msm.MSMError):
            msm.stationary_weights(np.zeros((3, 3)))

    def test_disconnected_state_gets_zero(self):
        counts = np.array([[5., 1, 0], [1, 5, 0], [0, 0, 0.0]])
        counts[2, 0] = 0.0
        w, _ = msm.stationary_weights(counts)
        self.assertEqual(w[2], 0.0)
        self.assertAlmostEqual(w.sum(), 1.0)


class TestTICA(unittest.TestCase):
    def test_lag_warning(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            tica.lag_warning(1)
            self.assertFalse(rec)
            tica.lag_warning(3)
            self.assertTrue(any(issubclass(r.category, tica.WETICALagWarning) for r in rec))
            self.assertIn('UNWEIGHTED', str(rec[-1].message))

    def test_pairs_never_cross_runs(self):
        f0 = np.zeros((4, 2, 1)); f1 = np.ones((3, 2, 1)) * 100
        w0 = lineage.consecutive_windows(1, 4, 2)
        w1 = lineage.consecutive_windows(1, 3, 2)
        X0, X1 = tica.lagged_pairs([f0, f1], [w0, w1])
        self.assertEqual(len(X0), len(w0) + len(w1))
        np.testing.assert_array_equal(X0[:, 0] == 100, X1[:, 0] == 100)


# --------------------------------------------------------------------------- #
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='wemsm_test_')
        top = os.path.join(cls.tmp, 'top.pdb')
        t = synthetic.make_topology_pdb(top)
        cls.top = top
        cls.runs = []
        for i, (nc, nw) in enumerate([(30, 8), (22, 8), (26, 6)]):      # ragged
            p = os.path.join(cls.tmp, f'run{i}.h5')
            synthetic.make_run(p, t.xyz[0], nc, nw, seed=10 + i)
            cls.runs.append(C.RunSpec(f'r{i}', p))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cfg(self, out, **kw):
        cfg = C.Config(
            system_name='toy', output_dir=os.path.join(self.tmp, out),
            runs=list(self.runs), topology_pdb=self.top,
            features=C.FeatureParams(cutoff_nm=2.0, min_seq_sep=2),
            tica=C.TICAParams(lag=1, dim=2),
            cluster=C.ClusterParams(n_clusters=8),
            properties=C.PropertyParams(
                references=[C.ReferenceSpec('start', self.top, chain=0)],
                observables=[C.ObservableSpec('rmsd_start', 'ca_rmsd', reference='start')]),
            fe=C.FEParams(surfaces=[('tic1', 'tic2'), ('tic1', 'rmsd_start')],
                          weightings=['msm', 'revo', 'none'], bins=20, dpi=50))
        for k, v in kw.items():
            C.apply_override(cfg, f'{k}={v}')
        return cfg

    def actions(self, pipe):
        return {n: it.action for n, it in pipe.plan.items.items()}

    def test_full_run_reuse_and_explicit_rebuild(self):
        cfg = self.cfg('out1')
        Pipeline(cfg).run(['features', 'tica', 'cluster', 'properties', 'msm', 'fe'])

        # second pass: everything reused, nothing conflicting
        p2 = Pipeline(self.cfg('out1'))
        self.assertEqual(set(self.actions(p2).values()), {'reuse'})

        # shapes are ragged and preserved per run
        labels = p2.get('cluster')['labels']
        self.assertEqual([l.shape for l in labels], [(30, 8), (22, 8), (26, 6)])
        res = p2.get('msm')
        self.assertAlmostEqual(res['msm_weights'].sum(), 1.0)
        self.assertTrue(os.path.isfile(os.path.join(cfg.output_dir, 'msm',
                                                    f'csn_full_{cfg.msm_tag}.gexf')))

        # a new cluster count gets new files; tica/features still reused
        p3 = Pipeline(self.cfg('out1', **{'cluster.n_clusters': 5}))
        a = self.actions(p3)
        self.assertEqual(a['tica'], 'reuse')
        self.assertEqual(a['cluster'], 'build')
        self.assertEqual(a['msm'], 'build')

        # retraining tICA without reclustering is refused, naming the flag
        p4 = Pipeline(self.cfg('out1', **{'tica.retrain': 'true'}))
        with self.assertRaises(store.StaleArtifactError) as cm:
            p4.check(['msm'])
        self.assertIn('cluster.recluster', str(cm.exception))
        self.assertIn('msm.rebuild', str(cm.exception))

        # ... and allowed when every flag is explicit
        p5 = Pipeline(self.cfg('out1', **{'tica.retrain': 'true',
                                           'cluster.recluster': 'true',
                                           'msm.rebuild': 'true'}))
        p5.run(['msm'])
        self.assertTrue(os.path.isfile(p5.plan.items['tica'].path + '.prev'))

    def test_config_change_conflict(self):
        Pipeline(self.cfg('out2')).run(['features'])
        cfg = self.cfg('out2')
        cfg.runs = [C.RunSpec(r.label, r.h5_path, max_cycles=10) for r in self.runs]
        with self.assertRaises(store.StaleArtifactError) as cm:
            Pipeline(cfg).check(['features'])
        self.assertIn('features.recompute', str(cm.exception))
        self.assertIn('max_cycles', str(cm.exception))

    def test_counts_equal_reference_on_pipeline_data(self):
        cfg = self.cfg('out3')
        p = Pipeline(cfg)
        p.run(['msm'])
        labels = p.get('cluster')['labels']
        weights = [p.get(f'weights:{r.label}') for r in self.runs]
        import h5py
        tables = []
        for r in self.runs:
            with h5py.File(r.h5_path, 'r') as f:
                tables.append(np.asarray(f['test_parent_table']).tolist())
        ref = reference_counts(labels, weights, tables, cfg.cluster.n_clusters, 'w2')
        np.testing.assert_allclose(p.get('msm')['counts'], ref)


if __name__ == '__main__':
    unittest.main()
