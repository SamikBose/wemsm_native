# wemsm_native

`wemsm_native` builds Markov state models (MSMs) from WEPY weighted ensemble
simulation data. It follows resampling records (walker lineage) to build time-lagged data for tica training or weighted counts matrices.
It also project the simulation data into states (namely kmeans clusters) and compute state properties. 
From the transition count matrices, one can estimate stationary state probabilities, transition rates etc. with 
CSNAnalysis. Also, one can build networkx objects and free energy landscapes directly from the `wemsm_native`-derived transition matrices, using CSNAnalysis.

**Important:** It should be noted that the current version of `wemsm_native` is implemented without merging bias correction (MBS-MSM). The MBC-MSM version will be added soon. 

```text
WEPY HDF5 ->  features -> tICA training -> k-means -> weighted transition counts -> MSM state weights -> networks and free-energy plots
```

## Quick start

With Python ≥ 3.10 and Git installed, run the complete workflow on the bundled
SiaP (sialic acid binding protein) example data without editing any paths:

```bash
git clone https://github.com/SamikBose/wemsm_native.git
cd wemsm_native
python -m pip install -e ".[align]"
wemsm-native all --config examples/siap_example_config.py
```

Results are written to `examples/siap_example_out/`: features, tICA projections,
cluster assignments, transition matrices, stationary state weights, GEXF
networks, and free-energy plots with their underlying arrays.


For other installation options, see [Install](#install). To analyze your own
simulations, see [Use your own data](#use-your-own-data) and the
[configuration reference](docs/configuration.md).

Contents:

- [Quick start](#quick-start)
- [Install](#install)
- [Use your own data](#use-your-own-data)
- [Bundled example data](#bundled-example-data)
- [Command line](#command-line)
- [Configuration reference](docs/configuration.md)
- [Stages and output files](#stages-and-output-files)
- [Using the results in Python](#using-the-results-in-python)
- [Methodology](#methodology)
- [Cache reuse and explicit rebuilds](#cache-reuse-and-explicit-rebuilds)
- [Custom features and properties](#custom-features-and-properties)
- [Requirements and assumptions](#requirements-and-assumptions)
- [Differences from msm.py](#differences-from-msmpy)
- [Tests](#tests)

## Install

Requires Python ≥ 3.10. `pip` installs every dependency, including
[CSNAnalysis](https://github.com/ADicksonLab/CSNAnalysis), which is not on
PyPI and is fetched from GitHub (network access is needed during
installation).

**Option A: pip, in any Python ≥ 3.10 environment**

```bash
git clone https://github.com/SamikBose/wemsm_native.git
cd wemsm_native
pip install -e ".[align]"        # [align] adds Biopython; optional but recommended
```

Or install directly from GitHub without cloning; the bundled examples and
tests are then not available locally:

```bash
pip install "wemsm_native[align] @ git+https://github.com/SamikBose/wemsm_native.git"
```

**Option B: conda**

```bash
git clone https://github.com/SamikBose/wemsm_native.git
cd wemsm_native
conda env create -f environment.yml          # creates env "wemsm_native"
conda activate wemsm_native
```

**Using an existing environment that already has wepy.** Install without
touching the existing packages:

```bash
pip install --no-deps -e .
```

This needs numpy, h5py, mdtraj, deeptime, wepy, networkx, matplotlib and
csnanalysis to be present already. (At MSU, the `aSAM_wepy` conda env has all
of them.)

**Check the installation:**

```bash
wemsm-native --help
python -m unittest discover -s tests -v                      # 34 tests, ~10 s
wemsm-native all --config examples/siap_example_config.py    # example run
```

Tested with:
- Python 3.10;
- wepy 1.1.0 and 1.2;
- deeptime 0.4.5;
- mdtraj 1.10.3;
- numpy 1.26 and 2.2;
- CSNAnalysis 0.1.0b0.

Biopython is optional. It is used for the sequence alignment behind CA RMSD,
and a built-in Needleman–Wunsch is used without it.

## Use your own data

Copy the production template, edit the run paths and analysis settings, then
inspect the plan and run the pipeline:

```bash
cp examples/example_config.py my_config.py
wemsm-native status --config my_config.py
wemsm-native all --config my_config.py
```

The [configuration reference](docs/configuration.md) documents every setting,
including input runs, features, tICA, clustering, observables, and MSM weights.

## Bundled example data

`examples/data/` holds three real SiaP REVO runs (aSAM cluster 2, 10 cycles x
48 walkers, merge distances 2, 5 and 10; about 34 MB each) and the reference
PDB. They were made from the original 731 MB `wepy.results.h5` files by
`examples/make_example_data.py`, which drops solvent atoms and copies
everything else unchanged, including the real resampling records:

| file | source |
|---|---|
| `SiaP_clust2_nmerge{2,5,10}.h5` | `wepy_sims/SiaP_2CEY/aSAM_clust2/nmerge{N}_tica_test_clust2/run0_walk48_steps10000_cycs10_substep0_nmerge{N}/wepy.results.h5` |
| `SiaP_clust2_start.pdb` | protein of cycle 0, walker 0 of the nmerge2 run |

Run the example (about 10 s); results go to `examples/siap_example_out/`:

```bash
wemsm-native all --config examples/siap_example_config.py
```

Two of the runs are capped at 7 and 5 cycles to show runs of different
lengths. Results are identical to running the pipeline on the original
unstripped files. To make similar files from other runs:

```bash
python examples/make_example_data.py SRC.h5 DEST.h5 --selection protein
```

| example file | purpose |
|---|---|
| `examples/example_config.py` | template for production runs (all main options) |
| `examples/siap_example_config.py` | runnable config for the bundled data |
| `examples/make_example_data.py` | strips atoms from a wepy HDF5 file and keeps wepy compatibility |

## Command line

```
wemsm-native COMMAND --config FILE [--set SECTION.KEY=VALUE ...]
python -m wemsm_native COMMAND --config FILE [...]          # equivalent
```

| command | does |
|---|---|
| `status` | lists every artifact found in `output_dir`, then prints the REUSE/BUILD/REBUILD/CONFLICT plan for all stages. Computes nothing. |
| `features` | features and walker weights per run (also computes configured properties in the same pass) |
| `tica` | tICA model and projections |
| `cluster` | k-means |
| `properties` | per-frame properties |
| `msm` | counts, transition matrix, CSN weights, frame weights, GEXF |
| `fe` | free-energy surfaces |
| `all` | every stage in the order above |

A stage builds whatever it needs upstream: for example, `msm` on an empty
`output_dir` runs features, tica and cluster first.

`--set` overrides any config field and can be repeated. The value is parsed as
JSON when possible (`true`, `200`, `null`, `[1, 2]`) and as a string otherwise,
e.g. `--set cluster.n_clusters=200 --set msm.scheme=w2w1 --set tica.retrain=true`.

Exit codes: 0 on success, 2 when existing files conflict with the configuration
(see [Cache reuse](#cache-reuse-and-explicit-rebuilds)). Other errors raise a
Python exception (non-zero exit).

## Configuration reference

See the [full configuration reference](docs/configuration.md) for every
field, its default, supported config formats, and a complete TOML example.
Start with the [runnable SiaP config](examples/siap_example_config.py) or
the [production template](examples/example_config.py).

## Stages and output files

All per-frame arrays are stored per run as `(n_cycles, n_walkers, ...)`, so
runs may have different numbers of cycles and walkers. Lists below are ordered
like `runs`.

| file (under `output_dir/`) | contents |
|---|---|
| `features/pairs_<feat_tag>.pkl` | `(n_pairs, 2)` int atom indices (not written when `pairs_path` is given) |
| `features/<run>__<feat_tag>.pkl` | `(n_cycles, n_walkers, n_features)` float32 features, nm |
| `features/<run>__weights.pkl` | `(n_cycles, n_walkers)` walker weights from the HDF5 |
| `tica/tica_<feat_tag>_lag<L>_dim<D>.pkl` | dict: `model` (deeptime TICA model), `tics` (list of `(n_cycles, n_walkers, D)`), `timescales` |
| `cluster/kmeans_<tica_tag>_k<K>.pkl` | dict: `centers` `(K, D)`, `labels` (list of `(n_cycles, n_walkers)` ints) |
| `properties/<run>__<name>.pkl` | `(n_cycles, n_walkers)` (or `(..., k)` for vector custom properties) |
| `msm/msm_<msm_tag>.pkl` | dict: `counts` `(K, K)` `[to, from]`, `transmat` (column-stochastic), `revo_cluster_weights`, `msm_weights` `(K,)`, `trim_indices` (states in the main component) |
| `msm/frame_weights_<msm_tag>_<mapping>.pkl` | dict: `frame_weights` (list of `(n_cycles, n_walkers)`), `run_labels` (run names, same order as `frame_weights`), `mapping` |
| `msm/csn_full_<msm_tag>.gexf`, `msm/csn_trimmed_<msm_tag>.gexf` | CSN networks for Gephi (all states / main component) |
| `fe/<system>_<x>_<y>_<weighting>[...].png` / `.npz` | plot, and arrays `F` (kcal/mol or −ln P, NaN = empty), `P`, `xedges`, `yedges` |

Tags: `feat_tag` is, for example, `cadist_c1_s5`. `tica_tag` =
`<feat_tag>_lag<L>_dim<D>`, `cluster_tag` = `<tica_tag>_k<K>`, and `msm_tag` =
`<cluster_tag>_msmlag<L>_<scheme>`. MSM-weighted FE files also carry
`_<msm_tag>_<mapping>`.

Every `.pkl` has a `.meta.json` sidecar, and a forced rebuild keeps the
previous file as `.prev`. The frame-weight file, the GEXF files and the FE
files are cheap and are rewritten every time their stage runs. Lineage windows
are recomputed from the resampling records on each run and are not cached.

Units: positions and features are in nm, RMSD is in Å by default, and free
energies are in kcal/mol at `temperature`.

GEXF node attributes:
- `label`, `count` and `trim` (from CSNAnalysis);
- `occupancy` (number of frames);
- `cluster_weight` (summed raw walker weights, normalised);
- `msm_weight`;
- `msm_fe` (−ln w, minimum 0, as in `msm.py`; 100 for zero weight);
- `msm_free_energy_kcal_mol`;
- `tic<N>_center`;
- for every scalar property whose files are all up to date:
  - `avg_<property>`, the plain mean per state;
  - `revo_avg_<property>`, the walker-weighted mean per state.

## Using the results in Python

```python
import pickle, numpy as np
from wemsm_native import load_config
from wemsm_native.pipeline import Pipeline

cfg  = load_config('examples/siap_example_config.py')
pipe = Pipeline(cfg)                     # plans only; files are loaded on demand
res    = pipe.get('msm')                 # counts, transmat, msm_weights, ...
labels = pipe.get('cluster')['labels']   # cluster assignments per run (n_cycles, n_walkers)
tics   = pipe.get('tica')['tics']        # per run (n_cycles, n_walkers, dim)
rmsd   = pipe.get('properties:nmerge2:rmsd_start')

# state weights -> frame weights -> free-energy landscape
from wemsm_native import msm, plots
fw = msm.frame_weights(labels, res['msm_weights'], 'cluster')
x = np.concatenate([t[..., 0].ravel() for t in tics])
y = np.concatenate([t[..., 1].ravel() for t in tics])
F, P, xe, ye = plots.free_energy_2d(x, y, np.concatenate([w.ravel() for w in fw]),
                                    bins=100, temperature=300.0)
```

`pipe.get(name)` loads a product if it is cached and builds it otherwise, under
the same reuse and conflict rules as the CLI. Call `pipe.check([...stages])`
first to see the plan. Product names are:
- `pairs`;
- `features:<run>`;
- `weights:<run>`;
- `tica`;
- `cluster`;
- `properties:<run>:<name>`;
- `msm`.

`pipe.run(['msm', 'fe'])` runs stages exactly as the CLI does.

## Methodology

- **Lineage windows.** These come from each run's own resampling records
  (`resampling_panel` → `parent_panel(MultiCloneMergeDecision)` →
  `net_parent_table` → `sliding_window`, all from `wepy.analysis.parents`; the
  contig tree is not used).
  - Windows are built per run and only index that run's arrays, so a window
    never joins two runs. Each window is checked to span consecutive cycles
    inside its run.
  - The parent table is cut to the analysed cycles before `sliding_window`.
  - If a run has one parent-table row fewer than its stored cycles (no
    resampling after the last cycle), an identity row is appended. That row is
    never read as a parent, but without it `sliding_window` would drop every
    window ending at the last cycle. Fewer rows than that is an error.
- **No fallback without asking.** If resampling records can't be read, the run
  stops. `lineage.allow_contiguous_fallback=True` opts into per-walker windows,
  which are only correct without cloning or merging.
- **Features.**
  - CA–CA pairs are the union, over frame 0 / walker 0 of every run, of pairs
    closer than `cutoff_nm` with residue separation `> min_seq_sep`.
  - Distances use the minimum-image convention, matching mdtraj periodic
    distances.
  - Features and properties are computed in one pass over each run, reading
    only the contiguous atom span they need.
- **tICA.**
  - Time-lagged pairs are the two ends of each lineage window. They are built
    per run and then pooled, so both ends of a pair always come from the same
    run.
  - Training uses the WE data with `tica.lag = 1` by default: two-snapshot
    lineage windows, one resampling step.
  - deeptime's TICA is unweighted.
  - With `tica.lag > 1`, a `WETICALagWarning` says these are WE datapoints
    joined through several resampling steps, with unweighted tICA training.
- **Clustering.** deeptime k-means runs on the TICs of all frames of all runs
  pooled together. Labels are split back per run.
- **Counts.** Counting follows `msm.py`: `counts[x2, x1] += w2` (`scheme='w2'`)
  or `w2/w1` (`'w2w1'`), where `w` are the raw HDF5 walker weights. For
  `w2w1`, windows with `w1 = 0` are skipped and reported. Each run's windows
  index only that run's labels and weights; the per-run counts are then
  summed.
- **Stationary weights.** CSNAnalysis `trim()` then `calc_mult_weights()`.
  States outside the main component get weight 0. There is no eigenvector
  fallback: a missing CSNAnalysis, an empty count matrix, failed trimming, or
  weights that are non-finite, negative or don't sum to 1 all raise `MSMError`
  with an explanation.
- **Frame weights** (`msm.frame_weight_mapping`):
  - `'cluster'` (default): each frame gets its state's MSM weight,
    `msm_weights[labels]`, exactly as in `msm.py`.
  - `'normalized'`: the state weight is divided by the number of frames in the
    state, so a state's frame weights sum to its MSM weight.

  The free-energy histogram is normalised in both cases. The difference is
  that with `'cluster'`, a state's contribution to P also scales with how many
  frames it contains.
- **REVO weighting.** The `'revo'` free energy and the `cluster_weight` node
  attribute use the raw walker weights of all frames of all runs, pooled. Each
  run's weights sum to 1 at every cycle, so a run with more cycles contributes
  proportionally more. This matches `msm.py`'s segment 3.
- **Free energy.** `F = −kT ln P`, shifted so the minimum is 0. Bins with
  `P < pmin` are blank.
- **Properties in msm/fe.** These are optional inputs to the msm and fe stages.
  If a property's files are out of date, the msm stage notes it and leaves the
  property out of the GEXF attributes. An `fe` surface that uses it fails with
  a message to run the properties stage.

## Cache reuse and explicit rebuilds

Each product is a pickle plus a `<file>.meta.json` sidecar that records the
parameters, the input files and the ids of the upstream products it was built
from. Before computing anything, the pipeline scans `output_dir` and plans
every product:

- **REUSE**: the file exists and matches the configuration. It is loaded.
- **BUILD**: the file does not exist. New parameters, such as a new cluster
  count, give new file names, so those products are built alongside the old
  ones.
- **REBUILD**: the stage's flag is set. The old file is kept as `<file>.prev`.
- **CONFLICT**: the file exists but disagrees with the configuration (for
  example different runs or `max_cycles`), has no sidecar, or one of its
  inputs is being rebuilt or is itself out of date. The run stops before
  computing anything, lists every reason and names the flags needed.

| flag | rebuilds |
|---|---|
| `features.recompute` | pair list, per-run features and weights |
| `tica.retrain` | tICA model and projections |
| `cluster.recluster` | k-means centers and labels |
| `properties.recompute` | per-run properties |
| `msm.rebuild` | counts, T, CSN weights |

For example, `--set tica.retrain=true` alone is refused if a clustering and an
MSM already exist for that tICA; add `--set cluster.recluster=true --set
msm.rebuild=true`. Adding or removing a run changes the recorded run list, so
it needs `features.recompute` (the pair union changes) and the downstream
flags. Alternatively, use a new `output_dir`.

## Custom features and properties

A custom function receives:
- `xyz`: `(n_frames, n_selected_atoms, 3)` in nm, in selection order;
- `box`: `(n_frames, 3)` box lengths in nm;
- `topology`: an mdtraj topology of the selected atoms.

It must return `(n_frames,)` or `(n_frames, k)`. It is called once per walker
with all of that walker's cycles. For FE axes and GEXF averages, properties
must be scalar `(n_frames,)`.

```python
# my_obs.py
import numpy as np
def radius_of_gyration(xyz, box, topology):
    c = xyz.mean(axis=1, keepdims=True)
    return np.sqrt(((xyz - c) ** 2).sum(-1).mean(-1))
```

```python
ObservableSpec('rg', kind='custom', function='my_obs.py:radius_of_gyration',
               selection='name CA')
FeatureParams(kind='custom', custom_function='my_feats.py:func', custom_selection='name CA')
```

## Requirements and assumptions

- wepy HDF5 layout `/runs/<run_idx>/trajectories/<walker>/{positions,box_vectors,weights}`
  with resampling records (`/runs/<run_idx>/resampling`).
- Orthorhombic boxes. A triclinic cell raises an error because the
  minimum-image distances assume a rectangular box.
- Constant walker count within a run, with walkers numbered `0..n-1`. Runs may
  differ in walkers and cycles.
- All runs share the same atom ordering (one topology): pair and atom indices
  are applied to every run.
- `topology_pdb=None` takes the topology from the first run's HDF5.
- Clone/merge decisions are interpreted with `MultiCloneMergeDecision`, as in
  `msm.py`.

## Differences from msm.py

The weighted counts and CSN stationary weights follow the reference
`adkexample/ticaWE_MSM/build_MarkovModels/msm.py` workflow.

- Run paths, counts of runs/cycles/walkers and the tICA model are not
  hardcoded: runs are configured, shapes are read from each file, and tICA is
  trained on the WE data.
- Runs of different lengths are supported; data are lists per run instead of
  one `(nruns, n_cycles, n_walkers)` array.
- Lineage windows are restricted to the analysed cycles (`max_cycles`) and
  validated, and the last-cycle windows are kept when the final resampling row
  is missing (see [Methodology](#methodology)).
- Caches are checked against the configuration instead of being reused just
  because a file exists.
- CSN errors are reported instead of silently producing weights.
- GEXF files are written with a `.gexf` extension (`msm.py` used `.pkl`). Node
  attributes include the kcal/mol free energy, TIC centers and property
  averages.
- Properties (CA RMSD, custom) and reweighted free-energy surfaces are new
  stages.

## Tests

```bash
python -m unittest discover -s tests -v        # 34 tests, ~10 s
```

`tests/test_example_data.py` (20 tests) runs the whole pipeline on the bundled
real SiaP data, using wepy's own resampling records with no patching. It
checks each step against an independent transcription of the reference
`msm.py` (segments 3, 5, 6 and 7):

- **Lineage windows** are identical to `sliding_window` over wepy's
  `net_parent_table`. They include real clone/merge lineages and stay inside
  their run.
- **Weighted counts** are exactly equal for `w2` and `w2w1`. The transition
  matrix, the CSN `calc_mult_weights` weights and the cluster weights also
  match.
- **Default frame weights** equal `msm_weights[cluster_labels]`. The
  normalized mapping preserves the state weights.
- **Features** match mdtraj periodic distances. The RMSD of the reference
  frame is 0. The walker weights match the HDF5 file.
- **Runs of different lengths:** labels and TICs keep their per-run
  (10, 7, 5) × 48 shapes.
- **Outputs:** FE `.npz` files are normalised, and the GEXF files carry all
  node attributes.
- **Caching and errors:**
  - a second run reuses everything;
  - `tica.lag > 1` warns;
  - the CLI `status` command works;
  - an implicit rebuild exits with code 2 and names the flags to set;
  - unreadable resampling records raise an error.

`tests/test_wemsm.py` (14 tests) uses tiny synthetic runs with known
clone/merge parent tables. It covers:

- lineage edge cases;
- counts compared with the `msm.py` loop;
- CSN weights compared with the stationary eigenvector;
- CSN error cases;
- both frame-weight mappings;
- that tICA pairs never cross runs;
- the reuse, conflict and rebuild rules.

The example-data tests are skipped if `examples/data` or one of wepy,
deeptime or csnanalysis is missing.

## License

MIT; see [LICENSE](LICENSE).
