# Configuration reference

[Back to the README](../README.md) · [Runnable SiaP example](../examples/siap_example_config.py) · [Production template](../examples/example_config.py)

Run CLI examples from the repository root. Configuration fields below use
the same names in Python, TOML, and JSON. See the README for
[command-line overrides](../README.md#command-line) and
[cache and rebuild rules](../README.md#cache-reuse-and-explicit-rebuilds).

A config is one of:

- a `.py` file that defines `CONFIG = Config(...)`;
- a `.toml` file;
- a `.json` file with the same nesting.

TOML needs Python ≥ 3.11 or the `tomli` package. On Python 3.10, use
`.py`/`.json` or run `pip install tomli`.

Paths in a `.py` config can be built from `__file__` so they are relative to
the config file (as in `siap_example_config.py`). Paths in TOML/JSON are
relative to the directory you run from.

## Top level (`Config`)

| field | default | meaning |
|---|---|---|
| `system_name` | `'system'` | used in FE file names and plot titles |
| `output_dir` | `'wemsm_out'` | where all products are written and looked for |
| `runs` | `[]` | list of `RunSpec`; if given, overrides the template fields below |
| `md_root`, `h5_template`, `seed_list` | `None`, `None`, `[]` | template runs: `h5_template.format(root=md_root, c=seed)` for each seed |
| `run_label_template` | `'clust{c}'` | label of each template run |
| `max_cycles` | `None` | cycle cap applied to every template run |
| `topology_pdb` | `None` | topology PDB; `None` uses the topology JSON stored in the first run's HDF5 |
| `unwrap_protein` | `False` | make the selected atoms whole (minimum-image against the first selected atom) before RMSD/custom functions; only for a compact molecule split across the box |

## `RunSpec(label, h5_path, run_idx=0, max_cycles=None)`

| field | meaning |
|---|---|
| `label` | unique run name, used in file names |
| `h5_path` | wepy results HDF5 |
| `run_idx` | run index inside the HDF5 (`/runs/<run_idx>`) |
| `max_cycles` | analyse only cycles `[0, max_cycles)`; `None` = all |

## `features` (`FeatureParams`)

| field | default | meaning |
|---|---|---|
| `kind` | `'ca_distances'` | `'ca_distances'` or `'custom'` |
| `cutoff_nm` | `1.0` | CA–CA pair kept if closer than this in frame 0 of any run |
| `min_seq_sep` | `5` | pair kept only if residue indices differ by more than this |
| `resid_range` | `None` | `(lo, hi)` 0-based inclusive residue window for pair selection |
| `pairs_path` | `None` | pickle of an `(n_pairs, 2)` atom-index array to use instead of selecting pairs |
| `custom_function` | `None` | `'file.py:func'` or `'module:func'` for `kind='custom'` |
| `custom_selection` | `'all'` | mdtraj selection of atoms passed to the custom function |
| `tag` | `None` | name used in file names; derived from the settings if `None` (e.g. `cadist_c1_s5`) |
| `recompute` | `False` | rebuild pairs, features and weights |

## `lineage` (`LineageParams`)

| field | default | meaning |
|---|---|---|
| `allow_contiguous_fallback` | `False` | if resampling records can't be read, use per-walker windows `(w, c) → (w, c+lag)` instead of stopping. Only correct without cloning/merging. |

## `tica` (`TICAParams`)

| field | default | meaning |
|---|---|---|
| `lag` | `1` | lag in WE cycles; sliding window length `lag + 1`. `> 1` issues a `WETICALagWarning`. |
| `dim` | `2` | number of TICs kept |
| `scaling` | `'kinetic_map'` | deeptime TICA scaling (`'kinetic_map'`, `'commute_map'` or `None`) |
| `retrain` | `False` | rebuild the tICA model and projections |

## `cluster` (`ClusterParams`)

| field | default | meaning |
|---|---|---|
| `n_clusters` | `100` | k-means states |
| `fixed_seed` | `True` | deeptime KMeans fixed seed (reproducible labels) |
| `max_iter` | `500` | k-means iterations |
| `recluster` | `False` | rebuild centers and labels |

## `properties` (`PropertyParams`)

| field | default | meaning |
|---|---|---|
| `references` | `[]` | list of `ReferenceSpec(label, path, chain='A')`; `chain` is a chain ID letter or 0-based index |
| `observables` | `[]` | list of `ObservableSpec` (below) |
| `common_core` | `True` | with several references, every RMSD uses only CA matched to all of them |
| `require_identity` | `False` | drop aligned CA pairs whose residue types differ |
| `units` | `'angstrom'` | RMSD units: `'angstrom'` or `'nm'` |
| `recompute` | `False` | rebuild properties |

`ObservableSpec(name, kind='ca_rmsd', reference=None, function=None, selection='all', metric_resid_ranges=None, alignment_resid_ranges=None)`:

| field | meaning |
|---|---|
| `name` | unique property name; used as an FE coordinate and in GEXF attributes |
| `kind` | `'ca_rmsd'` (Kabsch-fitted CA RMSD to `reference`) or `'custom'` |
| `reference` | `ReferenceSpec` label for `ca_rmsd` |
| `function`, `selection` | for `custom`: callable spec and mdtraj atom selection |
| `metric_resid_ranges` | list of `(lo, hi)` 0-based residue ranges whose CA are measured (default all matched) |
| `alignment_resid_ranges` | list of `(lo, hi)` ranges whose CA are used for the fit (default all matched) |

The simulation CA sequence is aligned to the reference chain, so references
may be renumbered, have gaps or be homologs. The matched-atom count is printed;
an unexpectedly low count usually means the wrong chain.

## `msm` (`MSMParams`)

| field | default | meaning |
|---|---|---|
| `lag` | `1` | MSM lag in WE cycles (no warning is issued for `> 1`) |
| `scheme` | `'w2'` | `'w2'`: `counts[x2,x1] += w2`; `'w2w1'`: `+= w2/w1` |
| `tol` | `1e-10` | CSNAnalysis `calc_mult_weights` tolerance |
| `frame_weight_mapping` | `'cluster'` | `'cluster'` (= `msm.py`) or `'normalized'`, see [Methodology](../README.md#methodology) |
| `temperature` | `300.0` | K, for the kcal/mol free energies on GEXF nodes |
| `write_gexf` | `True` | write full and trimmed CSN GEXF files |
| `rebuild` | `False` | rebuild counts, T and CSN weights |

## `fe` (`FEParams`)

| field | default | meaning |
|---|---|---|
| `surfaces` | `[('tic1','tic2')]` | pairs of coordinates: `ticN` (1-based) or a scalar property name |
| `weightings` | `['msm','revo']` | any of `'msm'` (MSM frame weights), `'revo'` (raw walker weights), `'none'` (unweighted; diagnostic only) |
| `bins` | `100` | histogram bins per axis |
| `temperature` | `300.0` | K; `None` plots −ln P instead of kcal/mol |
| `pmin` | `1e-20` | bins with P below this are blank |
| `dpi` | `300` | PNG resolution |

## TOML example

```toml
system_name = "SiaP_example"
output_dir = "siap_toml_out"

[[runs]]
label = "nmerge2"
h5_path = "examples/data/SiaP_clust2_nmerge2.h5"

[[runs]]
label = "nmerge5"
h5_path = "examples/data/SiaP_clust2_nmerge5.h5"
max_cycles = 7

[tica]
lag = 1
dim = 2

[cluster]
n_clusters = 20

[properties]
references = [{label = "start", path = "examples/data/SiaP_clust2_start.pdb", chain = 0}]
observables = [{name = "rmsd_start", kind = "ca_rmsd", reference = "start"}]

[fe]
surfaces = [["tic1", "tic2"], ["tic1", "rmsd_start"]]
```
