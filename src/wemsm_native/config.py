"""
config.py
=========

Configuration dataclasses. A config is a Python file defining ``CONFIG``, a
TOML file or a JSON file (see ``load_config``).

Rebuild flags
-------------
Every cached product is REUSED if it already exists in ``output_dir``. Nothing
is recomputed unless the corresponding flag is set explicitly:

    features.recompute   per-run features (and the CA-CA pair list)
    tica.retrain         tICA model and projections
    cluster.recluster    k-means model and labels
    properties.recompute per-run properties (observables)
    msm.rebuild          transition counts, transition matrix, CSN weights

If a cached file is inconsistent with the current configuration (different
runs, cycles, parameters) or with its rebuilt upstream products, the run stops
before any computation and names the flag(s) that must be set.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# --------------------------------------------------------------------------- #
@dataclass
class RunSpec:
    """One WE run: a label plus the HDF5 file (and run index) that holds it."""
    label: str
    h5_path: str
    run_idx: int = 0
    max_cycles: Optional[int] = None         # None -> every cycle in the file


@dataclass
class FeatureParams:
    # 'ca_distances' -> minimum-image CA-CA distances (union pair selection)
    # 'custom'       -> user function, see README
    kind: str = 'ca_distances'
    cutoff_nm: float = 1.0
    min_seq_sep: int = 5
    resid_range: Optional[Tuple[int, int]] = None   # 0-based inclusive
    pairs_path: Optional[str] = None         # explicit (n_pairs, 2) atom-pair pickle
    custom_function: Optional[str] = None    # 'file.py:func' or 'module:func'
    custom_selection: str = 'all'            # mdtraj selection passed to it
    tag: Optional[str] = None                # filename tag; derived if None
    recompute: bool = False


@dataclass
class LineageParams:
    # Lineage windows come from each run's resampling records. If they cannot
    # be read the pipeline stops, unless this is set, in which case per-walker
    # consecutive windows are used (only correct without cloning/merging).
    allow_contiguous_fallback: bool = False


@dataclass
class TICAParams:
    lag: int = 1                             # WE cycles (sliding window = lag + 1)
    dim: int = 2
    scaling: Optional[str] = 'kinetic_map'
    retrain: bool = False


@dataclass
class ClusterParams:
    n_clusters: int = 100
    fixed_seed: bool = True
    max_iter: int = 500
    recluster: bool = False


@dataclass
class ReferenceSpec:
    """A reference structure for CA RMSD properties."""
    label: str
    path: str
    chain: Union[str, int] = 'A'


@dataclass
class ObservableSpec:
    """One per-frame property.

    kind='ca_rmsd': CA RMSD (Kabsch fit) to ``reference`` (a ReferenceSpec label).
    kind='custom' : ``function`` ('file.py:func' or 'module:func') called as
        func(xyz, box_lengths, topology) -> (n_frames,) or (n_frames, k),
        where xyz holds only the atoms of ``selection`` (nm).
    """
    name: str
    kind: str = 'ca_rmsd'
    reference: Optional[str] = None
    function: Optional[str] = None
    selection: str = 'all'
    metric_resid_ranges: Optional[List[Tuple[int, int]]] = None
    alignment_resid_ranges: Optional[List[Tuple[int, int]]] = None


@dataclass
class PropertyParams:
    references: List[ReferenceSpec] = field(default_factory=list)
    observables: List[ObservableSpec] = field(default_factory=list)
    common_core: bool = True                 # CA matched to ALL references
    require_identity: bool = False
    units: str = 'angstrom'                  # RMSD units: 'angstrom' or 'nm'
    recompute: bool = False


@dataclass
class MSMParams:
    lag: int = 1                             # WE cycles
    scheme: str = 'w2'                       # 'w2' or 'w2w1'
    tol: float = 1e-10
    # 'cluster'    -> every frame gets its state's full MSM weight
    #                 (msm_weights[labels], as in the reference msm.py)
    # 'normalized' -> state weight divided by the number of frames in the
    #                 state, so per-frame weights sum to the state weight
    frame_weight_mapping: str = 'cluster'
    temperature: float = 300.0               # K, for kcal/mol free energies
    write_gexf: bool = True
    rebuild: bool = False


@dataclass
class FEParams:
    # Pairs of coordinates: 'tic1', 'tic2', ... or a property name.
    surfaces: List[Tuple[str, str]] = field(default_factory=lambda: [('tic1', 'tic2')])
    weightings: List[str] = field(default_factory=lambda: ['msm', 'revo'])
    bins: int = 100
    temperature: Optional[float] = 300.0
    pmin: float = 1e-20
    dpi: int = 300


# --------------------------------------------------------------------------- #
@dataclass
class Config:
    system_name: str = 'system'
    output_dir: str = 'wemsm_out'

    # Runs: give them explicitly, or use md_root + h5_template + seed_list.
    runs: List[RunSpec] = field(default_factory=list)
    md_root: Optional[str] = None
    h5_template: Optional[str] = None        # formatted with {root} and {c}
    seed_list: List[int] = field(default_factory=list)
    run_label_template: str = 'clust{c}'
    max_cycles: Optional[int] = None         # default cap for template runs

    # None -> topology JSON stored inside the first HDF5 file
    topology_pdb: Optional[str] = None
    unwrap_protein: bool = False

    features: FeatureParams = field(default_factory=FeatureParams)
    lineage: LineageParams = field(default_factory=LineageParams)
    tica: TICAParams = field(default_factory=TICAParams)
    cluster: ClusterParams = field(default_factory=ClusterParams)
    properties: PropertyParams = field(default_factory=PropertyParams)
    msm: MSMParams = field(default_factory=MSMParams)
    fe: FEParams = field(default_factory=FEParams)

    # ===================================================================== #
    def run_specs(self) -> List[RunSpec]:
        if self.runs:
            return list(self.runs)
        if not self.h5_template:
            return []
        return [RunSpec(label=self.run_label_template.format(c=c),
                        h5_path=self.h5_template.format(root=self.md_root, c=c),
                        max_cycles=self.max_cycles)
                for c in self.seed_list]

    # ---- filename tags --------------------------------------------------- #
    @property
    def feat_tag(self) -> str:
        f = self.features
        if f.tag:
            return f.tag
        if f.kind == 'custom':
            return 'custom_' + str(f.custom_function).split(':')[-1]
        if f.pairs_path:
            return 'cadist_' + os.path.splitext(os.path.basename(f.pairs_path))[0]
        tag = f'cadist_c{f.cutoff_nm:g}_s{f.min_seq_sep}'
        if f.resid_range is not None:
            tag += f'_r{f.resid_range[0]}-{f.resid_range[1]}'
        return tag

    @property
    def tica_tag(self) -> str:
        return f'{self.feat_tag}_lag{self.tica.lag}_dim{self.tica.dim}'

    @property
    def cluster_tag(self) -> str:
        return f'{self.tica_tag}_k{self.cluster.n_clusters}'

    @property
    def msm_tag(self) -> str:
        return f'{self.cluster_tag}_msmlag{self.msm.lag}_{self.msm.scheme}'

    def stage_dir(self, name: str) -> str:
        d = os.path.join(self.output_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    # ---- validation ------------------------------------------------------ #
    def validate(self) -> None:
        specs = self.run_specs()
        if not specs:
            raise ValueError('no runs configured (set `runs` or '
                             '`h5_template` + `seed_list`)')
        labels = [s.label for s in specs]
        if len(set(labels)) != len(labels):
            raise ValueError(f'run labels must be unique: {labels}')
        if self.features.kind not in ('ca_distances', 'custom'):
            raise ValueError("features.kind must be 'ca_distances' or 'custom'")
        if self.features.kind == 'custom' and not self.features.custom_function:
            raise ValueError("features.kind='custom' needs features.custom_function")
        if self.tica.lag < 1:
            raise ValueError(f'tica.lag must be >= 1; got {self.tica.lag}')
        if self.msm.lag < 1:
            raise ValueError(f'msm.lag must be >= 1; got {self.msm.lag}')
        if self.msm.scheme not in ('w2', 'w2w1'):
            raise ValueError("msm.scheme must be 'w2' or 'w2w1'")
        if self.msm.frame_weight_mapping not in ('cluster', 'normalized'):
            raise ValueError("msm.frame_weight_mapping must be 'cluster' or "
                             "'normalized'")
        bad = set(self.fe.weightings) - {'msm', 'revo', 'none'}
        if bad:
            raise ValueError(f'unknown fe.weightings {sorted(bad)}; '
                             f"use 'msm', 'revo' or 'none'")
        ref_labels = {r.label for r in self.properties.references}
        names = [o.name for o in self.properties.observables]
        if len(set(names)) != len(names):
            raise ValueError(f'observable names must be unique: {names}')
        for o in self.properties.observables:
            if o.kind == 'ca_rmsd' and o.reference not in ref_labels:
                raise ValueError(f"observable {o.name!r}: reference "
                                 f"{o.reference!r} is not in properties.references "
                                 f'{sorted(ref_labels)}')
            if o.kind == 'custom' and not o.function:
                raise ValueError(f"observable {o.name!r}: kind='custom' needs "
                                 f'`function`')
            if o.kind not in ('ca_rmsd', 'custom'):
                raise ValueError(f"observable {o.name!r}: unknown kind {o.kind!r}")


# --------------------------------------------------------------------------- #
_NESTED = {'features': FeatureParams, 'lineage': LineageParams,
           'tica': TICAParams, 'cluster': ClusterParams,
           'properties': PropertyParams, 'msm': MSMParams, 'fe': FEParams}


def config_from_dict(d: Dict[str, Any]) -> Config:
    d = dict(d)
    kw = {}
    for key, cls in _NESTED.items():
        if key in d:
            sub = dict(d.pop(key))
            if key == 'properties':
                sub['references'] = [ReferenceSpec(**r) for r in sub.get('references', [])]
                sub['observables'] = [ObservableSpec(**o) for o in sub.get('observables', [])]
            if key == 'fe' and 'surfaces' in sub:
                sub['surfaces'] = [tuple(s) for s in sub['surfaces']]
            kw[key] = cls(**sub)
    if 'runs' in d:
        kw['runs'] = [RunSpec(**r) for r in d.pop('runs')]
    kw.update(d)
    return Config(**kw)


def load_config(path: str) -> Config:
    """Load a config from a .py (defining CONFIG), .toml or .json file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'config not found: {path}')
    ext = os.path.splitext(path)[1].lower()
    if ext == '.py':
        spec = importlib.util.spec_from_file_location('wemsm_user_config', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.CONFIG
    elif ext == '.toml':
        try:
            import tomllib
        except ImportError:                                       # Python < 3.11
            try:
                import tomli as tomllib
            except ImportError as exc:
                raise ImportError('TOML configs need Python >= 3.11 or the '
                                  '`tomli` package; or use a .py/.json config') from exc
        with open(path, 'rb') as fh:
            cfg = config_from_dict(tomllib.load(fh))
    elif ext == '.json':
        with open(path) as fh:
            cfg = config_from_dict(json.load(fh))
    else:
        raise ValueError(f'unsupported config type {ext!r}; use .py, .toml or .json')
    cfg.validate()
    return cfg


def apply_override(cfg: Config, assignment: str) -> None:
    """Apply one ``section.key=value`` override, e.g. ``tica.retrain=true``.

    The value is parsed as JSON when possible (true, 3, [1, 2], null) and is
    otherwise taken as a string.
    """
    if '=' not in assignment:
        raise ValueError(f'override must look like section.key=value; got {assignment!r}')
    key, raw = assignment.split('=', 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    obj = cfg
    parts = key.strip().split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f'unknown config field {key!r}')
    setattr(obj, parts[-1], value)


def as_jsonable(obj) -> Any:
    """Dataclasses/tuples -> plain JSON types, for metadata comparison."""
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    return json.loads(json.dumps(obj, default=str))
