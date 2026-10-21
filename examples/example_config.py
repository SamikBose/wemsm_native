"""Example wemsm_native configuration. Copy it, edit the paths, then run

    wemsm-native status --config my_config.py     # what will be reused / built
    wemsm-native all    --config my_config.py
"""

from wemsm_native import (ClusterParams, Config, FEParams, FeatureParams,
                          MSMParams, ObservableSpec, PropertyParams,
                          ReferenceSpec, RunSpec, TICAParams)

CONFIG = Config(
    system_name='GBP',
    output_dir='/ABSOLUTE/PATH/TO/wemsm_out',

    # Runs: a template over seeds ...
    md_root='/ABSOLUTE/PATH/TO/wepy_sims/GBP_1GGG',
    h5_template=('{root}/aSAM_clust{c}/'
                 'run0_walk48_steps50000_cycs250_substep0_md0.025/wepy.results.h5'),
    seed_list=[0, 2, 3, 4, 5, 6, 8],
    run_label_template='clust{c}',
    max_cycles=None,                 # None -> all cycles of each file
    # ... or explicit runs (overrides the template); lengths may differ:
    # runs=[RunSpec('clust2', '/path/a.h5'), RunSpec('clust3', '/path/b.h5', max_cycles=150)],

    topology_pdb=None,               # None -> topology stored in the first HDF5
    unwrap_protein=False,

    features=FeatureParams(kind='ca_distances', cutoff_nm=1.0, min_seq_sep=5),
    tica=TICAParams(lag=1, dim=2),   # lag > 1 warns: unweighted tICA on WE data
    cluster=ClusterParams(n_clusters=100),

    properties=PropertyParams(
        references=[ReferenceSpec('open', '/ABSOLUTE/PATH/TO/open.pdb', chain='A'),
                    ReferenceSpec('closed', '/ABSOLUTE/PATH/TO/closed.pdb', chain='A')],
        observables=[ObservableSpec('rmsd_open', 'ca_rmsd', reference='open'),
                     ObservableSpec('rmsd_closed', 'ca_rmsd', reference='closed')],
        common_core=True, units='angstrom'),

    msm=MSMParams(lag=1, scheme='w2', frame_weight_mapping='cluster'),

    fe=FEParams(surfaces=[('tic1', 'tic2'), ('rmsd_open', 'rmsd_closed')],
                weightings=['msm', 'revo'], bins=100, temperature=300.0),
)
