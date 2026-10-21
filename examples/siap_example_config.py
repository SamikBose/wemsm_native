"""Runnable example on the bundled SiaP data (examples/data).

Three 10-cycle, 48-walker REVO runs of SiaP (aSAM cluster 2; merge distances
2, 5 and 10), solvent stripped by make_example_data.py. Two runs are capped at
fewer cycles to show runs of different lengths.

    wemsm-native all --config examples/siap_example_config.py
"""

import os

from wemsm_native import (ClusterParams, Config, FEParams, FeatureParams,
                          MSMParams, ObservableSpec, PropertyParams,
                          ReferenceSpec, RunSpec, TICAParams)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

CONFIG = Config(
    system_name='SiaP_example',
    output_dir=os.path.join(HERE, 'siap_example_out'),
    runs=[RunSpec('nmerge2', os.path.join(DATA, 'SiaP_clust2_nmerge2.h5')),
          RunSpec('nmerge5', os.path.join(DATA, 'SiaP_clust2_nmerge5.h5'), max_cycles=7),
          RunSpec('nmerge10', os.path.join(DATA, 'SiaP_clust2_nmerge10.h5'), max_cycles=5)],
    topology_pdb=None,                       # topology stored in the HDF5 files
    features=FeatureParams(cutoff_nm=1.0, min_seq_sep=5),
    tica=TICAParams(lag=1, dim=2),
    cluster=ClusterParams(n_clusters=20),
    properties=PropertyParams(
        references=[ReferenceSpec('start', os.path.join(DATA, 'SiaP_clust2_start.pdb'),
                                  chain=0)],
        observables=[ObservableSpec('rmsd_start', 'ca_rmsd', reference='start')]),
    msm=MSMParams(lag=1, scheme='w2', frame_weight_mapping='cluster'),
    fe=FEParams(surfaces=[('tic1', 'tic2'), ('tic1', 'rmsd_start')],
                weightings=['msm', 'revo'], bins=40),
)
