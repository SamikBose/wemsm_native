"""
wemsm_native
============

Markov state models from weighted-ensemble (wepy/REVO) simulations:

    wepy HDF5 runs -> features -> tICA (trained on the WE data, lineage lag)
    -> k-means -> per-frame properties -> weighted lineage transition counts
    -> CSN stationary state weights -> reweighted free-energy landscapes

Every per-frame quantity is stored per run as an array indexed
``(cycle, walker, ...)``; runs may have different numbers of cycles and
walkers, so collections of runs are plain lists, never one stacked array.
"""

__version__ = '0.1.0'

from .config import (ClusterParams, Config, FEParams, FeatureParams,
                     LineageParams, MSMParams, ObservableSpec, PropertyParams,
                     ReferenceSpec, RunSpec, TICAParams, load_config)

__all__ = ['Config', 'RunSpec', 'FeatureParams', 'LineageParams', 'TICAParams',
           'ClusterParams', 'PropertyParams', 'ObservableSpec', 'ReferenceSpec',
           'MSMParams', 'FEParams', 'load_config', '__version__']
