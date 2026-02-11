# Data used in the study of Brain Bicommunities

## Summary

This directory contains brain connectivity data and derivatives used to study directed bicommunities in the directed human connectome.

## Primary Datasets

### Multiscale Atlas of White Matter Connectivity (BundleAtlas)

> Alemán-Gómez Y, Griffa A, Houde JC, Najdenovska E, Magon S, Cuadra MB, et al. A multi-scale probabilistic atlas of the human connectome. Sci Data. 2022 Aug 23;9(1):516. 

**Repository**: https://github.com/connectomicslab/probconnatlas

Probabilistic atlas of white matter connectivity that provides structural connectivity matrices (see [derivative](#structural-connectivity)) along with probabilistic fiber bundle maps and streamline centroids. Used as the structural backbone for the directed connectome.

### Invasive Recordings of Electrical Brain Conduction (F-Tract)

> Trebaul L, Deman P, Tuyisenge V, Jedynak M, Hugues E, Rudrauf D, et al. Probabilistic functional tractography of the human cortex revisited. NeuroImage. 2018 Nov 1;181:414–29.

**Repository**: https://search.kg.ebrains.eu/instances/41db823e-7e1b-44c7-9c69-eaa26e226384

Intracranial sEEG recordings from patients with epilepsy providing ground-truth measurements of electrical signal propagation between brain regions. Used to validate the directionality of grouped connections identified through bicommunity analysis. Contains cortico-cortical evoked potentials (CCEPs) directed brain connectivity.

### Segmented Anatomical White Matter Fiber Bundles (SCIL)

> St-Onge E, Schilling KG, Rheault F. BundleSeg: A Versatile, Reliable and Reproducible Approach to White Matter Bundle Segmentation. In: Karaman M, Mito R, Powell E, Rheault F, Winzeck S, editors. Computational Diffusion MRI. Cham: Springer Nature Switzerland; 2023. p. 47–57.

**Repository**: https://github.com/scilus/rbx_flow

The RecoBundles X (RBX) or SCIL atlas containing 51 anatomically-defined white matter fiber bundles and provides anatomical validation for individual white matter tracts identifies as bicommunities.

## Derivatives

### Structural Connectivity

Group-averaged structural connectivity matrices derived from BundleAtlas.

### rDCM (Regression Dynamic Causal Modeling)

> Paquola C, Garber M, Frässle S, Royer J, Zhou Y, Tavakol S, et al. The architecture of the human default mode network explored through cytoarchitecture, wiring and signal flow. Nat Neurosci. 2025 Mar;28(3):654–64.

**Repository**: https://zenodo.org/records/14034721

Effective connectivity estimates computed using regression dynamic causal modeling on resting-state fMRI data. Provides directionality to structural connectivity from an estimated edge-wise asymmetry of connection at the whole-brain scale.

### Brain Lobes and Yeo 7, 17 Networks Matching

Mapping between anatomical parcellations (Lausanne atlas, scale 2) and functional networks from the Yeo 7 and 17 network as well as the main brain lobes. Used to interpret bicommunities in terms of established anatomical lobes (frontal, parietal, temporal, occipital) and functional networks (visual, somatomotor, attention, default mode, etc.).

### Results of Consensus Clustering

Consensus clustering results across multiple runs (n=50 trials) of the bicommunity detection algorithm at varying k values (10-80 bicommunities). Includes the consensus matrices.

### Results of Permutation Testing

Statistical validation through permutation testing by either:
- Randomly swapping the edge weights with the opposite direction to remove any structured directionality (`DIR-` prefix). This is used for non-parametric testing of significantly asymmetric bicommunities.
- Reshuffling the edge-to-cluster assignment to generate random bicommunity structure. This is used for non-parametric testing of the meaningful aggregation of edge asymmetry at the level of true against random bicommunities.

The permuted data and results are saved to keep a fixed state of the figures and results. While subtle changes may occur with re-computation, the results and significance are proven to remain robust.