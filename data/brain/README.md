# Data used in the study of Brain Bicommunities

## Summary

This directory contains brain connectivity data and derivatives used to study directed bicommunities in the directed human connectome.

## Primary Datasets

### Multiscale Atlas of White Matter Connectivity (BundleAtlas)

> Alemán-Gómez Y, Griffa A, Houde JC, Najdenovska E, Magon S, Cuadra MB, et al. A multi-scale probabilistic atlas of the human connectome. Sci Data. 2022 Aug 23;9(1):516. 

**Repository**: https://github.com/connectomicslab/probconnatlas

Probabilistic atlas of white matter connectivity that provides structural connectivity matrices (see [derivative](#structural-connectivity)) along with probabilistic fiber bundle maps and streamline centroids. Used as the structural backbone for the directed connectome.

- `./BundleAtlas/centroids/scale2/lausanne2018.scale2.sym.corrected+aseg_MaxProb.nii` is the volumetric definition of gray matter regions used in the atlas.
- `./BundleAtlas/centroids/scale2/group_centroids_scale2/wm.connatlas.scale2.centroids.h5.gz` is the centroid data. In detail, each edge of the structural connectome is defined by a set of white matter streamlines which are then summarized into 20 centroids.

### Invasive Recordings of Electrical Brain Conduction (F-Tract)

> Trebaul L, Deman P, Tuyisenge V, Jedynak M, Hugues E, Rudrauf D, et al. Probabilistic functional tractography of the human cortex revisited. NeuroImage. 2018 Nov 1;181:414–29.

**Repository**: https://search.kg.ebrains.eu/instances/41db823e-7e1b-44c7-9c69-eaa26e226384

Intracranial sEEG recordings from patients with epilepsy providing ground-truth measurements of electrical signal propagation between brain regions. Used to validate the directionality of grouped connections identified through bicommunity analysis. Contains cortico-cortical evoked potentials (CCEPs) directed brain connectivity.

- `./F-Tract/Lausanne2018-scale2/Lausanne2018-scale2.txt` is the parcel labels and order of the F-Tract dataset.
- `./F-Tract/Lausanne2018-scale2/15_inf/50/probability.txt` is the matrix of F-Tract probability that captures the proportion of patients in which a directed connection exists between regions pairs.
- `./F-Tract/Lausanne2018-scale2/15_inf/50/max_peak_delay_50__zth5/min_value_gen__0` is the directory in which F-Tract features are stored. Each feature has a specific directory  (e.g., `feature_ampl_zth5`) in which several files can be found. Here are those that are used in the analyses:
    - `N_with_values.txt.gz`: Number of entries (meansurements)
    - `nanquantile_0.5.txt.gz`: Median of the measured value for each pairs of regions (connectivity matrix)
    - Features in `implantation_name` allow to assess the robustness of measurements by considering, for example, the total number of measurement for a feature (`N_with_values`) or the number of unique implants for a connection (ratio between `N_with_values` and `count_unique_str`).

### Segmented Anatomical White Matter Fiber Bundles (SCIL)

> St-Onge E, Schilling KG, Rheault F. BundleSeg: A Versatile, Reliable and Reproducible Approach to White Matter Bundle Segmentation. In: Karaman M, Mito R, Powell E, Rheault F, Winzeck S, editors. Computational Diffusion MRI. Cham: Springer Nature Switzerland; 2023. p. 47–57.

**Repository**: https://github.com/scilus/rbx_flow

The RecoBundles X (RBX) or SCIL atlas containing 51 anatomically-defined white matter fiber bundles and provides anatomical validation for individual white matter tracts identifies as bicommunities.

- `./SCIL/bundle_names.csv` is the name and ids of segmented anatomical bundles.
- `./SCIL/atlas/pop_average` is the directory in which individual fiber bundles are found in separated `.trk` files (with labels such as `AF_L.trk` for the left Arcuate Fasciculus).
- `./SCIL/centroids` is the directory in which streamlines centroids are found in separated `.trk` files (with labels such as `AF_L_centroid.trk` for the left Arcuate Fasciculus).

## Derivatives

### Structural Connectivity

Group-averaged structural connectivity matrices derived from BundleAtlas. In `./derivatives/structural_connectome` there are several features for each scale of the Lausanne 2018 atlas:
- `Laus2018_brain_labels-scale<SCALE>.csv`: label and ordering of the gray matter parcels.
- `Laus2018_roi_centers-ftract-scale<SCALE>.pkl`: spatial location of the center of each gray matter parcel of the atlas in MNI coordinate space.
- `Laus2018_bundle_probability_atlas-scale<SCALE>.pkl`: structural connectome in which connections are modeled as the proportion of participants (out of $N=66$) with at least one streamline between region pairs. This is used for thresholding the structural connectome at a level of 0.5 (half of the participants).
- `Laus2018_bundle_streamlines_atlas-scale<SCALE>.pkl`: structural connectome in which connections are modeled as the numbe of streamlines between region pairs. This is used for thresholding the structural connectome by considering connections with least 5 streamlines.

### rDCM (Regression Dynamic Causal Modeling)

> Paquola C, Garber M, Frässle S, Royer J, Zhou Y, Tavakol S, et al. The architecture of the human default mode network explored through cytoarchitecture, wiring and signal flow. Nat Neurosci. 2025 Mar;28(3):654–64.

**Repository**: https://zenodo.org/records/14034721

Effective connectivity estimates computed using regression dynamic causal modeling on resting-state fMRI data. Provides directionality to structural connectivity from an estimated edge-wise asymmetry of connection at the whole-brain scale.

In `./derivatives/atlas_correspondance/rDCM`, rDCM effective connectivity matrices can be found. Note that these matrices where originally computed with the Schaeffer400 parcellation (with 14 subcortical structures) and have been mapped to the Lausanne2018 atlas.

### Brain Lobes and Yeo 7, 17 Networks Matching

Mapping between anatomical parcellations (Lausanne atlas, scale 2) and functional networks from the Yeo 7 and 17 network as well as the main brain lobes. Used to interpret bicommunities in terms of established anatomical lobes (frontal, parietal, temporal, occipital) and functional networks (visual, somatomotor, attention, default mode, etc.).

The assignment of atlas parcels to brain lobes and Yeo2011 resting state networks can be found in `./derivatives/atlas_correspondance/Lobes` and `./derivatives/atlas_correspondance/YeoNetworks` respectively. The `.pkl` files contain the matrix of correspondence from the lobe/network definition to the nodal definition (size $N_{Network}\times N_{Parcels}$) where each entry $A_{ij}$ represents the normalized volumetric overlap between network $i$ and region $j$.

### Results of Consensus Clustering

Consensus clustering results across multiple runs (n=50 trials) of the bicommunity detection algorithm at varying k values (10-80 bicommunities). The consensus matrices (of size $N_{Edges} \times N_{Edges}$) capture the proportion of k-means runs in which two edges were found in a similar cluster. The filename `brain_consensus-EC_scale2_nvec20_trials50_ninit50_kmeans10-79_slines_thresh5____.pkl` captures the parameters of that specific run:
- `scale2`: Lausanne2018 atlas scale is `2`,
- `nvec20`: `20` first components of the SVD are included in the clustering,
- `trials50`: K-means is repeated over `50` random initializations,
- `kmeans10-79`: K-means is repeated for $K$ number of clusters going from `10` to `79`,
- `slines_thresh5`: the undirected connectome is thresholded to consider only connections with at least `5` streamlines.

### Results of Permutation Testing

Statistical validation through permutation testing by either:
- Randomly swapping the edge weights with the opposite direction to remove any structured directionality. This is used for non-parametric testing of significantly asymmetric bicommunities. The filename `Dir-permutations_scale2_gamma1-10000Perm-K13_RatioOnly.pkl` captures the parameters of a permuation setting in which:
    - `scale2`: Lausanne2018 atlas scale is `2`,
    - `gamma1`: non-linear transformation to the graph asymmetry by element-wise power by a value $\gamma=1$ (no non-linearity),
    - `10000Perm`: number of permutations is `10000`,
    - `K13`: testing has been made for $K=13$ bicommunities,

- Reshuffling the edge-to-cluster assignment to generate random bicommunity structure. This is used for non-parametric testing of the meaningful aggregation of edge asymmetry at the level of true against random bicommunities. The filename `permutations_scale2_gamma1-F_meas49_impl2-4999Perm-Abs-K3.pkl` captures the parameters of the permuation setting in which:
    - `scale2`: Lausanne2018 atlas scale is `2`,
    - `gamma1`: non-linear transformation to the graph asymmetry by element-wise power by a value $\gamma=1$ (no non-linearity),
    - `F_meas49_impl2`: F-Tract thresholding parameters considering strictly more than `49` measurements in strictly more than `2` implants,
    - `4999Perm`: number of permutations is `4999`,
    - `Abs`: considers the absolute value of correlations thus allowing negative correlation,
    - `K3`: permutations have been computed for the `3` highest local maxima of cluster stability,


The permuted data and results are saved to keep a fixed state of the figures and results. While subtle changes may occur with re-computation, the results and significance are proven to remain robust.