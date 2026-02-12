# Bimodularity Framework

Main code for the analysis of directed graph communities, or *bicommunities*.

If you use parts of this code and framework, please cite the following article:

> A. Cionca, C.H.M. Chan, & D. Van De Ville, Community detection for directed networks revisited using bimodularity, Proc. Natl. Acad. Sci. U.S.A. 122 (35) e2500571122, [https://doi.org/10.1073/pnas.2500571122](https://doi.org/10.1073/pnas.2500571122) (2025).

## Requirements

### Data

#### *C. elegans*

*C. elegans* data can be accessed on [WormAtlas.org](https://www.wormatlas.org/neuronalwiring.html#NeuronalconnectivityII) under "Neuronal Connectivity II: by L.R. Varshney, B.L. Chen, E. Paniagua, D.H. Hall and D.B. Chklovskii" (see [Varshney et al., 2011](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066) for more information).
All data used are summarized in `./data/celegans`.

#### Human

All data used in the analysis of bicommunities of the directed human connectome are present in `./data/brain` (see the [brain data directory](./data/brain) for further details).

### Python 3.11.8

The environment can be easily installed using `pip install -r requirements.txt`, or alternatively with a new `conda` environment:
```shell
conda create -n bimodularity python==3.11.8
conda activate bimodularity
pip install -r requirements.txt
```
Note that compatiblity is ensured for **python 3.11.8** - you may have to play with dependencies when using other versions.

## Usage

All the code is compiled within the main `bimodularity` module which include:
- `dgsp` for processing and analyses,
- `data_load` for handling the datasets,
- `bimod_plot` for visualization,
- `bundle` for utilities specific to the analysis of white matter fiber bundles.

### Replication of [Cionca, et al., 2025](https://doi.org/10.1073/pnas.2500571122)

In `./notebooks/01-bimodularity`, you may find the main figure (`Bimodularity-Figures.ipynb`) and supplementary (`Bimodularity-Supplementary.ipynb`) notebooks to reproduce the analyses and figures of the article.

The `Bimodularity-Figures.ipynb` notebook:
- generates the canonical graphs,
- loads the *C. elegans* data,
- computes the bimodularity analyses,
- and creates all the figures in the article.

The `Bimodularity-Supplementary.ipynb` notebook:
- creates all the supplementary figures in the article.

### Bicommunities of the Directed Connectome

In `./notebooks/02-brain_bicommunities`, you may find the main figure (`brainbicom-Figures.ipynb`) and supplementary (`brainbicom-Supplementary.ipynb`) notebooks to produce the analyses and figures of bicommunities of the human brain.

The `brainbicom-ConsensusClustering.ipynb` notebook:
- Computes the consensus matrix for specific settings of the bicommunity detection and for various definition of directed connectomes (later saved in `./data/brain/derivatives/consensus_clustering`). Note that you **NEED** to first run this notebook to generate the consensus data (if not already present) before running the figures notebooks.

The `brainbicom-Figures.ipynb` notebook:
- loads consensus data,
- computes the analyses of bicommunity structure,
- and creates all the visualizations.

The `brainbicom-Supplementary.ipynb` notebook:
- creates additional figures.

All notebooks are self-sufficient and should run when using the `run all` command.