# Bimodularity Framework

Code to replicate analyses and figures of [Cionca et al. 2025 (Preprint)](https://arxiv.org/abs/2502.04777).

## Requirements

### Data

*C. elegans* data can be accessed on [WormAtlas.org](https://www.wormatlas.org/neuronalwiring.html#NeuronalconnectivityII) under "Neuronal Connectivity II: by L.R. Varshney, B.L. Chen, E. Paniagua, D.H. Hall and D.B. Chklovskii" (see [Varshney et al., 2011](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066) for more information).

### Python 3.11.8

The environment can be easily installed using `pip install -r requirements.txt`, or alternatively with a new `conda` environment:
```shell
conda create -n bimodularity python==3.11.8
conda activate bimodularity
pip install -r requirements.txt
```
Note that compatiblity is ensured for **python 3.11.8** - you may have to play with dependencies when using other versions.

## Usage

The `Bimodularity-Figures.ipynb` notebook:
- generates the canonical graphs,
- loads the *C. elegans* data,
- computes the bimodularity analyses,
- and creates all the figures in the article.

The `Bimodularity-Supplementary.ipynb` notebook:
- creates all the supplementary figures in the article.

Both notebooks are self-sufficient and should run when using the `run all` command.