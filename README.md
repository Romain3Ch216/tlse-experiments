# Self-supervision for spectral representation learning 
## Experiments on the Toulouse Hyperspectral Data Set

This repository contains the code to reproduce the experiments of the paper:

[R. Thoreau, L. Risser, V. Achard, B. Berthelot and X. Briottet, "Toulouse Hyperspectral Data Set: a benchmark data set to assess semi-supervised spectral representation learning and pixel-wise classification techniques", 2023.](https://arxiv.org/pdf/2311.08863.pdf)

Please cite this paper if you use the code in this repository as part of a published research project:

> @misc{thoreau2023toulouse,
title={Toulouse Hyperspectral Data Set: a benchmark data set to assess semi-supervised spectral representation learning and pixel-wise classification techniques},
author={Romain Thoreau and Laurent Risser and Véronique Achard and Béatrice Berthelot and Xavier Briottet},
year={2023},
eprint={2311.08863},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

## The Toulouse Hyperspectral Data Set

The [Toulouse Hyperspectral Data Set](https://www.toulouse-hyperspectral-data-set.com/) is the combination of 1) an airborne hyperspectral image acquired by the AisaFENIX sensor over Toulouse, France, during the [CAMCATT-AI4GEO campaign](https://www.sciencedirect.com/science/article/pii/S2352340923002287) and of 2) a land cover ground truth, provided with standard train / test splits for the validation of machine learning models.

## Setup
The code was tested with python 3.8:

1. create a python [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
2. clone this repo: ```git clone https://github.com/Romain3Ch216/tlse-experiments.git```
3. navigate to the repository: ```cd tlse-experiments```
4. install python requirements: ```pip install -r requirements.txt```

## Usage

The `main.py` script allows to train a standard Autoencoder or a [Masked Autoencoder](https://github.com/facebookresearch/mae) (and to perform a random search of the best hyperparameters). To validate the potential of the learned spectral representations, the `k_neighbours.py` and `random_forest.py` scripts allow to train a KNN and a RF on top of the frozen features, respectively.

## Checkpoints

Pre-trained models and training logs are available in the `checkpoints` folder.

## Feedback

Please send any feedback to romain.thoreau@cnes.fr

