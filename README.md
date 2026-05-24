# AggressionObservation

Code repository for:

> **Aggression experience and observation promote shared behavioral and neural changes** Jorge M. Iravedra-García, Eartha Mae Guthman, Dexter Tsin, Polina Cherepanova, Lenca Cuture, Edgar J. Ocasio-Arce, Jonathan W. Pillow1 Annegret L. Falkner
> bioRxiv preprint: (https://www.biorxiv.org/content/10.1101/2024.12.26.630396v1)

This repository contains the full analysis pipeline supporting the figures in the paper. It is organized into two subfolders:

---

## Repository Structure

```
AggressionObservation/
├── behavior_classification/     # Pose-based feature extraction and behavior segmentation
│   ├── behavior_feature_extraction.py
│   ├── behavior_feature_post-processing.py
│   ├── unsupervised_classifier.py
│   └── supervised_classifier.ipynb
│
└── figures/                     # Statistical analyses and figure generation
    ├── unsupervised_supervised_behavior_analysis__figure1_.ipynb
    ├── handscored_behavior_analysis__figures1_5_.ipynb
    ├── ARD_linearModeling__figure5_.ipynb
    └── Gq-DREADD-unsupervised_analysis__figure5_.ipynb
```

---

## Overview

We recorded simultaneous fiber photometry signals from excitatory, inhibitory and dopaminergic neural populations across a conserved neural network in mice performing, observing, or not observing aggressive social encounters. This repository provides code for:

1. **Behavior classification** — extracting kinematic, postural, and social features from SLEAP pose estimates and segmenting behavior via unsupervised clustering (UMAP + watershed) and supervised classification (gradient-boosted classifier trained on BORIS annotations)

2. **Figure generation** — statistical analyses of behavioral repertoires across experimental groups, signal processing, neural decoding, multi-region linear encoding models, and chemogenetic manipulation experiments

See the README in each subfolder for detailed documentation.

---

## Data

Processed data files required to run the analysis notebooks will be made available in the future.

Raw video recordings and SLEAP pose estimates are available upon reasonable request.

---

## Requirements

All code is written in Python 3. Core dependencies:

```
numpy
pandas
scipy
scikit-learn
statsmodels
umap-learn
matplotlib
seaborn
h5py
joblib
tqdm
pyarrow
sleap
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn statsmodels umap-learn matplotlib seaborn h5py joblib tqdm pyarrow
```

SLEAP installation: https://sleap.ai


---

## Contact

Falkner Lab, Princeton Neuroscience Institute
