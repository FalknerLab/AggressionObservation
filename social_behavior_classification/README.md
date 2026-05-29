# AggressionObservation — Social Behavior Classification

This folder contains the full pipeline for extracting kinematic and social features from SLEAP pose estimates and classifying aggressive and investigative behaviors in resident-intruder mouse assays.

---

## Overview

Behavior classification is performed in three stages:

1. **Feature extraction** — raw pose coordinates from SLEAP are used to compute kinematic, postural, and social features for each frame
2. **Feature post-processing** — features are smoothed, outlier-corrected, and z-scored across the dataset for downstream use
3. **Behavior segmentation** — either unsupervised (UMAP/t-SNE + watershed clustering) or supervised (gradient-boosted classifier trained on human-annotated labels via BORIS)

---

## Files

| File | Description |
|------|-------------|
| `behavior_feature_extraction.py` | Extracts per-frame kinematic and social features from SLEAP `.pickle` files |
| `behavior_feature_post-processing.py` | Smooths, z-scores, and saves processed features across sessions |
| `unsupervised_classifier.py` | `BehaviorClustering` class: UMAP/t-SNE dimensionality reduction, MLP embedding, and watershed-based behavior clustering |
| `supervised_classifier.ipynb` | Gradient-boosted classifier trained on BORIS-annotated attack and investigation frames; includes training, evaluation, and batch prediction |

---

## Pipeline Details

### 1. Feature Extraction (`behavior_feature_extraction.py`)

Takes SLEAP-tracked `.pickle` files as input (one per session) and computes a large feature set per frame for both the resident and intruder mouse. Features fall into five categories:

- **Inter-node distances** — e.g., tailbase-to-head, forepaw-to-trunk, hindpaw spread
- **Body orientation angles** — e.g., tailbase→neck→head, centroid→head→nose
- **Locomotion** — centroid, head, and tail displacement at 7 temporal scales (50 ms – 5000 ms)
- **Social features** — inter-mouse proximity, bounding box IoU, head/nose orientation toward the other mouse, head-to-head and head-to-tailbase distances
- **Temporal lag features** — mean, median, and sum of each feature across a ±2 s lag window

Output: per-session `_raw_features.pickle` files.

**Key dependencies:** `SLEAP`, `numpy`, `pandas`, `scipy`, `shapely`, `planar`, `opencv-python`

---

### 2. Feature Post-Processing (`behavior_feature_post-processing.py`)

Loads raw feature files, applies dataset-wide collated z-scoring (using mean and std computed across all sessions), and optionally applies Gaussian smoothing. Distance and displacement features are converted from pixels to centimeters (`px2cm = 25.0142`).

Output: per-session `_zscored.parquet` files.

---

### 3. Unsupervised Classification (`unsupervised_classifier.py`)

The `BehaviorClustering` class implements a full unsupervised pipeline:

1. **Uniform sampling** — frames are sampled evenly across sessions to build a balanced training set; attack frames can be additionally oversampled for representation
2. **Dimensionality reduction** — UMAP (default) or t-SNE reduces the selected feature set to 2D
3. **Embedding generalization** — a `MLPRegressor` (optimized via grid search over architecture and solver) learns to map raw features to the 2D embedding space, enabling projection of held-out sessions
4. **Watershed clustering** — a 2D density map of the embedding is computed, and watershed segmentation over local maxima produces discrete behavior cluster labels
5. **Label assignment** — every frame in every session is assigned a cluster label based on its position in the embedding

Multiple feature list variants (`feats_list1`–`feats_list5`) encode different combinations of distance, orientation, speed, IoU, and lag features and can be run in parallel.

---

### 4. Supervised Classification (`supervised_classifier.ipynb`)

Trains a gradient-boosted classifier to detect aggression frames using manual annotations exported from [BORIS](https://www.boris.unito.it/). Steps include:

1. Parsing BORIS `.csv` annotation files and aligning them to SLEAP frame indices
2. Building balanced training sets with temporal resampling to address class imbalance
3. Training and cross-validating on annotated sessions
4. Evaluating on a held-out set using balanced accuracy, F1, and a custom **attack detection rate** metric (fraction of discrete attack bouts detected, with tolerance for detection latency)
5. Batch-predicting attack labels across all sessions and saving results as a dictionary

---

## Requirements

```
numpy
pandas
scipy
scikit-learn
umap-learn
matplotlib
seaborn
opencv-python
shapely
planar
h5py
joblib
tqdm
pyarrow       # for .parquet I/O
sleap         # for pose tracking (upstream of this pipeline)
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn umap-learn matplotlib seaborn opencv-python shapely planar h5py joblib tqdm pyarrow
```

SLEAP installation: https://sleap.ai

---

## Usage

Feature extraction and unsupervised clustering are designed to run on a compute cluster with one job per session/feature set (via `INPUT_DATA_FILE` environment variable):

```bash
# Feature extraction
INPUT_DATA_FILE=/path/to/session.pickle python behavior_feature_extraction.py

# Unsupervised clustering (one job per feature list variant)
INPUT_DATA_FILE=Feats1 python unsupervised_classifier.py
```

The supervised classifier notebook (`supervised_classifier.ipynb`) is designed for interactive use — update the path variables at the top of the notebook to point to your local feature and annotation directories.

---

## Data

Processed feature files and trained classifier weights will be made available in the future.
