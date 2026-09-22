# AggressionObservation — Figures

Analysis and figure-generation notebooks.

This folder contains Jupyter notebooks that reproduce the main figures in the paper. Each notebook loads preprocessed fiber photometry traces and behavior labels (output from the `behavior_classification` pipeline) and performs statistical analyses and plotting. 

---

## Data

All notebooks load from a common multi-session data file:

```
multifiber.pickle
```

This contains fiber photometry traces from 23 neural populations (excitatory and inhibitory) across ~12 brain regions (PrL, vLS, POA, BNST, AH, MeA, VMH, PAG, PMv, LHb, PA, NAc), recorded simultaneously in resident aggressors, social observers, and non-social controls during resident-intruder assays. 

---

## Notebooks

### `Figure1.1_HandscoredBehavior.ipynb`
**Figures 1 (and 5) (hand-scored behavior overlays)**

Computes occupancy for multiple behaviors in the hard fight, both from the recording cohort and the OBS-Gq/OBS-Ctrl cohorts. Labels are derived from [BORIS](https://www.boris.unito.it/). Handles two BORIS CSV export formats and supports multiple behavior categories (resident/intruder unilateral attack, mutual fighting with resistance, flee). Includes:

- Parsing and aligning BORIS annotations to SLEAP frame indices across sessions
- Extracting frame-level binary labels for aggression, resistance, and flight behaviors
- Building matched feature-label datasets for statistical analysis and figure overlays
- Cross-referencing hand-scored labels against classifier predictions

---

### `Figure1.2_UnsupervisedAnalysis.ipynb`
**Figures 1 (and related)**

Characterizes the behavioral repertoire of resident aggressors, observers, and non-social controls using the unsupervised clustering pipeline from `behavior_classification`. Includes:

- Loading UMAP embeddings and watershed cluster labels across experimental groups (aggressor / observer / non-social)
- Quantifying cluster occupancy, trial counts, and persistence per animal per session
- Statistical validation of cluster-level behavioral differences across groups (with significance brackets)
- Decoding experimental group identity from behavior cluster vectors (occupancy/persistence)
- Separate decoding models for observer vs. non-observer and for observer vs. experienced animals
- Transition analysis between behavioral clusters
- JS divergence analysis comparing behavior and transition maps between groups

---

### `Figure2&3_TrainingPeriodAnalysis.ipynb`
**Figures 2 and 3 - PETH comparisons, time shifting and attack-aligned decoding**

Features code wrangling attack-aligned data, time shifting and neural decoding during observation. 

- Implements shift-only time warping to individual group neural activity tensors
- Generates PETHs for EXP, OBS and each control condition following time shifting
- Attention-based filtering of neural activity during attacks
- Features neural decoding analyses during observation involving:
      Resident fast action vs resident slow action attacks
      Intruder fast action vs intruder slow action attacks
      Toy vs live conspecific attacks
      Familiar aggressor vs novel aggressor attacks
- Features neural decoding analyses during hard fight, classifying attack or attacked conditions (Ext 9)

---

### `Figure4_SimilarityAnalysis.ipynb`
**Figure 4 - Cosine distance analyses**

Compares activity maps (vectors containing mean activity per cluster) between groups during training or the hard fight.
Includes:

- Extracting mean activity vectors
- Application of cosine distance to compare maps between groups
- Statistical comparisons with significance brackets, different visualization styles, and stat overlays with multiple comparisons

---

### 'Figure5.1_HardFightNeuralComparisons.ipynb'
**Figure 5 - Mean activity comparisons and LDA modeling**

Compares neural activity aligned to social behaviors (and normalized to asocial behaviors) between groups. 
Includes:

- Extracting mean activity and associated group statistics
- LDA fitting: per-animal mean activity vectors are utilized to classify EXP vs NON; OBS activity is projected to this EXP-NON axis and compared
- Leave-one-population-out analysis detailing how removing one population from the axis above pushes OBS animals in either EXP or NON direction
- PCA fits of EXP, OBS and NON

---

### `Figure5.2_ARDModeling.ipynb`
**Figure 5 — Multi-region linear modeling (ARD)**

Tests how well activity in one neural population can be predicted from the rest of the recorded network, using regularized regression. Includes:

- Formatting multi-region trace data into per-animal, per-epoch arrays across 9 experimental time windows (days 1–9)
- Cluster-by-cluster excitatory vs. inhibitory regression
- Ridge regression with cross-validation for predicting single-region activity from all other regions
- **Automatic Relevance Determination (ARD) regression** — sparse Bayesian linear regression that identifies which brain regions carry unique predictive weight for each target region
- Mixed linear models (via `statsmodels`) for statistical testing of encoding weights across experimental conditions
- Visualization of regression weight matrices across the fitted network

---

### `Figure5.3_GqComparisons.ipynb`
**Figure 5 — Gq-DREADD chemogenetic manipulation analysis**

Examines how Gq-DREADD activation pushes behavior into NON territory:

- Extracting cluster occupancy and transition probabilities for OBS-Gq and OBS-Ctrl animals
- JS divergence between above groups and NON (occupancy and transitions)

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib
seaborn
tqdm
pyarrow
pickle
h5py
```

These are shared with the `behavior_classification` pipeline. No additional installs are required if that environment is already set up.

---

## Notes

- Notebooks expect preprocessed `.parquet` feature files and behavior label dictionaries as output by `behavior_classification/`. Update path variables at the top of each notebook to point to your local data directories.
- Statistical tests used throughout: mixed linear models, repeated-measures ANOVA, permutation tests, and cross-validated decoding. All figures are generated with `matplotlib`/`seaborn` and saved as `.svg` or `.png`.
