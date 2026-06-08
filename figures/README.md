# AggressionObservation — Figures

Analysis and figure-generation notebooks.

This folder contains Jupyter notebooks that reproduce the main figures in the paper. Each notebook loads preprocessed fiber photometry traces and behavior labels (output from the `behavior_classification` pipeline) and performs statistical analyses and plotting. **This folder is under active development** — additional notebooks will be added as the manuscript is finalized.

---

## Data

All notebooks load from a common multi-session data file:

```
multifiber_data_reviews.pickle
```

This contains fiber photometry traces from 23 neural populations (excitatory and inhibitory) across ~12 brain regions (PrL, vLS, POA, BNST, AH, MeA, VMH, PAG, PMv, LHb, PA, NAc), recorded simultaneously in resident aggressors, social observers, and non-social controls during resident-intruder assays. 

---

## Notebooks

### `unsupervised_supervised_behavior_analysis__figure1_.ipynb`
**Figures 1 (and related)**

Characterizes the behavioral repertoire of resident aggressors, observers, and non-social controls using the unsupervised clustering pipeline from `behavior_classification`. Includes:

- Loading UMAP embeddings and watershed cluster labels across experimental groups (aggressor / observer / non-social)
- Quantifying cluster occupancy, trial counts, and persistence per animal per session
- Statistical validation of cluster-level behavioral differences across groups (with significance brackets)
- Decoding experimental group identity from behavior cluster vectors (occupancy/persistence)
- Separate decoding models for observer vs. non-observer and for observer vs. experienced animals
- Transition analysis between behavioral clusters
- Feature importance analysis of the embedding
- Correlation of observer behavior clusters with resident fight outcome metrics

---

### `handscored_behavior_analysis__figures1_5_.ipynb`
**Figures 1 and 5 (hand-scored behavior overlays)**

Complements the unsupervised analysis with human-annotated behavioral labels from [BORIS](https://www.boris.unito.it/). Handles two BORIS CSV export formats and supports multiple behavior categories (resident/intruder unilateral attack, mutual fighting with resistance, flee). Includes:

- Parsing and aligning BORIS annotations to SLEAP frame indices across sessions
- Extracting frame-level binary labels for aggression, resistance, and flight behaviors
- Building matched feature-label datasets for statistical analysis and figure overlays
- Cross-referencing hand-scored labels against classifier predictions

---

### `Gq-DREADD-unsupervised_analysis__figure5_.ipynb`
**Figure 5 — Gq-DREADD chemogenetic manipulation analysis**

Examines how Gq-DREADD activation alters behavior cluster usage, comparing vehicle vs. CNO conditions. Includes:

- Extracting cluster occupancy, trial counts, and bout persistence for each cluster under each condition
- Statistical comparisons with significance brackets
- Transition analysis between behavior clusters under chemogenetic manipulation
- Comparing UMAP embedding structure between vehicle and CNO sessions (embedding similarity analysis)
- Visualization of cluster-level behavioral changes as 3×2 panel plots

---

### `Similarity_Analyses_(Figure4).ipynb`
** Figure 4 - Cosine distance analyses**

Compares activity maps (vectors containing mean activity per cluster) between groups during training or the hard fight.
Includes:

- Extracting mean activity vectors
- Application of cosine distance to compare maps between groups
- Statistical comparisons with significance brackets, different visualization styles, and stat overlays with multiple comparisons

---

### `Time-shifting_PETHs_ATTN_&_Decoding_(Figures2&3).ipynb`
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

### `ARD_linearModeling__figure5_.ipynb`
**Figure 5 — Multi-region linear modeling (ARD)**

Tests how well activity in one neural population can be predicted from the rest of the recorded network, using regularized regression. Includes:

- Formatting multi-region trace data into per-animal, per-epoch arrays across 9 experimental time windows (days 1–9)
- Cluster-by-cluster excitatory vs. inhibitory regression
- Ridge regression with cross-validation for predicting single-region activity from all other regions
- **Automatic Relevance Determination (ARD) regression** — sparse Bayesian linear regression that identifies which brain regions carry unique predictive weight for each target region
- Mixed linear models (via `statsmodels`) for statistical testing of encoding weights across experimental conditions
- Visualization of regression weight matrices across the 23-region network

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
- This folder is **under active development**. Notebooks covering additional figures will be added prior to final publication.
