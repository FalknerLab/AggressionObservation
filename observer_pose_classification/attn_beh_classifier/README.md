# Supervised video-based classification of observer behavior

Last updated: 2026/04/17

This folder contains minimal implementation of a behavior classification model using video frames as inputs. The model has a backbone of 3D CNN with spatiotemporal attention and a MLP classifier.

---

## Model Overview

The model processes 20-frame video clips (0.5 s at 40 fps) and outputs a probability distribution over seven behavior classes: attention, grooming, scratching, still, investigate, rearing, and turning.

**Encoder** (`models/spatial_attention.py`, `models/action_recognition.py`): A four-layer 3D CNN. Each layer applies 3D convolutions (3×3×3 kernel), batch normalization, ReLU, max pooling (2×2×2), and a spatiotemporal attention module (CBAM-style, adapted for video). Channel dimensions double at each layer (64 → 128 → 256 → 512); final features are 512-d global average pooled vectors.

**Spatiotemporal Attention** (`models/spatial_attention.py`): Applied after each convolutional block. Channel attention uses adaptive average pooling through a bottleneck (reduction ratio 8) to weight informative channels. Spatial attention applies a 3D convolution to produce a per-location importance map. Both stages use sigmoid gating and are learned end-to-end.

**Classifier** (`models/action_recognition.py`): Three-layer MLP (512 → 256 → 128 → 7) with dropout (p = 0.3) between layers. Optionally replaceable with a cross-attention head (`AttentiveClassifier`).

**Training** (`train_action_recognition.py`, `utils/training_utils.py`): AdamW with dual learning rates — classifier fixed at 3×10⁻⁴, encoder warmed up from 7×10⁻⁵ to 3×10⁻⁴ over 20 epochs (exponential schedule), then cosine-decayed. 300 epochs, effective batch size 256 (128 × 2 accumulation steps), cross-entropy with label smoothing (α = 0.1), gradient clipping (max norm 1.0), mixed-precision (FP16).

**Augmentation** (`data/augmentations.py`, `data/sampling.py`): Each training clip is randomly assigned one of four strategies — spatial only (rotation ±45°, random erase 2–33%, crop; p = 0.3), temporal only (frame reversal; p = 0.3), spatial+temporal (p = 0.3), or none (p = 0.1). Validation and inference use no augmentation.

**Data splitting** (`utils/data_utils.py`): Stratified 80/20 train/validation split by segment or by video session (video-based split is harder; use `--split_by_video`). Supports K-fold cross-validation via `--fold_index` / `--n_folds`.

---

## Code Structure

```
attn_beh_classifier/
├── train_action_recognition.py   # Entry point: argument parsing, training loop
├── models/
│   ├── spatial_attention.py      # SpatialAttention module + VideoEncoder backbone
│   └── action_recognition.py    # ActionRecognitionModel (encoder + classifier head)
├── datasets/
│   └── action_recognition_dataset.py  # HDF5-backed dataset, frame sampling, balancing
├── data/
│   ├── augmentations.py          # VideoAugmentation (rotation, erase, flip, crop)
│   └── sampling.py               # AugmentationSampler (probabilistic strategy selection)
└── utils/
    ├── data_utils.py             # Segment loading, train/val splits, K-fold
    ├── training_utils.py         # ProgressiveLRScheduler, train_epoch, evaluate_epoch
    ├── output_utils.py           # Logging, checkpoint saving, training curve plots
    └── metrics_utils.py          # Per-class precision/recall/F1, class name mapping
```

---

## Data Format

The pipeline expects frames pre-extracted into per-session HDF5 files. Each HDF5 file stores one recording session and contains a `frames` group with one dataset per segment:

```
{session_id}.h5
└── frames/
    ├── segment_0    # shape (T, H, W, C), dtype uint8
    ├── segment_1
    └── ...
```

A global metadata pickle (`global_metadata.pkl`) lists all segments with their session ID, segment index, and integer behavior label (1-indexed; the loader converts to 0-indexed automatically):

```python
{
  "version": ...,
  "segments": [
      {"session_id": "m01_cam1", "segment_idx": 0, "label": 3},
      ...
  ]
}
```

### Expected layout

```
data/
├── cache/
│   └── frames_classifier_v5/
│       ├── global_metadata.pkl
│       ├── m01_cam1.h5
│       ├── m01_cam2.h5
│       └── ...
```

### Behavior classes (0-indexed)

| Index | Label       | Description |
|-------|-------------|-------------|
| 0     | grooming    | Rhythmic front-limb movements toward head/ears |
| 1     | investigate | Active locomotion or head movements, not wall-directed |
| 2     | rearing     | Upright posture, front limbs on chamber walls |
| 3     | scratching  | Rhythmic hind-limb movements toward rear of body |
| 4     | sniffing    | Close proximity to perforated wall, head oriented toward barrier |
| 5     | still       | Stationary idle posture |
| 6     | turning     | Fast reorientation of body heading |

---

## Environment

Tested on Python 3.10, CUDA 11.8, PyTorch 2.1 (A100 80 GB).

```bash
module purge && module load anaconda3/2023.9
conda activate samv2_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install h5py scikit-learn opencv-python tqdm pandas matplotlib
```

---

## Training

```bash
cd for_submission
python train_action_recognition.py \
    --cache_dir /path/to/data/cache/frames_classifier_v5 \
    --annotation_file /path/to/data/cache/frames_classifier_v5/global_metadata.pkl \
    --disable_pretrained \
    --batch_size 128 --num_epochs 300 \
    --classifier_lr 3e-4 --encoder_start_lr 7e-5 --encoder_target_lr 3e-4 \
    --warmup_epochs 20 --lr_schedule exponential \
    --dropout_rate 0.3 --weight_decay 1e-4 \
    --use_augmentation --balance_classes --label_smoothing 0.1 \
    --clip_length 20 --num_workers 24 \
    --output_dir outputs/run_01 \
    --use_amp --gradient_accumulation_steps 2 --max_grad_norm 1.0 \
    --persistent_workers --prefetch_factor 3 --compile_model
```

Outputs saved to `--output_dir`: `best_model.pt`, `final checkpoint`, `training_metrics.csv`, `training_curves.png`, and per-epoch validation CSVs (if `--save_val_outputs`).
