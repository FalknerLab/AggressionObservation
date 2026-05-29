"""Data utilities for action recognition training.

Contains functions for loading, splitting, and preparing datasets
for training with support for video-based splits and K-fold cross validation.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_class_distribution(segments: List[Dict]) -> Dict:
    """Get class distribution from segments (assumes 0-indexed integer labels)."""
    dist: Dict = {}
    for s in segments:
        label = s.get("label")
        dist[label] = dist.get(label, 0) + 1
    return dist


def load_and_split_segments(
    cache_dir: str,
    annotation_file: Optional[str] = None,
    val_ratio: Optional[float] = 0.2,
    random_state: int = 42,
    subsample: float = None,
    logger: Optional[logging.Logger] = None,
    state_transform_dict: Optional[Dict[Union[str, int], Union[str, int]]] = None,
    split_by_video: bool = False,
    fold_index: Optional[int] = None,
    n_folds: int = 3,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Load segments from metadata and split into train/val sets.

    Args:
        cache_dir: Directory containing preprocessed HDF5 files.
        annotation_file: Path to pickle file with action labels.
        val_ratio: Fraction for validation split. If None, all data is returned
            as train_segments with an empty val list.
        random_state: Random seed for reproducibility.
        subsample: If provided (0 < subsample <= 1), randomly keep this fraction
            of segments (useful for debugging).
        logger: Optional logger for progress messages.
        state_transform_dict: Optional label remapping dict applied after 0-indexing.
        split_by_video: If True, split by session/video rather than by segment
            (harder, more realistic evaluation).
        fold_index: K-fold validation fold index. If provided, K-fold is used
            instead of a single train/val split.
        n_folds: Number of K-fold folds.

    Returns:
        Tuple of (train_segments, val_segments, annotation_metadata).
    """
    logger = logger or logging.getLogger(__name__)
    cache_dir = Path(cache_dir)

    if annotation_file:
        logger.info(f"Loading annotations from {annotation_file}")
        with open(annotation_file, "rb") as f:
            annotation_metadata = pickle.load(f)

        if "version" not in annotation_metadata:
            raise ValueError("Invalid metadata format: missing version")
        if "segments" not in annotation_metadata:
            raise ValueError("No segments found in global metadata")

        all_segments = annotation_metadata["segments"]
        logger.info(f"Loaded {len(all_segments)} segments from metadata")

        segment_map = {
            (s["session_id"], s.get("segment_idx", i)): s
            for i, s in enumerate(all_segments)
        }

        segments = []
        for anno in annotation_metadata["segments"]:
            session_id = anno.get("session_id")
            segment_idx = anno.get("segment_idx")
            if (session_id, segment_idx) in segment_map:
                segment = segment_map[(session_id, segment_idx)]
                segment["label"] = anno["label"]
                segment["mouse_id"] = session_id.split("_")[0]
                segments.append(segment)
    else:
        raise ValueError("No annotation file provided")

    logger.info(f"Found {len(segments)} labeled segments")

    # Remap 1-indexed labels to 0-indexed for PyTorch compatibility
    for segment in segments:
        if "label" in segment:
            original = segment["label"]
            if isinstance(original, int) and original > 0:
                segment["label"] = original - 1

    if state_transform_dict is not None:
        logger.info("Applying state transformation to labels")
        orig_dist: Dict = {}
        new_dist: Dict = {}
        for segment in segments:
            orig = segment["label"]
            orig_dist[orig] = orig_dist.get(orig, 0) + 1
            if orig in state_transform_dict:
                segment["label"] = state_transform_dict[orig]
                new_dist[segment["label"]] = new_dist.get(segment["label"], 0) + 1
        logger.info(f"Label dist before: {orig_dist}")
        logger.info(f"Label dist after:  {new_dist}")

    if subsample is not None:
        n = int(len(segments) * subsample)
        indices = np.random.choice(len(segments), size=n, replace=False)
        segments = [segments[i] for i in indices]
        logger.info(f"Subsampled to {len(segments)} segments")

    if val_ratio is None or val_ratio == 0.0:
        logger.info("No validation split — returning all segments as training data")
        return segments, [], annotation_metadata

    # K-fold cross-validation
    if fold_index is not None:
        if split_by_video:
            session_segments: Dict = {}
            for seg in segments:
                sid = seg["session_id"]
                session_segments.setdefault(sid, []).append(seg)

            session_ids = list(session_segments.keys())
            session_labels = [
                max(
                    set(s["label"] for s in session_segments[sid]),
                    key=list(s["label"] for s in session_segments[sid]).count,
                )
                for sid in session_ids
            ]

            logger.info(
                f"Video-based K-fold: {n_folds} folds, fold {fold_index} as val"
            )
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
            folds = list(skf.split(session_ids, session_labels))
            train_idx, val_idx = folds[fold_index]

            train_segments: List[Dict] = []
            val_segments: List[Dict] = []
            for i in train_idx:
                train_segments.extend(session_segments[session_ids[i]])
            for i in val_idx:
                val_segments.extend(session_segments[session_ids[i]])
        else:
            logger.info(
                f"Segment-based K-fold: {n_folds} folds, fold {fold_index} as val"
            )
            labels = [s.get("label") for s in segments]
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
            folds = list(skf.split(segments, labels))
            train_idx, val_idx = folds[fold_index]
            train_segments = [segments[i] for i in train_idx]
            val_segments = [segments[i] for i in val_idx]

    elif split_by_video:
        session_segments = {}
        for seg in segments:
            sid = seg["session_id"]
            session_segments.setdefault(sid, []).append(seg)

        session_ids = list(session_segments.keys())
        session_labels = [
            max(
                set(s["label"] for s in session_segments[sid]),
                key=list(s["label"] for s in session_segments[sid]).count,
            )
            for sid in session_ids
        ]

        logger.info(f"Video-based split: {len(session_ids)} sessions")
        train_sids, val_sids = train_test_split(
            session_ids,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=session_labels,
        )

        train_segments = []
        val_segments = []
        for sid in train_sids:
            train_segments.extend(session_segments[sid])
        for sid in val_sids:
            val_segments.extend(session_segments[sid])
        logger.info(
            f"Train sessions: {len(train_sids)} | Val sessions: {len(val_sids)}"
        )

    else:
        logger.info("Segment-based random split")
        train_segments, val_segments = train_test_split(
            segments,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=[s.get("label") for s in segments],
        )

    logger.info(
        f"Class dist — Train: {get_class_distribution(train_segments)} | "
        f"Val: {get_class_distribution(val_segments)}"
    )
    return train_segments, val_segments, annotation_metadata
