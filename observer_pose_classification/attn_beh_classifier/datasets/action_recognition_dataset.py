"""Dataset for action recognition using preloaded HDF5 frame caches."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from data.sampling import AugmentationSampler


class ActionRecognitionDataset(Dataset):
    """Load video clips from HDF5 frame caches for action recognition.

    Frames are stored per session in HDF5 files and loaded on demand.
    Augmentation is applied only during training; class balancing is
    achieved via a WeightedRandomSampler returned by get_sampler().

    Args:
        segments: List of segment metadata dicts (must contain 'session_id',
            'segment_idx', and 'label' keys).
        cache_dir: Directory containing per-session HDF5 files.
        clip_length: Number of frames to sample per clip.
        target_size: Spatial size (H, W) to resize frames to.
        temporal_sampling: Frame sampling strategy ('uniform' or 'random').
        split: Dataset split; augmentation and balancing apply only to 'train'.
        use_augmentation: Enable spatiotemporal augmentation (train only).
        balance_classes: Enable inverse-frequency class weighting (train only).
        logger: Optional logger instance.
    """

    def __init__(
        self,
        segments: List[Dict],
        cache_dir: str,
        clip_length: int = 20,
        target_size: Tuple[int, int] = (128, 128),
        temporal_sampling: str = "uniform",
        split: Literal["train", "val", "test"] = "train",
        use_augmentation: bool = False,
        balance_classes: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.segments = segments
        self.cache_dir = Path(cache_dir)
        self.clip_length = clip_length
        self.target_size = target_size
        self.temporal_sampling = temporal_sampling
        self.split = split
        self.use_augmentation = use_augmentation and split == "train"
        self.balance_classes = balance_classes and split == "train"
        self.logger = logger or logging.getLogger(__name__)

        self._log_dataset_info()
        self._load_h5_files()

        if self.use_augmentation:
            self.aug_sampler = AugmentationSampler(target_size=target_size)

    def _log_dataset_info(self):
        """Log segment count and class distribution for this split."""
        class_dist: Dict = {}
        for segment in self.segments:
            label = segment.get("label")
            class_dist[label] = class_dist.get(label, 0) + 1
        self.logger.info(
            f"Split '{self.split}': {len(self.segments)} segments | dist: {class_dist}"
        )

    def _load_h5_files(self):
        """Resolve and cache HDF5 file paths for each session."""
        self.h5_files = {}
        session_ids = {seg["session_id"] for seg in self.segments}
        for session_id in session_ids:
            h5_path = self.cache_dir / f"{session_id}.h5"
            if h5_path.exists():
                self.h5_files[session_id] = h5_path
            else:
                self.logger.warning(f"HDF5 file not found: {h5_path}")
        self.logger.info(f"Resolved {len(self.h5_files)} HDF5 files")

    def get_frames(self, segment: Dict) -> torch.Tensor:
        """Load and preprocess frames for a segment.

        Args:
            segment: Segment metadata dict.

        Returns:
            Float tensor of shape [T, C, H, W] in [0, 1].
        """
        session_id = segment["session_id"]
        segment_idx = segment.get("segment_idx", 0)

        if session_id not in self.h5_files:
            raise ValueError(f"HDF5 file not found for session {session_id}")

        with h5py.File(self.h5_files[session_id], "r") as f:
            dataset_key = f"segment_{segment_idx}"
            if dataset_key not in f["frames"]:
                raise ValueError(
                    f"Segment {segment_idx} not found in HDF5 for session {session_id}"
                )
            frames = f["frames"][dataset_key][:]

        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        if frames.shape[2:] != self.target_size:
            frames = torch.nn.functional.interpolate(
                frames, size=self.target_size, mode="bilinear", align_corners=False
            )

        return frames

    def sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Temporally sample or pad frames to self.clip_length.

        Args:
            frames: Input frames [T, C, H, W].

        Returns:
            Tensor of shape [clip_length, C, H, W].
        """
        total_frames = frames.shape[0]

        if total_frames <= self.clip_length:
            if total_frames < self.clip_length:
                padding = torch.zeros(
                    (self.clip_length - total_frames, *frames.shape[1:]),
                    dtype=frames.dtype,
                    device=frames.device,
                )
                frames = torch.cat([frames, padding], dim=0)
            return frames

        if self.temporal_sampling == "uniform":
            indices = torch.linspace(0, total_frames - 1, self.clip_length).long()
            return frames[indices]
        else:
            start_idx = random.randint(0, total_frames - self.clip_length)
            return frames[start_idx : start_idx + self.clip_length]

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """Return a WeightedRandomSampler for inverse-frequency class balancing.

        Returns:
            WeightedRandomSampler if balance_classes is True, else None.
        """
        if not self.balance_classes:
            return None

        class_counts: Dict = {}
        for s in self.segments:
            label = s.get("label", 0)
            class_counts[label] = class_counts.get(label, 0) + 1

        weights = [1.0 / class_counts[s.get("label", 0)] for s in self.segments]
        return WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

    def __len__(self) -> int:
        """Return the total number of segments."""
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return a (frames, label) pair for the given index.

        Frames are loaded from HDF5, optionally augmented, and temporally
        sampled to clip_length. On error, returns a zero tensor with the
        correct label to avoid crashing the DataLoader.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (frames [clip_length, C, H, W], integer label).
        """
        idx = idx % len(self.segments)
        segment = self.segments[idx]

        try:
            frames = self.get_frames(segment)
            if self.use_augmentation:
                frames = self.aug_sampler(frames)
            frames = self.sample_frames(frames)
            label = segment.get("label") or 0
            return frames, label
        except Exception:
            frames = torch.zeros(
                (self.clip_length, 3, self.target_size[0], self.target_size[1])
            )
            return frames, segment.get("label", 0)
