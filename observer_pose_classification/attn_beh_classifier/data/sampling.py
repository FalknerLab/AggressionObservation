"""Augmentation strategy sampler for video training.

Randomly selects among spatial-only, temporal-only, mixed, or no augmentation
each time a clip is processed during training.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from data.augmentations import AugmentationConfig, VideoAugmentation


@dataclass
class SamplingConfig:
    """Probabilities and per-strategy AugmentationConfig instances.

    The four probabilities should sum to 1.0.
    """

    basic_aug_prob: float = 0.3
    temporal_aug_prob: float = 0.3
    mixed_aug_prob: float = 0.3
    no_aug_prob: float = 0.1

    basic_aug_config: AugmentationConfig = AugmentationConfig(
        rotation_prob=0.5,
        erase_prob=0.3,
        flip_prob=0.0,
        crop_prob=0.8,
        max_rotation_angle=45.0,
        erase_ratio=(0.02, 0.33),
        crop_ratio=0.5,
        target_size=(256, 256),
    )

    temporal_aug_config: AugmentationConfig = AugmentationConfig(
        rotation_prob=0.0,
        erase_prob=0.0,
        flip_prob=0.5,
        crop_prob=0.0,
        max_rotation_angle=45.0,
        erase_ratio=(0.02, 0.33),
        crop_ratio=0.5,
        target_size=(256, 256),
    )

    mixed_aug_config: AugmentationConfig = AugmentationConfig(
        rotation_prob=0.5,
        erase_prob=0.3,
        flip_prob=0.5,
        crop_prob=0.8,
        max_rotation_angle=45.0,
        erase_ratio=(0.02, 0.33),
        crop_ratio=0.5,
        target_size=(256, 256),
    )


class AugmentationSampler:
    """Samples an augmentation strategy and applies it to a video clip.

    Args:
        config: Sampling configuration with per-strategy probabilities and params.
        target_size: Spatial output size (H, W); overrides config target sizes.
    """

    def __init__(
        self,
        config: Optional[SamplingConfig] = None,
        target_size: Tuple[int, int] = (256, 256),
    ):
        self.config = config or SamplingConfig()
        self.target_size = target_size

        self.config.basic_aug_config.target_size = target_size
        self.config.temporal_aug_config.target_size = target_size
        self.config.mixed_aug_config.target_size = target_size

        self.basic_aug = VideoAugmentation(config=self.config.basic_aug_config)
        self.temporal_aug = VideoAugmentation(config=self.config.temporal_aug_config)
        self.mixed_aug = VideoAugmentation(config=self.config.mixed_aug_config)

    def _sample_augmentation_type(self) -> str:
        """Randomly select an augmentation strategy based on configured probabilities.

        Returns:
            One of 'basic', 'temporal', 'mixed', or 'none'.
        """
        r = random.random()
        if r < self.config.basic_aug_prob:
            return "basic"
        elif r < self.config.basic_aug_prob + self.config.temporal_aug_prob:
            return "temporal"
        elif (
            r
            < self.config.basic_aug_prob
            + self.config.temporal_aug_prob
            + self.config.mixed_aug_prob
        ):
            return "mixed"
        else:
            return "none"

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply the sampled augmentation to video frames.

        Args:
            frames: Input tensor [T, C, H, W].

        Returns:
            Augmented tensor [T, C, H, W].
        """
        aug_type = self._sample_augmentation_type()
        if aug_type == "basic":
            return self.basic_aug(frames)
        elif aug_type == "temporal":
            return self.temporal_aug(frames)
        elif aug_type == "mixed":
            return self.mixed_aug(frames)
        else:
            return frames
