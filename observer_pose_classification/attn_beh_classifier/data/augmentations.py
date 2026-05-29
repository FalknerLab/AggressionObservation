"""Spatiotemporal video augmentation pipeline.

Provides AugmentationConfig and VideoAugmentation for applying rotation,
random erasing, temporal flipping, and spatial cropping to video clips.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AugmentationConfig:
    """Configuration for video augmentation probabilities and parameters."""

    rotation_prob: float = 0.5
    erase_prob: float = 0.3
    flip_prob: float = 0.5
    crop_prob: float = 0.8

    max_rotation_angle: float = 45.0
    erase_ratio: Tuple[float, float] = (0.02, 0.33)
    crop_ratio: float = 0.5
    target_size: Tuple[int, int] = (256, 256)


class VideoAugmentation:
    """Spatiotemporal augmentation pipeline for video clips.

    Applies a configurable combination of rotation, random erasing,
    temporal flip, and spatial crop-resize.

    Args:
        config: Augmentation configuration; uses defaults if None.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def _rotation(self, frames: torch.Tensor) -> torch.Tensor:
        """Rotate all frames by the same random angle using grid sampling.

        Args:
            frames: Input tensor [T, C, H, W].

        Returns:
            Rotated tensor [T, C, H, W].
        """
        angle = random.uniform(
            -self.config.max_rotation_angle, self.config.max_rotation_angle
        )
        angle_rad = angle * np.pi / 180.0

        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            device=frames.device,
        ).float()

        H, W = frames.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frames.device),
            torch.linspace(-1, 1, W, device=frames.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        rotated_grid = torch.matmul(grid, rotation_matrix.T).reshape(H, W, 2)
        rotated_grid = rotated_grid.unsqueeze(0).repeat(frames.size(0), 1, 1, 1)

        return F.grid_sample(
            frames,
            rotated_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _random_erase(
        self,
        frames: torch.Tensor,
        same_region: bool = True,
    ) -> torch.Tensor:
        """Zero out a random rectangular region in the frames.

        Args:
            frames: Input tensor [T, C, H, W].
            same_region: If True, erase the same region in all frames.

        Returns:
            Erased tensor [T, C, H, W].
        """
        T, C, H, W = frames.shape
        erase_ratio = random.uniform(*self.config.erase_ratio)
        erase_area = H * W * erase_ratio
        erase_h = int(np.sqrt(erase_area))
        erase_w = int(erase_area / erase_h)

        if same_region:
            top = random.randint(0, H - erase_h)
            left = random.randint(0, W - erase_w)
            mask = torch.ones_like(frames)
            mask[:, :, top : top + erase_h, left : left + erase_w] = 0
            return frames * mask
        else:
            erased = []
            for t in range(T):
                top = random.randint(0, H - erase_h)
                left = random.randint(0, W - erase_w)
                mask = torch.ones_like(frames[t])
                mask[:, top : top + erase_h, left : left + erase_w] = 0
                erased.append(frames[t] * mask)
            return torch.stack(erased)

    def _temporal_flip(self, frames: torch.Tensor) -> torch.Tensor:
        """Reverse the temporal order of frames.

        Args:
            frames: Input tensor [T, C, H, W].

        Returns:
            Temporally flipped tensor [T, C, H, W].
        """
        return frames.flip(0)

    def _spatial_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """Randomly crop a region covering crop_ratio of the area and resize.

        Args:
            frames: Input tensor [T, C, H, W].

        Returns:
            Cropped and resized tensor [T, C, target_H, target_W].
        """
        T, C, H, W = frames.shape
        crop_area = H * W * self.config.crop_ratio
        crop_h = (int(np.sqrt(crop_area)) // 8) * 8
        crop_w = (int(crop_area / crop_h) // 8) * 8

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        cropped = frames[:, :, top : top + crop_h, left : left + crop_w]

        return F.interpolate(
            cropped, size=self.config.target_size, mode="bilinear", align_corners=False
        )

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatiotemporal augmentations to a video clip.

        Each augmentation is applied independently with its configured probability.

        Args:
            frames: Input tensor [T, C, H, W].

        Returns:
            Augmented tensor [T, C, H, W].
        """
        if random.random() < self.config.rotation_prob:
            frames = self._rotation(frames)
        if random.random() < self.config.erase_prob:
            frames = self._random_erase(frames, same_region=random.random() < 0.5)
        if random.random() < self.config.flip_prob:
            frames = self._temporal_flip(frames)
        if random.random() < self.config.crop_prob:
            frames = self._spatial_crop(frames)
        return frames
