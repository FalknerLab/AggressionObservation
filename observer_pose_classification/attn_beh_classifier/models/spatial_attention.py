"""3D CNN encoder with multi-level spatial attention for video feature extraction.

This module provides the VideoEncoder and SpatialAttention components used
by ActionRecognitionModel.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant regions in video frames.

    Args:
        in_channels: Number of input channels.
        reduction_ratio: Channel reduction ratio for attention computation.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W).

        Returns:
            Tuple of (attended features, spatial attention map).
        """
        channel_att = self.channel_attention(x)
        x = x * channel_att
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        return x, spatial_att


class VideoEncoder(nn.Module):
    """3D CNN encoder with multi-level spatial attention for video feature extraction.

    Builds a stack of 3D convolutional blocks, each followed by a SpatialAttention
    module. Temporal downsampling is applied when the current temporal dimension
    is greater than 1; otherwise only spatial downsampling is applied.

    Args:
        in_channels: Number of input channels (default 3 for RGB).
        base_channels: Number of base channels, doubled at each layer.
        num_layers: Number of 3D conv + attention blocks.
        input_size: Input size as (C, T, H, W).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        input_size: Tuple[int, int, int, int] = (3, 10, 256, 256),
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        in_ch = in_channels
        _, T, H, W = input_size
        current_T, current_H, current_W = T, H, W

        for i in range(num_layers):
            out_ch = base_channels * (2**i)

            if current_T > 1:
                conv_block = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=(2, 2, 2)),
                )
                current_T = current_T // 2
            else:
                conv_block = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=(1, 2, 2)),
                )

            current_H = current_H // 2
            current_W = current_W // 2

            self.layers.append(conv_block)
            self.attention_layers.append(SpatialAttention(out_ch))
            in_ch = out_ch

        self.feature_dim = in_ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input video tensor of shape (B, C, T, H, W).

        Returns:
            Tuple of (global-average-pooled features [B, feature_dim],
                      list of spatial attention maps from each layer).
        """
        attention_maps = []
        for conv_block, attention in zip(self.layers, self.attention_layers):
            x = conv_block(x)
            x, attn_map = attention(x)
            attention_maps.append(attn_map)

        features = x.mean(dim=[2, 3, 4])
        return features, attention_maps
