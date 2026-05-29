"""Action recognition model with custom 3D CNN encoder and MLP/attentive classifier.

The default architecture uses VideoEncoder (3D CNN with spatial attention) as the
backbone and either an MLP head or an AttentiveClassifier (cross-attention) as
the classification head.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.spatial_attention import VideoEncoder

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AttentiveClassifier(nn.Module):
    """Cross-attention based classifier.

    A learnable query vector attends over the encoder feature sequence, then a
    linear layer maps the attended representation to class logits.

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of output classes.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate applied before the classification linear layer.
        use_encoder_attention: Whether to use encoder attention maps (unused in default workflow).
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_encoder_attention: bool = False,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate * 0.5,
        )
        self.use_encoder_attention = use_encoder_attention
        self.feature_dropout = nn.Dropout(dropout_rate * 0.5)
        self.pre_classifier_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, attention_maps=None, return_features=False):
        """Forward pass.

        Args:
            x: Input features of shape [B, D] or [B, N, D].
            attention_maps: Optional encoder attention maps (ignored in default workflow).
            return_features: If True, also return the attended feature vector.

        Returns:
            Class logits [B, num_classes], and optionally the attended features [B, D].
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        x = self.feature_dropout(x)
        attn_output, _ = self.cross_attention(query, x, x)
        attn_output = self.layer_norm(attn_output)
        attn_output = self.pre_classifier_dropout(attn_output)
        logits = self.classifier(attn_output.squeeze(1))

        if return_features:
            return logits, attn_output.squeeze(1)
        return logits


class ActionRecognitionModel(nn.Module):
    """Action recognition model: VideoEncoder backbone + MLP or AttentiveClassifier head.

    Args:
        num_classes: Number of action classes.
        pretrained_encoder_path: Optional path to a contrastive pre-training checkpoint
            from which encoder weights are extracted.
        feature_dim: Feature dimension from the encoder (auto-detected if None).
        dropout_rate: Dropout rate for the classification head.
        freeze_encoder: Whether to freeze the encoder during initial training.
        input_size: Encoder input size as (C, T, H, W).
        use_attentive_classifier: If True, use AttentiveClassifier instead of MLP.
        num_heads: Number of attention heads (used only when use_attentive_classifier=True).
        use_encoder_attention: Whether to pass encoder attention maps to AttentiveClassifier.
        encoder_type: Reserved for future encoder variants; currently only 'custom' is used.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_encoder_path: Optional[str] = None,
        feature_dim: int = None,
        dropout_rate: float = 0.1,
        freeze_encoder: bool = True,
        input_size: Tuple[int, int, int, int] = (3, 20, 128, 128),
        use_attentive_classifier: bool = False,
        num_heads: int = 8,
        use_encoder_attention: bool = False,
        encoder_type: str = "custom",
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.encoder = VideoEncoder(in_channels=input_size[0], input_size=input_size)

        if pretrained_encoder_path:
            self._load_pretrained_encoder(pretrained_encoder_path)

        if feature_dim is None:
            feature_dim = self.encoder.feature_dim

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.use_attentive_classifier = use_attentive_classifier
        self.use_encoder_attention = use_encoder_attention

        if use_attentive_classifier:
            self.classifier = AttentiveClassifier(
                feature_dim=feature_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_encoder_attention=use_encoder_attention,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes),
            )

        self.feature_dim = feature_dim

    def _load_pretrained_encoder(self, checkpoint_path: str):
        """Load encoder weights from a contrastive pre-training checkpoint.

        Extracts keys matching 'encoder.*' (excluding momentum_encoder) from
        model_state_dict and loads them into self.encoder with strict=False.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

        encoder_state_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if "encoder" in key and "momentum_encoder" not in key:
                if key.startswith("_orig_mod.encoder."):
                    new_key = key[18:]
                elif key.startswith("encoder."):
                    new_key = key[8:]
                else:
                    encoder_idx = key.find("encoder.")
                    new_key = key[encoder_idx + 8 :]
                encoder_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            encoder_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(
                f"Missing keys loading encoder ({len(missing_keys)}): "
                f"{missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
            )
        if unexpected_keys:
            logger.warning(f"Unexpected keys loading encoder: {unexpected_keys}")

    def freeze_encoder(self):
        """Freeze encoder parameters (use during classifier-only fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def count_trainable_parameters(self) -> Dict[str, int]:
        """Count trainable parameters in encoder and classifier separately.

        Returns:
            Dictionary with keys 'encoder', 'classifier', 'total'.
        """
        encoder_states = {
            n: p.requires_grad for n, p in self.encoder.named_parameters()
        }
        classifier_states = {
            n: p.requires_grad for n, p in self.classifier.named_parameters()
        }

        for param in self.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = True
        encoder_params = count_parameters(self)

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        classifier_params = count_parameters(self)

        for name, param in self.encoder.named_parameters():
            param.requires_grad = encoder_states[name]
        for name, param in self.classifier.named_parameters():
            param.requires_grad = classifier_states[name]

        return {
            "encoder": encoder_params,
            "classifier": classifier_params,
            "total": encoder_params + classifier_params,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input video tensor of shape [B, T, C, H, W].

        Returns:
            Class logits of shape [B, num_classes].
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        features, attention_maps = self.encoder(x)

        if (
            self.use_attentive_classifier
            and self.use_encoder_attention
            and attention_maps
        ):
            logits = self.classifier(features, attention_maps)
        else:
            logits = self.classifier(features)

        return logits
