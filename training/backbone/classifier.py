"""TerraFM-B with early S1/S2 fusion and a BigEarthNet multi-label head.

Trimmed from the AI4DIS `use_case_2/model.py` (EarthObservationClassifier) to the one
configuration used for the released backbone: TerraFM base, 14-channel early fusion.
Attribute names are unchanged, so the state dict of a trained model has the same keys
(`backbone.*`, `classifier.*`) as the original checkpoint and can be passed directly to
scripts/export_checkpoint.py.
"""
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f2a.terrafm import terrafm_base  # noqa: E402

EMBED_DIM = 768


class EarlyFusionClassifier(nn.Module):
    def __init__(self, num_classes=19, pretrained_path=None, freeze_backbone=False, dropout=0.1):
        super().__init__()
        self.backbone = terrafm_base()
        if pretrained_path:
            load_terrafm_weights(self.backbone, pretrained_path)
        self.classifier = nn.Sequential(nn.LayerNorm(EMBED_DIM), nn.Dropout(dropout),
                                        nn.Linear(EMBED_DIM, num_classes))
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze the backbone except the new early-fusion patch embedding."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.patch_embed.conv2d_s1_s2.parameters():
            parameter.requires_grad = True
        self.backbone.patch_embed.s1_s2_embed.requires_grad = True

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, x, is_l2a=True):
        # 14 channels (VV, VH, B01-B12) go through the early-fusion convolution together.
        return self.classifier(self.backbone.forward(x, is_l2a=is_l2a))

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def load_terrafm_weights(backbone, path):
    """Load the public TerraFM-B weights; the early-fusion layers stay newly initialised."""
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {k: v for k, v in state.items() if not k.startswith('head')}
    report = backbone.load_state_dict(state, strict=False)
    print(f'Loaded TerraFM weights from {path}: {report}')
