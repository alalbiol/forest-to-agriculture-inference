"""Temporal forest-to-agriculture scoring with a frozen TerraFM backbone.

Pipeline, identical to the study that produced the benchmark scores:
  1. A 14-band chip (Sentinel-1 VV, VH in dB; Sentinel-2 B01-B12 surface-reflectance DN)
     is bilinearly resized to 224 x 224 and normalised with the BigEarthNet statistics.
  2. The before and after chips are passed together through the frozen TerraFM-B backbone
     (early S1/S2 fusion, fine-tuned on BigEarthNet). Block-11 tokens form a 14 x 14 grid
     of 768-d features, stored in float16 as in the study.
  3. A linear land-use head gives per-patch probabilities of primary forest, secondary
     forest, agriculture and mining.
  4. Temporal score = mean of the 9 largest p_before(primary forest) * p_after(agriculture).
     After-only score = mean of the 9 largest p_after(agriculture).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .terrafm import terrafm_base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONE = ROOT / 'checkpoints/terrafm_b_bigearthnet_early_fusion.pt'
DEFAULT_HEAD = ROOT / 'heads/landuse_head.pt'

S1_BANDS = ['VV', 'VH']
S2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
BAND_ORDER = S1_BANDS + S2_BANDS
# BigEarthNet band statistics used when the backbone was fine-tuned (order: BAND_ORDER).
MEAN = [-12.64386368, -19.35255814, 361.07678223, 438.37207031, 614.05566406,
        588.40960693, 942.84332275, 1769.93164062, 2049.55151367, 2193.29199219,
        2235.55664062, 2241.45532227, 1568.22680664, 997.73248291]
STD = [5.1334939, 5.5905056, 575.06872559, 607.02685547, 603.29681396,
       684.56884766, 738.43267822, 1100.45605469, 1275.80541992, 1369.3717041,
       1356.54406738, 1316.39331055, 1070.16125488, 813.52764893]
CLASSES = ['primary_forest', 'secondary_forest', 'agriculture', 'mining']
INPUT_SIZE = 224
TOKEN_GRID = 14
TOKEN_BLOCK = 11
EMBED_DIM = 768
_MEAN = np.asarray(MEAN, dtype=np.float32).reshape(-1, 1, 1)
_STD = np.asarray(STD, dtype=np.float32).reshape(-1, 1, 1)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def preprocess(stack: np.ndarray) -> torch.Tensor:
    """(14, H, W) raw bands in BAND_ORDER -> (14, 224, 224) normalised float32 tensor."""
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[0] != len(BAND_ORDER):
        raise ValueError(f'expected (14, H, W) bands in order {BAND_ORDER}, got {stack.shape}')
    if not np.isfinite(stack).all():
        raise ValueError('input contains non-finite values')
    data = stack.astype(np.float32, copy=True)
    for index in range(len(S1_BANDS)):
        # Linear backscatter is converted to dB; Earth Engine S1_GRD is already in dB.
        if np.min(data[index]) >= 0:
            data[index] = 10.0 * np.log10(data[index] + 1e-6)
    if data.shape[1:] != (INPUT_SIZE, INPUT_SIZE):
        tensor = torch.from_numpy(data)[None]
        data = F.interpolate(tensor, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear',
                             align_corners=False)[0].numpy()
    return torch.from_numpy((data - _MEAN) / _STD).float()


def load_backbone(path=DEFAULT_BACKBONE, device='cpu') -> torch.nn.Module:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    backbone = terrafm_base()
    backbone.load_state_dict(checkpoint['state_dict'], strict=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone.to(device).eval()


def load_head(path=DEFAULT_HEAD):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if list(state['classes']) != CLASSES:
        raise ValueError(f'unexpected head classes: {state["classes"]}')
    head = torch.nn.Linear(EMBED_DIM, len(CLASSES))
    head.load_state_dict(state['state_dict'])
    return head.eval(), state['mean'].float(), state['std'].float()


@torch.no_grad()
def tokens(backbone, chips: list[np.ndarray], device='cpu') -> np.ndarray:
    """Block-11 tokens, shape (len(chips), 768, 14, 14), rounded to float16 as in the study."""
    batch = torch.stack([preprocess(chip) for chip in chips]).to(device)
    grid = backbone.extract_feature(batch, return_h_w=True, out_indices=[TOKEN_BLOCK], is_l2a=True)[0]
    return grid.cpu().numpy().astype(np.float16)


@torch.no_grad()
def probabilities(head, mean, std, grid: np.ndarray) -> np.ndarray:
    """(768, 14, 14) tokens -> (4, 14, 14) class probabilities."""
    x = torch.from_numpy(grid.astype(np.float32).reshape(EMBED_DIM, -1).T)
    p = head((x - mean) / std).softmax(-1).numpy()
    return p.T.reshape(len(CLASSES), TOKEN_GRID, TOKEN_GRID)


def top9(values: np.ndarray) -> float:
    return float(np.sort(np.asarray(values).reshape(-1))[-9:].mean())


def score_pair(backbone, head_bundle, before: np.ndarray, after: np.ndarray, device='cpu') -> dict:
    """Score one before/after chip pair; returns maps and chip-level scores."""
    grid = tokens(backbone, [before, after], device)
    p_before = probabilities(*head_bundle, grid[0])
    p_after = probabilities(*head_bundle, grid[1])
    product = p_before[0] * p_after[2]
    return {'p_before': p_before, 'p_after': p_after, 'temporal_map': product,
            'temporal_score': top9(product), 'after_only_score': top9(p_after[2]),
            'tokens': grid}
