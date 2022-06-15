from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

ALGORITHMS = MODELS
BACKBONES = MODELS
PROJECTIONS = MODELS


def build_algorithm(cfg):
    """Build algorithm."""
    return ALGORITHMS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_projection(cfg):
    """Build projection."""
    return PROJECTIONS.build(cfg)
