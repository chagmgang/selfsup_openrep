from selfsup.utils import Registry, build_from_cfg

MODELS = Registry('models', build_func=build_from_cfg)

ALGORITHM = MODELS
BACKBONE = MODELS
PROJECTION = MODELS


def build_algorithm(cfg):
    return ALGORITHM.build(cfg)


def build_backbone(cfg):
    return BACKBONE.build(cfg)


def build_projection(cfg):
    return PROJECTION.build(cfg)
