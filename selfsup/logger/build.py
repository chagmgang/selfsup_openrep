from selfsup.utils import Registry, build_from_cfg

LOGGERS = Registry('logger', build_func=build_from_cfg)


def build_logger(cfg):
    return LOGGERS.build(cfg)
