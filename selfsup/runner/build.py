from selfsup.utils import Registry, build_from_cfg

RUNNERS = Registry('runner', build_func=build_from_cfg)


def build_runner(cfg):
    return RUNNERS.build(cfg)
