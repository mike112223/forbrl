from .motor_cortex import MOTOR_CORTEX
from .posterior_parietal_cortex import POSTERIOR_PARIETAL_CORTEX
from .visual_cortex import VISUAL_CORTEX
from ..utils import basic_builder

REGISTRY = {'motor_cortex': MOTOR_CORTEX,
            'posterior_parietal_cortex': POSTERIOR_PARIETAL_CORTEX,
            'visual_cortex': VISUAL_CORTEX, }


def build_cerebrum(cfg, env):
    cerebrum, update_list = {}, {}
    for cortex in cfg:
        if cortex not in REGISTRY:
            raise KeyError(f"Unrecognized cortex {cortex}, should be in "
                           f"{list(REGISTRY.keys())}")
        if cortex in cerebrum:
            raise KeyError(f"{cortex} already exist in cerebrum, with details "
                           f"{cerebrum[cortex]}")
        update_config(cfg[cortex], env)
        cerebrum[cortex] = basic_builder(cfg[cortex], REGISTRY[cortex])

    return cerebrum, update_list


def update_config(cfg, env):
    for component in cfg:
        for kwarg, val in component.items():
            if not isinstance(val, str):
                continue
            if val in env.keys():
                component[kwarg] = env[val]
