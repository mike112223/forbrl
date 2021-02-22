from addict import Dict

from .end_effectors import END_EFFECTORS
from .fiducial_markers import FIDUCIAL_MARKERS
from .robotic_arms import ROBOTIC_ARMS
from .vision_sensors import VISION_SENSORS
from .sim_environments import SIM_ENVIRONMENTS
from .objects import OBJECTS
from ..utils import basic_builder

REGISTRY = {'fiducial_markers': FIDUCIAL_MARKERS,
            'robotic_arms': ROBOTIC_ARMS,
            'vision_sensors': VISION_SENSORS,
            'end_effectors': END_EFFECTORS,
            'sim_environments': SIM_ENVIRONMENTS,
            'objects': OBJECTS}


def build_equipment(cfg):
    equipment = Dict()
    for key in cfg:
        if key not in REGISTRY:
            raise KeyError(f"Unrecognized key {key}, should be in "
                           f"{list(REGISTRY.keys())}")
        if key in equipment:
            raise KeyError(f"{key} already exist in equipment, with details "
                           f"{equipment[key]}")
        equipment[key] = basic_builder(cfg[key], REGISTRY[key])

    return equipment
