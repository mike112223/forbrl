import datetime
import inspect
import logging
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np


def draw_axis(img, axis_points):
    corner = tuple(axis_points[0].ravel())
    for i in range(3):
        color = [0, 0, 0]
        color[i] = 255
        cv2.line(img, corner, tuple(axis_points[i + 1].ravel()), color, 5)
    # cv2.line(img, corner, tuple(axis_points[1].ravel()), (255, 0, 0), 5)
    # cv2.line(img, corner, tuple(axis_points[2].ravel()), (0, 255, 0), 5)
    # cv2.line(img, corner, tuple(axis_points[3].ravel()), (0, 0, 255), 5)
    return img


def get_time_iso():
    time_stamp = time.time()
    now = datetime.datetime.fromtimestamp(time_stamp)
    return now.isoformat()[:19].replace(':', '-')


def save_collected(session_dir, flat=False, **data):
    # TODO: generate a summery for the session include info like devices,
    #  robotic_arms, calibrate board, blah blah blah
    dirs = dict()
    timestamp = get_time_iso()
    for key in data:
        if flat:
            dirs[key] = os.path.join(session_dir)
        else:
            dirs[key] = os.path.join(session_dir, key)
        os.makedirs(dirs[key], exist_ok=True)
    for key in data:
        file_name = os.path.join(dirs[key], f"{key}_{timestamp}")
        if 'color_img' in key:
            cv2.imwrite(file_name + '.png',
                        cv2.cvtColor(data[key], cv2.COLOR_BGR2RGB))
        else:
            with open(file_name + '.pkl', 'wb') as file:
                pickle.dump(data[key], file)


def basic_builder(cfg, registry_):
    components = {}
    for component_cfg in cfg:
        component_name = component_cfg.pop('name', None)
        if not component_name:
            raise KeyError(f"'name' not specified in the config for "
                           f"{cfg[component_cfg]}")
        component_name.replace(' ', '_')  # replace spaces in names
        if component_name in components:
            raise KeyError(f"The name {component_name} was already used by "
                           f"{components[component_name]}")
        components[component_name] = build_from_cfg(component_cfg, registry_)
    return components


# modify from mmcv and mmdetection


def build_from_cfg(cfg, parent, default_args=None, src='registry'):
    if src == 'registry':
        return obj_from_dict_registry(cfg, parent, default_args)
    elif src == 'module':
        return obj_from_dict_module(cfg, parent, default_args)
    else:
        raise ValueError('Method %s is not supported' % src)


def obj_from_dict_module(info, parent=None, default_args=None):
    """
        Initialize an object from dict.
    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.
    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.
    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def obj_from_dict_registry(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    np.set_printoptions(precision=4)
    if not logger.hasHandlers():
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=log_level)
    return logger
