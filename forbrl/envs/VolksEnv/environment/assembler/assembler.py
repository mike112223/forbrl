import logging
import os

from addict import Dict

from .. import utils
from ..cerebrum import build_cerebrum
from ..equipment import build_equipment
from ..loggers import build_logger
from ..runners import build_runner


def assemble(cfg_fp):

    logging_step = 1
    env = Dict()

    _, fullname = os.path.split(cfg_fp)
    fname, ext = os.path.splitext(fullname)

    cfg = utils.Config.fromfile(cfg_fp)

    # make workdir if not exist
    root_workdir = cfg.pop('root_workdir')
    cfg['workdir'] = os.path.join(root_workdir, fname)

    os.makedirs(cfg['workdir'], exist_ok=True)

    # set seed if provided
    seed = cfg.pop('seed', None)
    if seed is not None:
        utils.set_random_seed(seed)

    # 1. logging
    _ = build_logger(cfg['logger'], dict(workdir=cfg['workdir'],
                                         logger_name='environment'))
    assemble_logger = logging.getLogger(__name__)

    # 2. equipment
    env['equipment'] = build_equipment(cfg['equipment'])
    flatten_env(env, 'equipment')
    assemble_logger.info(f"Assemble, Step {logging_step}, built equipment")

    # 3. cerebrum
    if cfg.get('cerebrum', None) is not None:
        logging_step += 1
        cerebrum, update_list = build_cerebrum(cfg['cerebrum'], env)
        env['cerebrum'] = cerebrum
        flatten_env(env, 'cerebrum')
        assemble_logger.info(f"Assemble, Step {logging_step}, built cerebrum")

    # 4. runner
    if cfg.get('runner', None) is not None:
        logging_step += 1
        new_kwarg = {}
        for kwarg, name in cfg['runner'].items():
            if isinstance(name, str):
                if name in env:
                    new_kwarg[kwarg] = env[name]
                else:
                    new_kwarg[kwarg] = name
            else:
                new_kwarg[kwarg] = name
        runner = build_runner(new_kwarg)
        assemble_logger.info(f"Assemble, Step {logging_step}, built runner")
        return runner

    return env


def flatten_env(env, field):
    for cat, components in env[field].items():
        for components_name, component in components.items():
            if components_name in env:
                raise KeyError(f"Name {components_name} already used "
                               f"by {env[components_name]}.")
            env[components_name] = component
