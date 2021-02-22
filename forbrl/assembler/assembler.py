
import os
import time

import random
import numpy as np
import torch

from ..utils import Config, build_agent, build_env, build_runner


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def assemble(cfg_fp, test_mode=False):

    # 1.base config
    # workdir
    print('build config!')
    cfg = Config.fromfile(cfg_fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    _, fullname = os.path.split(cfg_fp)
    fname, ext = os.path.splitext(fullname)

    # make workdir if not exist
    root_workdir = cfg.pop('root_workdir')
    timestamp = time.strftime('%Y-%m-%d.%H:%M:%S', time.localtime())
    cfg['workdir'] = os.path.join(root_workdir, fname, timestamp)

    os.makedirs(cfg['workdir'], exist_ok=True)
    print('configs: ', cfg)

    # seed
    seed = cfg.get('seed', None)
    deterministic = cfg.get('deterministic', False)
    if seed is not None:
        set_random_seed(seed, deterministic)

    # 2. agent
    print('build agent!')
    agent = build_agent(cfg['agents'],
                        dict(workdir=cfg['workdir']))

    # 3. env
    print('build env!')
    env = build_env(cfg['envs'])

    # 4. runner
    print('build runner!')
    runner = build_runner(
        cfg['runner'],
        dict(
            agent=agent,
            env=env,
            workdir=cfg['workdir']
        )
    )

    return runner
