
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('../forbrl'))

from forbrl.utils import Config, build_env


cfg_fp = os.path.join(os.path.abspath('configs/vpg'), 'vpg.py')
cfg = Config.fromfile(cfg_fp)

env = build_env(cfg['envs'])

env.reset()
env.step({'action': 'push', 'best_idx': np.array([1, 15, 15])})
env.step({'action': 'grasp', 'best_idx': np.array([1, 15, 15])})
