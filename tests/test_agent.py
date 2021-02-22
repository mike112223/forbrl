
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('../forbrl'))

from forbrl.utils import Config, build_agent


cfg_fp = os.path.join(os.path.abspath('configs/vpg'), 'vpg.py')
cfg = Config.fromfile(cfg_fp)

agent = build_agent(cfg['agents'])


import numpy as np
agent.algorithm.model(np.zeros((320,320,6)))