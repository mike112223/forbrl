
import numpy as np

from .base import Policy
from ...utils import POLICIES


@POLICIES.register_module
class VPGPolicy(Policy):

    def __init__(self, epsilon, algo='epsilon_greedy'):
        self.epsilon = epsilon
        self.algo = algo
        self.action = dict()

    def default(self, feats):
        push_feat, grasp_feat = feats
        if push_feat > grasp_feat:
            self.action['action'] = 'push'
        else:
            self.action['action'] = 'grasp'

    def random(self, feats):
        if np.random.randint(0, 2) == 0:
            self.action['action'] = 'push'
        else:
            self.action['action'] = 'grasp'

    def choose(self, feats):
        push_feat, grasp_feat = feats
        feats_max = [np.max(push_feat), np.max(grasp_feat)]

        try:
            getattr(self, self.algo)(feats_max)
        except NotImplementedError:
            print('Not Implemented')
        else:
            if self.action['action'] == 'push':
                self.action['best_idx'] = np.array(np.unravel_index(np.argmax(push_feat), push_feat.shape))
                print('push max: ', np.max(push_feat))
            else:
                self.action['best_idx'] = np.array(np.unravel_index(
                    np.argmax(grasp_feat), grasp_feat.shape))
                print('grasp max: ', np.max(grasp_feat))

            return self.action.copy()
