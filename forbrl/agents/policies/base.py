
from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):

    def __init__(self, epsilon, algo='epsilon_greedy'):
        self.epsilon = epsilon

    def epsilon_greedy(self, feats):
        '''Epsilon-greedy policy: with probability epsilon, do random action, otherwise do default sampling.'''
        print('epsilon: ', self.epsilon)
        if self.epsilon > np.random.rand():
            return self.random(feats)
        else:
            return self.default(feats)

    @abstractmethod
    def default(self, feats):
        return NotImplementedError

    @abstractmethod
    def random(self, feats):
        raise NotImplementedError

    @abstractmethod
    def choose(self, feats):
        raise NotImplementedError
