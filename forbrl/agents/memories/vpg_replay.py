
import operator
from collections import deque

import numpy as np

from .base import Memory
from ...utils import MEMORIES


def batch_get(arr, idxs):
    '''Get multi-idxs from an array depending if it's a python list or np.array'''
    if isinstance(arr, (list, deque)):
        batch = operator.itemgetter(*idxs)(arr)
        if len(idxs) == 1:
            batch = [batch]

        if isinstance(arr[0], str):
            batch = np.array([np.load(open(_, 'rb')) for _ in batch])
        else:
            batch = np.array(batch)

        return batch
    else:
        return arr[idxs]


def sample_next_states(head, max_size, ns_idx_offset, batch_idxs, states, ns_buffer):
    '''Method to sample next_states from states, with proper guard for next_state idx being out of bound'''
    # idxs for next state is state idxs with offset, modded
    ns_batch_idxs = (batch_idxs + ns_idx_offset) % max_size
    # if head < ns_idx <= head + ns_idx_offset, ns is stored in ns_buffer
    ns_batch_idxs = ns_batch_idxs % max_size
    buffer_ns_locs = np.argwhere((head < ns_batch_idxs) & (ns_batch_idxs <= head + ns_idx_offset)).flatten()
    # find if there is any idxs to get from buffer

    to_replace = buffer_ns_locs.size != 0
    if to_replace:
        # extract the buffer_idxs first for replacement later
        # given head < ns_idx <= head + offset, and valid buffer idx is [0, offset)
        # get 0 < ns_idx - head <= offset, or equiv.
        # get -1 < ns_idx - head - 1 <= offset - 1, i.e.
        # get 0 <= ns_idx - head - 1 < offset, hence:
        buffer_idxs = ns_batch_idxs[buffer_ns_locs] - head - 1
        # set them to 0 first to allow sampling, then replace later with buffer
        ns_batch_idxs[buffer_ns_locs] = 0
    # guard all against overrun idxs from offset
    ns_batch_idxs = ns_batch_idxs % max_size
    next_states = batch_get(states, ns_batch_idxs)

    if to_replace:
        # now replace using buffer_idxs and ns_buffer
        buffer_ns = batch_get(ns_buffer, buffer_idxs)
        next_states[buffer_ns_locs] = buffer_ns

    return next_states


@MEMORIES.register_module
class VPGReplay(Memory):
    '''
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    If 'use_cer', sampling will add the latest experience.

    e.g. memory_spec
    "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self,
                 batch_size,
                 max_size,
                 use_cer,
                 alpha=2):
        super().__init__()

        self.batch_size = batch_size
        self.max_size = max_size
        self.use_cer = use_cer
        self.alpha = alpha

        self.batch_idxs = None
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        self.head = -1  # index of most recent experience
        # generic next_state buffer to store last next_states (allow for multiple for venv)
        self.ns_idx_offset = 1
        self.ns_buffer = deque(maxlen=self.ns_idx_offset)
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states',
                          'dones', 'priorities']
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        # set self.states, self.actions, ...
        for k in self.data_keys:
            if k != 'next_states':  # reuse self.states
                # list add/sample is over 10x faster than np, also simpler to handle
                setattr(self, k, [None] * self.max_size)
        self.size = 0
        self.head = -1
        self.ns_buffer.clear()

    def update(self, state, action, reward, next_state, done):
        '''Interface method to update memory'''
        self.add_experience(state, action, reward, next_state, done)

    def add_experience(self, state, action, reward, next_state, done, error=100000):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = state
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.ns_buffer.append(next_state)
        self.dones[self.head] = done
        # Actually occupied size of memory
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1
        # set to_train using memory counters head, seen_size instead of tick since clock will step by num_envs when on venv; to_train will be set to 0 after training step
        # algorithm = self.body.agent.algorithm
        # algorithm.to_train = algorithm.to_train or (self.seen_size > algorithm.training_start_step and self.head % algorithm.training_frequency == 0)

        self.priorities[self.head] = error

    def sample(self):
        '''
        Returns a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}
        '''
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
            if k == 'next_states':
                batch[k] = sample_next_states(
                    self.head, self.max_size, self.ns_idx_offset,
                    self.batch_idxs, self.states, self.ns_buffer)
            else:
                batch[k] = batch_get(getattr(self, k), self.batch_idxs)

        return batch

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''

        cer_action = self.actions[self.head]['action']
        cer_reward = self.rewards[self.head][0]

        if cer_action == 'push':
            sample_reward = 0 if cer_reward == 0.5 else 0.5
        else:
            sample_reward = 0 if cer_reward == 1. else 1.

        sample_idxs = []
        for i in range(self.head):
            if self.actions[i]['action'] == cer_action and \
               self.rewards[i][0] == sample_reward:
                sample_idxs.append(i)

        if len(sample_idxs) > 0:
            batch_idxs = np.zeros(batch_size, dtype=np.int16)
            sample_idxs = np.asarray(sample_idxs)
            sorted_prior_idxs = np.argsort(np.asarray(self.priorities)[sample_idxs])
            sorted_sample_idxs = sample_idxs[sorted_prior_idxs]
            sample_idx = int(np.round(np.random.power(self.alpha, 1) * (len(sample_idxs)-1)))
            sample_idx = sorted_sample_idxs[sample_idx]
            batch_idxs[0] = sample_idx
            print('replay: ', sample_idx)
        else:
            batch_idxs = np.zeros(batch_size - 1, dtype=np.int16)

        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
        return batch_idxs

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = errors
        assert len(priorities) == self.batch_idxs.size
        for idx, p in zip(self.batch_idxs, priorities):
            self.priorities[idx] = p
