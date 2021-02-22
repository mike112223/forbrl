
import os

import torch
import numpy as np

from ..utils import (build_memory, build_algorithm, build_policy,
                     AGENTS, get_pred_vis, save_vis, get_class_name)


@AGENTS.register_module
class VPGAgent(object):
    '''
    Agent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm, memory, body
    '''

    def __init__(self,
                 algorithm,
                 memory,
                 policy,
                 base_explore=0.5,
                 min_explore=0.1,
                 workdir=None,
                 save=True):

        self.algorithm = build_algorithm(algorithm)
        self.memory = build_memory(memory)
        self.policy = build_policy(policy, dict(epsilon=base_explore))

        self.workdir = workdir

        self.base_explore = base_explore
        self.min_explore = min_explore

        self.save = save
        self.iter = 0

        self.dir_name = ['vis', 'ckpt', 'state']
        # self.init_dir()

    def init_dir(self):
        for name in self.dir_name:
            path = os.path.join(self.workdir, name)
            setattr(self, name, path)
            os.makedirs(path, exist_ok=True)

    def act(self, state):
        '''
        Standard act method from algorithm.
        '''
        with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
            feats = self.algorithm.extract_feat(state)
        action = self.policy.choose(feats)

        if self.save:
            grasp_vis = get_pred_vis(
                feats[1], state[:, :, :3].astype(np.uint8), action['best_idx'])
            push_vis = get_pred_vis(
                feats[0], state[:, :, :3].astype(np.uint8), action['best_idx'])
            save_vis(self.vis, self.iter, grasp_vis, 'grasp')
            save_vis(self.vis, self.iter, push_vis, 'push')

        return action

    def sample(self):
        '''
        Samples a batch from memory of size self.memory_spec['batch_size']
        '''
        batch = self.memory.sample()
        return batch

    def update(self, state, action, reward, next_state, done):
        '''
        Update per timestep after env transitions, e.g. memory, algorithm,
        update agent params, train net
        '''
        self.iter += 1
        state_path = self.save_state(state)
        self.memory.update(state_path, action, reward, next_state, done)
        batch = self.sample()
        loss, error = self.algorithm.train(batch)
        self.explore_update()

        if 'prioritized' in get_class_name(self.memory):
            self.memory.update_priorities(error)

        return loss

    def explore_update(self):
        '''Update the agent after training'''
        self.policy.epsilon = max(
            self.base_explore * np.power(0.9998, self.iter),
            self.min_explore)

    def save_ckpt(self):
        '''Save agent'''
        torch.save(
            self.algorithm.model.cpu().state_dict(),
            os.path.join(self.ckpt, 'snapshot-%06d.pth' % (self.iter))
        )
        self.algorithm.model = self.algorithm.model.cuda()

    def save_state(self, state):
        path = os.path.join(self.state, 'state-%06d.npy' % (self.iter))
        np.save(path, state)
        return path

    def close(self):
        '''Close and cleanup agent at the end of a session, e.g. save model'''
        self.save_ckpt()
