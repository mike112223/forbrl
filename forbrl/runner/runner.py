
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..utils import RUNNERS


@RUNNERS.register_module
class Runner(object):

    def __init__(self,
                 agent,
                 env,
                 workdir,
                 max_iter=10000):
        self.agent = agent
        self.env = env
        self.workdir = workdir
        self.max_iter = max_iter

    def __call__(self,):
        self.run_rl()

    def run_rl(self):
        '''Run the main RL loop until clock.max_frame'''
        state = self.env.reset()
        done = False
        self.agent.save_ckpt()
        self.records = -np.ones((self.max_iter, 2))

        while True:
            if self.agent.iter >= self.max_iter:
                break

            print('===== iter %d =====' % self.agent.iter)

            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)

            if done:  # before starting another episode
                if self.agent.iter < self.max_iter:  # reset and continue
                    state = self.env.reset()
                    done = False
                self.agent.save_ckpt()
                self.plot()
                continue

            self.records[self.agent.iter] = [action['action'] == 'grasp', reward[0]]

            print('action & reward & done: ', action, reward, done)

            loss = self.agent.update(state, action, reward, next_state, done)
            print('loss: ', loss)

            state = next_state

        self.plot()
        np.savetxt(os.path.join(self.workdir, 'records.txt'), self.records)

    def plot(self):
        interval = 200

        plt.figure()
        plt.ylim((0, 1))
        plt.ylabel('Grasping performance (success rate)')
        plt.xlim((0, self.max_iter))
        plt.xlabel('Number of training steps')
        plt.grid(True, linestyle='-', color=[0.8, 0.8, 0.8])

        plt_res = [0] * self.max_iter
        for i in range(len(self.records)):

            # Get indicies for previous x grasps, where x is the interval size
            grasp_idx = np.argwhere(self.records[:, 0] == 1)
            prev_grasp_idx = grasp_idx[np.argwhere(grasp_idx[:, 0] < i)[:, 0]]
            grasp_idx_over_interval = prev_grasp_idx[
                max(0, len(prev_grasp_idx) - interval):len(prev_grasp_idx), 0]

            grasp_success_over_interval = np.sum(
                self.records[grasp_idx_over_interval][:, 1] == 1) / \
                float(min(interval, max(i, 1)))
            if i < interval:
                grasp_success_over_interval *= (float(i) / float(interval))
            plt_res[i] = grasp_success_over_interval

        plt.plot(range(0, self.max_iter), plt_res, linewidth=3)
        plt.savefig(os.path.join(self.workdir, 'grasp_success_%d.png' % self.agent.iter))
