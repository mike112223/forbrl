
import numpy as np
import torch

from .base import Algorithm
from ...utils import build_model, build_criterion, build_optimizer, ALGORITHMS


@ALGORITHMS.register_module
class DQN(Algorithm):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 gamma=0.5,
                 mode='train'):
        self.model = build_model(model).cuda()
        self.criterion = build_criterion(criterion).cuda()
        self.optimizer = build_optimizer(
            optimizer, dict(params=self.model.parameters()))
        self.gamma = gamma
        self.mode = mode

        if self.mode == 'train':
            self.model.train()

        # import pdb
        # pdb.set_trace()

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        # TODO mini batch
        states = batch['states']
        next_states = batch['next_states']
        actions = batch['actions']
        reward, changed, grasp_res = batch['rewards']

        rot = actions['best_idx'][0]
        pos = actions['best_idx'][1:]

        if actions['action'] == 'grasp':
            rots = [(rot + self.model.num_rotations // 2) % self.model.num_rotations, rot]
        else:
            rots = [rot]

        for rot in rots:

            q_preds = self.model(states, rot)
            with torch.no_grad():
                next_q_preds = self.model(next_states)

            q_pred = q_preds[0] if actions['action'] == 'push' else q_preds[1]
            q_pred = q_pred[0]

            act_q_pred = q_pred[tuple(pos)]
            print('max, act:', q_pred.max().detach(), act_q_pred.detach())
            max_next_q_preds = torch.cat(next_q_preds, dim=1).max()
            # max_next_q_preds = np.max(np.concatenate(next_q_preds, axis=0))
            max_q_targets = reward * changed + (changed or grasp_res) * \
                (self.gamma * (1 - batch['dones']) * max_next_q_preds)

            print('value, pred, changed, res: ', act_q_pred.detach(), max_q_targets.detach(), changed, grasp_res)
            q_loss = self.criterion(act_q_pred, max_q_targets)

            error = (max_q_targets - act_q_pred.detach()).abs().cpu().numpy()

            q_loss.backward()

        return q_loss, error

    def extract_feat(self, state, cpu=True):
        return self.model(state, cpu=cpu)

    def divide_batch(self, batch):
        samples = [dict() for _ in range(len(batch['states']))]
        for k in batch:
            for i, sample in enumerate(samples):
                if 'states' in k:
                    samples[i][k] = batch[k][None, i]
                else:
                    samples[i][k] = batch[k][i]

        return samples

    def train(self, batch):
        '''
        Completes one training step for the agent if it is time to train.
        Otherwise this function does nothing.
        '''
        if self.mode == 'train':
            samples = self.divide_batch(batch)
            print('sample num: ', len(samples))
            for sample in samples:
                self.optimizer.zero_grad()
                loss, error = self.calc_q_loss(sample)
                self.optimizer.step()
            return loss.item(), error
        else:
            return np.nan, 0.
