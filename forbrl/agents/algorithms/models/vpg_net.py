
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage
from ....utils import build_backbone, build_head, MODELS


@MODELS.register_module
class VPGNet(nn.Module):
    def __init__(self,
                 backbone,
                 head,
                 mean,
                 std,
                 num_rotations=16,
                 size_divisor=32):
        super(VPGNet, self).__init__()

        self.push_color_backbone = build_backbone(backbone)
        self.grasp_color_backbone = build_backbone(backbone)

        self.push_depth_backbone = build_backbone(backbone)
        self.grasp_depth_backbone = build_backbone(backbone)

        self.push_head = build_head(head)
        self.grasp_head = build_head(head)
        self.push_upsample = nn.Upsample(scale_factor=16, mode='bilinear')
        self.grasp_upsample = nn.Upsample(scale_factor=16, mode='bilinear')

        self.mean = mean
        self.std = std
        # self.depth_mean = depth_mean
        # self.depth_std = depth_std
        self.num_rotations = num_rotations
        self.size_divisor = size_divisor

        self.init_affine_mat()
        self.init_weight()

    def init_weight(self): 
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def init_affine_mat(self):
        self.affine_mat = []
        for rotate_idx in range(self.num_rotations):
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                 [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]]
            ).reshape(2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().cuda()

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                 [-np.sin(rotate_theta), np.cos(rotate_theta), 0]]
            ).reshape(2, 3, 1)

            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().cuda()

            self.affine_mat.append([affine_mat_before, affine_mat_after])

    def preprocess(self, x):

        if len(x.shape) == 3:
            x = x[None, ...]

        assert len(x.shape) == 4, (x.shape, x[0].shape)

        # shape (n, h, w, 4)
        x = ndimage.zoom(x, zoom=[1, 2, 2, 1], order=0)

        # Add extra padding (to handle rotations inside network)
        diag_length = float(x.shape[1]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / self.size_divisor) * self.size_divisor
        self.padding_width = int((diag_length - x.shape[1]) / 2)

        x[:, :, :3] /= 255.
        x = (x - self.mean) / self.std

        x = np.pad(
            x, [[0], [self.padding_width], [self.padding_width], [0]],
            'constant', constant_values=0)

        x = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)

        return x.cuda()

    def postprocess(self, x, cpu):
        l = self.padding_width // 2
        r = x[0].shape[2] - self.padding_width // 2

        x = x[0, :, l:r, l:r]

        if cpu:
            x = x.detach().cpu().data.numpy()

        return x

    def forward_single(self, x, color_backbone, depth_backbone,
                       head, upsample, affine_mat_after):

        cx = color_backbone(x[:, :3, ...])
        dx = depth_backbone(x[:, 3:, ...])
        x = torch.cat([cx, dx], dim=1)
        x = head(x)

        flow_grid_after = F.affine_grid(affine_mat_after, x.data.size())
        x = F.grid_sample(x, flow_grid_after, mode='nearest')
        x = upsample(x)

        return x

    def forward(self, x, spec_rot=-1, cpu=False):
        x = self.preprocess(x)

        if spec_rot == -1:
            rot = range(self.num_rotations)
        else:
            rot = [spec_rot]

        push_prob = []
        grasp_prob = []
        # Apply rotations to images
        for rotate_idx in rot:

            affine_mat_before, affine_mat_after = self.affine_mat[rotate_idx]
            flow_grid_before = F.affine_grid(affine_mat_before, x.size())

            # Rotate images clockwise
            rot_x = F.grid_sample(x, flow_grid_before, mode='nearest')

            push_feat = self.forward_single(
                rot_x, self.push_color_backbone, self.push_depth_backbone,
                self.push_head, self.push_upsample,
                affine_mat_after)
            grasp_feat = self.forward_single(
                rot_x, self.grasp_color_backbone, self.grasp_depth_backbone,
                self.grasp_head, self.grasp_upsample,
                affine_mat_after)

            push_prob.append(push_feat)
            grasp_prob.append(grasp_feat)

        push_prob = torch.cat(push_prob, dim=1)
        grasp_prob = torch.cat(grasp_prob, dim=1)

        push_prob = self.postprocess(push_prob, cpu)
        grasp_prob = self.postprocess(grasp_prob, cpu)

        return push_prob, grasp_prob
