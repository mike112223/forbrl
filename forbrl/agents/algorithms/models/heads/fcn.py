
import torch.nn as nn

from ..utils import (ConvModule,
                     kaiming_init)
from .....utils import HEADS


@HEADS.register_module
class FCN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 order=('conv', 'norm', 'act'),
                 **kwargs):
        super(FCN, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_layers = len(self.out_channels)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.order = order

        self.init_layers()

    def init_layers(self):
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                ConvModule(
                    self.in_channels[i],
                    self.out_channels[i],
                    1,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    order=self.order))

    def init_weights(self):
        for m in self.convs:
            kaiming_init(m.conv)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
