import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import PROJECTIONS


@PROJECTIONS.register_module()
class BaseProjection(BaseModule):

    def __init__(self, input_dim, hidden_dim, last_dim):
        super(BaseProjection, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, last_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
