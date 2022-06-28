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


@PROJECTIONS.register_module()
class BatchNormProjection(BaseModule):

    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 last_dim,
                 last_bn=True):
        super(BatchNormProjection, self).__init__()

        mlp = list()
        for layer in range(num_layers):
            dim1 = input_dim if layer == 0 else hidden_dim
            dim2 = last_dim if layer == num_layers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if layer < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)
