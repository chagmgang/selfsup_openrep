import torch.nn as nn

from selfsup.models import PROJECTION


@PROJECTION.register_module()
class BaseProjection(nn.Module):

    def __init__(self, input_dim, hidden_dim, last_dim):
        super(BaseProjection, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, last_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
