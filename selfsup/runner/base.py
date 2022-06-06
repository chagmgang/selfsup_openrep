from argparse import Namespace

import torch

from selfsup.runner import RUNNERS


@RUNNERS.register_module()
class BaseRunner(object):

    def __init__(self, **kwargs):
        super(BaseRunner, self).__init__()

        self.global_step = 0
        self.variable = Namespace(**kwargs)
        keys = [key for key in vars(self.variable).keys()]
        self.max_epochs = self.variable.max_epochs if 'max_epochs' in keys else 1

    def register_datamodule(self, dataloader):
        self.dataloader = dataloader
        self.num_iteration = self.max_epochs * len(self.dataloader)

    def register_model(self, model):
        self.model = model

    def train(self):

        for epoch in range(self.max_epochs):

            if isinstance(self.dataloader.sampler,
                          torch.utils.data.distributed.DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)

            for data in self.dataloader:

                print(data, self.variable.global_rank,
                      self.variable.world_size)
