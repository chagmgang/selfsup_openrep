from argparse import Namespace

import torch
import torch.distributed as dist
from tqdm import tqdm

from selfsup.logger import build_logger
from selfsup.logger.checkpoint import Checkpointer
from selfsup.optim import build_optim, build_sched
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

    def register_optimizer(self, optim):
        self.optimizer = build_optim(params=self.model.parameters(), cfg=optim)

    def register_scheduler(self, sched):
        sched['num_training_steps'] = self.num_iteration
        self.scheduler = build_sched(optim=self.optimizer, cfg=sched)

    def register_logger(self, loggers):
        if not isinstance(loggers, list):
            loggers = [loggers]

        self.loggers = [build_logger(logger) for logger in loggers]
        for logger in self.loggers:
            logger.work_dir = self.checkpointer.work_dir

    def register_checkpoint(self, ckpt, cfg):
        self.checkpointer = Checkpointer(**ckpt)
        self.checkpointer.register_configs(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            cfg=cfg)

    def convert_batch_to_gpu(self, batch):
        batch_gpu = dict()
        for key in batch.keys():
            value = batch[key]
            if isinstance(value, list):
                value = [v.cuda() for v in value]
            else:
                value = value.cuda()
            batch_gpu[key] = value
        return batch_gpu

    def training_step(self, data):
        loss = self.model(data, train=True)
        self.log(
            dict(
                train_loss=float(loss.detach().cpu().numpy()),
                iteration=self.global_step,
                lr=self.optimizer.param_groups[0]['lr'],
            ))
        return loss

    def log(self, metrics):
        for logger in self.loggers:
            logger.log(metrics, self.global_step)

    def before_run(self):
        for logger in self.loggers:
            logger.before_run()

    def after_run(self):
        for logger in self.loggers:
            logger.after_run()

    def train(self):

        self.before_run()

        for epoch in range(self.max_epochs):

            if isinstance(self.dataloader.sampler,
                          torch.utils.data.distributed.DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)

            self.model.train()
            for data in tqdm(self.dataloader, desc=f'epoch:{epoch}'):

                data = self.convert_batch_to_gpu(data)
                self.optimizer.zero_grad()
                loss = self.training_step(data)
                if dist.is_initialized():
                    dist.barrier()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.checkpointer.interval == 0:
                    self.checkpointer.save(self.global_step)

        self.checkpointer.save('model_final')
        self.after_run()
