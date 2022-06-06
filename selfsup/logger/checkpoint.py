import json
import os

import torch

from selfsup.utils import get_root_logger, print_log
from .utils import rank_zero_only


class Checkpointer(object):

    def __init__(self, interval=4000, work_dir='work_dir', save_to_disk=True):

        self.work_dir = work_dir
        self.save_to_disk = save_to_disk
        self.interval = interval

    @rank_zero_only
    def register_configs(self, cfg, model, optimizer, scheduler):

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        os.makedirs(self.work_dir, exist_ok=True)
        cfg.dump(os.path.join(self.work_dir, 'model_config.py'))

    @rank_zero_only
    def save(self, name, **kwargs):
        data = {}
        data['state_dict'] = self.model.state_dict()
        data['optimizer'] = self.optimizer.state_dict()
        data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.work_dir, f'{name}.pth')
        torch.save(data, save_file)

    def load_state_dict(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False)
        if missing_keys:
            missing_keys_msg = 'missing keys : \n'
            missing_keys_msg += json.dumps(missing_keys)
            print_log(missing_keys_msg, logger=get_root_logger())
        if unexpected_keys:
            unexpected_keys_msg = 'unexpected keys : \n'
            unexpected_keys_msg += json.dumps(unexpected_keys)
            print_log(unexpected_keys_msg, logger=get_root_logger())
        return model
