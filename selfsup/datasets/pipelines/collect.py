import numpy as np
import torchvision

from ..builder import PIPELINES


@PIPELINES.register_module()
class Collect(object):

    def __init__(self, keys=['img', 'filename']):

        self.keys = keys
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, results):

        new_results = dict()
        for key in self.keys:
            if isinstance(results[key], np.ndarray):
                new_results[key] = self.to_tensor(results[key])
            else:
                new_results[key] = results[key]

        return new_results
