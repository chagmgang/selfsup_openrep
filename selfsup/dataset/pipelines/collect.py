from ..build import PIPELINES


@PIPELINES.register_module()
class Collect(object):

    def __init__(self, keys=['img', 'filename']):

        self.keys = keys

    def __call__(self, results):

        new_results = dict()
        for key in self.keys:
            new_results[key] = results[key]

        return new_results
