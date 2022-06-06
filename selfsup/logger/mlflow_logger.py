import os

from selfsup.logger import LOGGERS
from .base import BaseLogger
from .utils import rank_zero_only


@LOGGERS.register_module()
class MlflowLogger(BaseLogger):

    def __init__(self,
                 experiment_name=None,
                 run_name=None,
                 run_id=None,
                 interval=5):
        super(MlflowLogger, self).__init__()

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = run_id
        self.interval = interval
        self.import_mlflow()

    @rank_zero_only
    def import_mlflow(self):
        import mlflow

        self.mlflow = mlflow

    @rank_zero_only
    def before_run(self):

        if self.run_id is not None:
            self.mlflow.start_run(run_id=self.run_id)
        else:
            if self.experiment_name is not None:
                self.mlflow.set_experiment(self.experiment_name)
            self.mlflow.start_run(run_name=self.experiment_name
                                  if self.run_name is None else self.run_name)

    @rank_zero_only
    def log(self, metrics, step):
        if step % self.interval == 0:
            max_tries = 5
            for i in range(1, max_tries + 1):
                try:
                    self.mlflow.log_metrics(metrics, step=step)
                    break
                except Exception as e:
                    print(e)
                    print(f'Retrying ... {i}')

    @rank_zero_only
    def force_log(self, metrics, step):
        max_tries = 5
        for i in range(1, max_tries + 1):
            try:
                self.mlflow.log_metrics(metrics, step=step)
                break
            except Exception as e:
                print(e)
                print(f'Retrying ... {i}')

    @rank_zero_only
    def after_run(self):
        artifact_lists = [
            'model_config.py', 'model_final.pth', 'merges.txt', 'vocab.json'
        ]
        for artifact_list in artifact_lists:
            if os.path.exists(os.path.join(self.work_dir, artifact_list)):

                self.mlflow.log_artifact(
                    os.path.join(self.work_dir, artifact_list),
                    artifact_path='checkpoint')
        self.mlflow.end_run()
