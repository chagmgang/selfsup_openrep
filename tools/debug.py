from selfsup import Config
from selfsup.dataset import build_dataset

cfg_dict = dict(type='TestDataset')

cfg = Config(dict(data=cfg_dict))
dataset = build_dataset(cfg.data)
for i in range(len(dataset)):
    print(dataset[i])
