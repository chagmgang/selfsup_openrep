from mmrotate.datasets import DOTADataset
from mmrotate.datasets.builder import DATASETS


@DATASETS.register_module()
class DOTA2_0Dataset(DOTADataset):

    CLASSES = ('harbor', 'swimming-pool', 'roundabout', 'bridge',
               'baseball-diamond', 'ground-track-field', 'soccer-ball-field',
               'storage-tank', 'basketball-court', 'large-vehicle', 'plane',
               'tennis-court', 'ship', 'helicopter', 'helipad',
               'container-crane', 'airport', 'small-vehicle')

    PALETTE = [(248, 187, 150), (74, 31, 98), (236, 61, 237), (10, 213, 74),
               (201, 222, 55), (151, 200, 85), (79, 202, 164), (178, 60, 194),
               (32, 228, 30), (38, 100, 75), (235, 8, 205), (98, 16, 96),
               (156, 0, 206), (243, 202, 242), (148, 209, 239), (126, 67, 200),
               (247, 19, 181), (86, 16, 31)]
