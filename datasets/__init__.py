import os

from torch.utils.data import ConcatDataset
from .gopro import GOPRO_Train
from .bsergb import BSERGB_Train

def get_dataset(args):
    name = args.dataset
    datasets = []
    if name == 'gopro':
        for scene in os.listdir("/DATASSD1/GOPRO/train"):
            datasets.append(GOPRO_Train(args.voxel_bins, scene, 31))
    if name == 'bsergb':
        for skip in [1, 3]:
            for scene in os.listdir("/DATASSD1/BSERGB/3_TRAINING"):
                datasets.append(BSERGB_Train(args.voxel_bins, "/DATASSD1/BSERGB/3_TRAINING", scene, skip))
    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)