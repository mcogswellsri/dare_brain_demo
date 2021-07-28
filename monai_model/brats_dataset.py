from typing import Any, Callable

import sys
import os
import glob

from monai.data import CacheDataset
from monai.transforms import Randomizable, LoadNiftid

class BraTSDataset(Randomizable, CacheDataset):
    """
    This Dataset loads data from the BraTS2020 dataset.
    Args:
        root_dir: BraTS root directory like ./MICCAI_BraTS2020_TrainingData_dev/
        transform: transforms to execute operations on input data. the default transform is `LoadNiftid`,
            which can load Nifit format data into numpy array with [H, W, D] or [H, W, D, C] shape.
            for further usage, use `AddChanneld` or `AsChannelFirstd` to convert the shape to [C, H, W, D].
        seed: random seed
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            if 0 a single thread will be used. Default is 0.
    Raises:
        ValueError: root_dir must be a directory.
    """

    def __init__(self, root_dir: str,
        transform: Callable[..., Any] = LoadNiftid(["t1", "t2", "flair", "t1ce", "seg"]),
        seed: int = 0,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        max_examples=None,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("root_dir must be a directory.")
        self.set_random_state(seed=seed)

        # Get a list of files for all modalities individually                        
        subj_paths = sorted(glob.glob(f'{root_dir}/*/'))

        # TODO: remove
        filter_ids = [
            'BraTS20_Training_044',
            'BraTS20_Training_003',
            'BraTS20_Training_046',
            'BraTS20_Training_053',
        ]
        new = []
        for subj_path in subj_paths:
            subj_id = os.path.split(os.path.normpath(subj_path))[-1]
            if subj_id not in filter_ids:
                continue
            new.append(subj_path)
        #subj_paths = new
        # TODO: end remove block


        if max_examples:
            subj_paths = subj_paths[:max_examples]
        self.subj_paths = subj_paths
        data = []
        for subj_path in subj_paths:
            subj_id = os.path.split(os.path.normpath(subj_path))[-1]
            data.append({
                'subj_id': subj_id,
                't1': os.path.join(root_dir, subj_id, f'{subj_id}_t1.nii.gz'),
                't2': os.path.join(root_dir, subj_id, f'{subj_id}_t2.nii.gz'),
                'flair': os.path.join(root_dir, subj_id, f'{subj_id}_flair.nii.gz'),
                't1ce': os.path.join(root_dir, subj_id, f'{subj_id}_t1ce.nii.gz'),
                'seg': os.path.join(root_dir, subj_id, f'{subj_id}_seg.nii.gz'),
            })

        super().__init__(data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def randomize(self) -> None:
        self.rann = self.R.random()
