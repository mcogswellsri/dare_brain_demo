import os
import os.path as pth
import shutil

import brats_demo_server
from brats_example import BraTSExample


def main():
    brats_dir = '/usr/brats/MICCAI_BraTS2020_TrainingData/'
    modality = 't1'
    slice_id = 60
    params = brats_demo_server.params
    with open('dev_ids.txt', 'r') as f:
        dev_ids = [line.strip() for line in f]
    for dev_id in dev_ids:
        exam_dir = pth.join(brats_dir, dev_id)
        params.image_dir = exam_dir + '/'
        example = BraTSExample(params)
        fname = example.image_fname(slice_id, modality)

        out_dir = pth.join(params.thumbnail_dir, dev_id)
        os.makedirs(out_dir, exist_ok=True)
        out_fname = pth.join(out_dir, f'thumbnail.png')
        shutil.copy(fname, out_fname)


if __name__ == '__main__':
    main()
