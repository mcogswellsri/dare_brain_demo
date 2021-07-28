import os
import os.path as pth
import pandas as pd
import random
from glob import glob

import argparse

parser = argparse.ArgumentParser(description='''
This script splits the BraTS2020 Training data into two splits so the 2nd
can be used for development/validation. It does this by creating two new directories,
one for each split. The directory ending in _train/ will contain the first split
and the directory ending in _dev/ will contain the 2nd split.
The two directories each mirror the format of the
MICCAI_BraTS2020_TrainingData/ directory so they can be fed directly
to existing preprocessing code. These new directories will contain sym links
to the relevant data.
''')
parser.add_argument("--root", type=str,
                    help="Location of MICCAI_BraTS2020_TrainingData/.")
parser.add_argument("--output_dir", type=str,
                    help="Where to put the new split directories.")
parser.add_argument("--ratio", type=float, default=0.2,
                    help="Percentage of examples to put in dev/split2")





def main(args):
    # read ids and generate splits
    name_csv = os.path.join(args.root, 'name_mapping.csv')
    subj_ids = pd.read_csv(name_csv)['BraTS_2020_subject_ID']
    subj_ids = sorted(list(subj_ids))
    split1_ids = []
    split2_ids = []
    random.seed(9)
    for subj_id in subj_ids:
        if random.random() < args.ratio:
            split2_ids.append(subj_id)
        else:
            split1_ids.append(subj_id)

    # figure out where to put everything
    basename = pth.basename(pth.normpath(args.root))
    split1_root = pth.join(args.output_dir, f'{basename}_train')
    try:
        os.makedirs(split1_root)
    except FileExistsError as e:
        print(f'Remove {split1_root} and start over')
        raise e
    os.symlink(name_csv, pth.join(split1_root, 'name_mapping.csv'))
    split2_root = pth.join(args.output_dir, f'{basename}_dev')
    try:
        os.makedirs(split2_root)
    except FileExistsError as e:
        print(f'Remove {split2_root} and start over')
        raise e
    os.symlink(name_csv, pth.join(split2_root, 'name_mapping.csv'))

    # create split1 directory structure
    for subj_id in split1_ids:
        src_subj_dir = pth.join(args.root, subj_id)
        dst_subj_dir = pth.join(split1_root, subj_id)
        os.makedirs(dst_subj_dir)
        for img_file in glob(pth.join(src_subj_dir, '*')):
            img_file_name = pth.basename(img_file)
            src = img_file
            dst = pth.join(dst_subj_dir, img_file_name)
            os.symlink(src, dst)
    
    # create split2 directory structure
    for subj_id in split2_ids:
        src_subj_dir = pth.join(args.root, subj_id)
        dst_subj_dir = pth.join(split2_root, subj_id)
        os.makedirs(dst_subj_dir)
        for img_file in glob(pth.join(src_subj_dir, '*')):
            img_file_name = pth.basename(img_file)
            src = img_file
            dst = pth.join(dst_subj_dir, img_file_name)
            os.symlink(src, dst)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
