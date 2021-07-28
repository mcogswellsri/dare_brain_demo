import json
import os
import os.path as pth
import glob


def main():
    train_dir = '/dataSRI/DataSets/BrATS/MICCAI_BraTS2020_TrainingData_train/'
    val_dir = '/dataSRI/DataSets/BrATS/MICCAI_BraTS2020_TrainingData_dev/'


    train_ids = []
    for train_ex_dir in glob.glob(pth.join(train_dir, '*')):
        subj_id = pth.basename(train_ex_dir)
        if not subj_id.startswith('BraTS20'):
            continue
        train_ids.append(subj_id)

    val_ids = []
    for val_ex_dir in glob.glob(pth.join(val_dir, '*')):
        subj_id = pth.basename(val_ex_dir)
        if not subj_id.startswith('BraTS20'):
            continue
        val_ids.append(subj_id)

    fname = 'split_preprocessing/saved_ids.json'
    with open(fname, 'w') as f:
        json.dump({
            'train_ids': train_ids,
            'val_ids': val_ids,
        }, f)


if __name__ == '__main__':
    main()
