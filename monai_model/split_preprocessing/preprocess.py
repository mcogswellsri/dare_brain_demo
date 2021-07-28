import cv2
import os
import pdb
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

# This script takes the raw BraTS 2020 data (INPUT_ROOT) and breaks each image
# into jpg slices, saving the result to the OUTPUT_ROOT directory for loading
# by a dataloader.

INPUT_ROOT = '/dataSRI/DataSets/BrATS/MICCAI_BraTS2020_TrainingData_dev/'
OUTPUT_ROOT = './data/BraTS2020_Train_dev/'

L0 = 0      # Background
L1 = 50     # Necrotic and Non-enhancing Tumor
L2 = 100    # Edema
L3 = 150    # Enhancing Tumor

# MRI Image channels Description
# ch0: FLAIR / ch1: T1 / ch2: T1c/ ch3: T2
# cf) In this project, we use FLAIR and T1c MRI dataset
# 
# Data Load Example
#img = nib.load(IMG_PATH)
#img = (img.get_fdata())[:,:,:,3]                # img shape = (240,240,155)


# MRI Label Channels Description
# 0: Background         / 1: Necrotic and non-enhancing tumor (paper, 1+3)
# 2: edema (paper, 2)   / 3: Enhancing tumor (paper, 4)
# 
# <Input>           <Prediction>
# FLAIR             Complete(1,2,3)
# FLAIR             Core(1,3)
# T1c               Enhancing(3)
#
# Data Load Example
# label = nib.load(LABEL_PATH)
# label = (label.get_fdata()).astype(np.uint16)   # label shape = (240,240,155)


def nii2jpg_img(img_path, output_root):
    img_name = (img_path.split('/')[-1]).split('.')[0]
    img = nib.load(img_path)
    img = img.get_fdata()
    img = (img/img.max())*255
    img = img.astype(np.uint8)

    for i in range(img.shape[2]):
        filename = os.path.join(output_root, f'slice{i:0>3d}.jpg')
        gray_img = img[:,:,i]
        #color_img = np.expand_dims(gray_img, 3)
        #color_img = np.concatenate([color_img, color_img, color_img], 2)

        # COLOR LABELING
        #c255 = np.expand_dims(np.ones(gray_img.shape)*255, 3)
        #c0 = np.expand_dims(np.zeros(gray_img.shape), 3)
        #color = np.concatenate([c0,c0,c255], 2)
        #color_img = color_img.astype(np.float32) + color
        #color_img = (color_img / color_img.max()) *255

        cv2.imwrite(filename, gray_img)


def nii2jpg_label(img_path, output_root):
    img_name = (img_path.split('/')[-1]).split('.')[0]
    img = nib.load(img_path)
    img = img.get_fdata()
    img = img*50
    img = img.astype(np.uint8)

    for i in range(img.shape[2]):
        filename = os.path.join(output_root, f'{i:0>3d}.jpg')
        gray_img = img[:,:,i]
        #color_img = np.expand_dims(gray_img, 3)
        #color_img = np.concatenate([color_img, color_img, color_img], 2)

        # COLOR LABELING
        #c255 = np.expand_dims(np.ones(gray_img.shape)*255, 3)
        #c0 = np.expand_dims(np.zeros(gray_img.shape), 3)
        #color = np.concatenate([c0,c0,c255], 2)
        #color_img = color_img.astype(np.float32) + color
        #color_img = (color_img / color_img.max()) *255

        cv2.imwrite(filename, gray_img)




subj_ids = pd.read_csv(os.path.join(INPUT_ROOT, 'name_mapping.csv'))['BraTS_2020_subject_ID']

for subj_id in tqdm(subj_ids):
    # allows name_mapping to contain a superset of the set of examples in this directory
    if not os.path.exists(os.path.join(INPUT_ROOT, subj_id)):
        continue
    for modality in ['flair', 't1', 't1ce', 't2', 'seg']:
        output_dir = os.path.join(OUTPUT_ROOT, modality, subj_id) 
        os.makedirs(output_dir, exist_ok=True)
        input_img_path = os.path.join(INPUT_ROOT, subj_id, f'{subj_id}_{modality}.nii.gz')
        # labels
        if modality == 'seg':
            nii2jpg_label(input_img_path, output_dir)
        # input modalities
        else:
            nii2jpg_img(input_img_path, output_dir)
