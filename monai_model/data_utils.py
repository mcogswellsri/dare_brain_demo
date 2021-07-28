import os.path
from glob import glob

import monai
from monai.transforms import (
    MapTransform,
    Compose,
    LoadNiftid,
    AsChannelFirstd,
    Spacingd,
    Orientationd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CenterSpatialCropd,
    ToTensord,
    ConcatItemsd,
    AddChanneld,
    Lambdad,
    DeleteItemsd,
    CopyItemsd,
)
from custom_transforms import RandGenerateRegiond, ConvertToMultiChannelBasedOnBratsClassesd

from brats_dataset import BraTSDataset
from torch.utils.data import DataLoader, ConcatDataset, Subset


def data_has_seg(data_dir):
    # TODO: this could be much cleaner and efficient
    for fname in glob(os.path.join(data_dir, '*', '*.nii.gz')):
        if '_seg.nii.gz' in fname:
            return True
    return False


def load_data(args, vis=False, include_train=True):
    val_has_seg = data_has_seg(args.val_data)
    # pre-processing pipeline for BraTS2020 data
    if args.model == 'unet':
        val_transform = [
            LoadNiftid(keys=['t1', 't2', 'flair', 't1ce', 'seg']),
            AddChanneld(keys=['t1', 't2', 'flair', 't1ce']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            Spacingd(keys=['t1', 't2', 'flair', 't1ce', 'seg'], pixdim=(1.5, 1.5, 2.),
                mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest')),
            Orientationd(keys=['t1', 't2', 'flair', 't1ce', 'seg'], axcodes='RAS'),
            CenterSpatialCropd(keys=['t1', 't2', 'flair', 't1ce', 'seg'], roi_size=[128, 128, 64]),
            ConcatItemsd(keys=['t1', 't2', 'flair', 't1ce'], name='image'),
            DeleteItemsd(keys=['t1', 't2', 'flair', 't1ce']),
        ]
    elif args.model in ['2d_att_unet', '2d_unet']:
        val_transform = [
            LoadNiftid(keys=['t1', 't2', 'flair', 't1ce', 'seg']),
            AddChanneld(keys=['t1', 't2', 'flair', 't1ce']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            ConcatItemsd(keys=['t1', 't2', 'flair', 't1ce'], name='image'),
            DeleteItemsd(keys=['t1', 't2', 'flair', 't1ce']),
        ]
    if vis:
        val_transform.append(CopyItemsd(keys=['image'], times=1, names=['vis_image']))
    val_transform.extend([
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
    ])
    val_transform.extend([ ToTensord(keys=['image', 'seg']) ])
    val_transform = Compose(val_transform)

    # create data loaders
    val_ds = BraTSDataset(root_dir=args.val_data,
                          transform=val_transform, num_workers=0,
                          max_examples=args.max_examples)
    if args.split == 'all':
        # NOTE: not the same as above... this uses val_transform
        train_ds_val_trans = BraTSDataset(root_dir=args.train_data,
                                transform=val_transform, num_workers=0,
                                max_examples=args.max_examples)
        _val_ds = val_ds # TODO: remove
        val_ds = ConcatDataset([val_ds, train_ds_val_trans])
        if args.max_examples and len(val_ds) > args.max_examples:
            indices = list(range(len(val_ds)))
            val_ds = Subset(val_ds, indices[:args.max_examples])
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    if include_train:
        return train_ds, train_loader, val_ds, val_loader
    else:
        return val_ds, val_loader


def resize(img, shape, mode='trilinear'):
    '''
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    '''
    assert len(img.shape) == 4, 'Expects CHWD'
    assert len(shape) == 3, 'Expects just spatial dimensions HWD'
    resize_transform = monai.transforms.Resize(tuple(shape))
    return resize_transform(img, mode=mode)
