import tempfile
import argparse
import os
import os.path as pth
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import nibabel as nib
import skimage.transform
import skimage.io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import colorsys
import seaborn as sns

import scipy.spatial.distance as distance

gray_cmap = matplotlib.cm.get_cmap('gray')

import sys
sys.path.append('./monai_model/')
from data_utils import load_data

modality_to_idx = {
    't1': 0,
    't2': 1,
    't2flair': 2,
    't1ce': 3,
}

class BraTSExample:
    def __init__(self, params):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        self.tmpdir = self.make_single_example_dataset_dir(params.image_dir)
        args.val_data = self.tmpdir
        args.max_examples = 1
        args.split = None
        args.model = params.model
        self.model = params.model
        val_ds, val_loader = load_data(args, include_train=False, vis=True)
        self.val_ds = val_ds
        assert len(val_ds) == 1
        # includes the pre-processing pipeline
        self.val_loader = val_loader
        # after pre-processing
        self.example = next(iter(self.val_loader))

        self.shape = self.example['image'].shape
        self.slice_ids = list(map(str, range(0, self.shape[-1], 2)))
        self.modality_to_idx = {
            't1': 0,
            't2': 1,
            't2flair': 2,
            't1ce': 3,
        }
        self.cache_dir = tempfile.mkdtemp()
        self.pred = None
        self.pred_thresh = 0.5

        self.random_hash = pth.basename(params.image_dir.rstrip('/'))

        self.MAX_CONST = 2000


    def make_single_example_dataset_dir(self, example_dir):
        tmpdir = tempfile.mkdtemp()
        assert example_dir.endswith('/'), 'needed to get the right dname'
        dname = pth.basename(pth.dirname(example_dir))
        os.symlink(example_dir, pth.join(tmpdir, dname))
        print(example_dir, pth.join(tmpdir, dname))
        return tmpdir

    def list_slices(self):
        return self.slice_ids

    def image_fname(self, img_id, modality):
        slice_idx = int(img_id)
        modality_idx = self.modality_to_idx[modality]
        fname = f'slice_{modality}_{slice_idx}_{self.random_hash}.png'
        fname = pth.join(self.cache_dir, fname)
        if not pth.exists(fname):
            slice = self.example['vis_image'][0, modality_idx, :, :, slice_idx]
            plt.imsave(fname, slice, cmap=gray_cmap)
        return fname

    def gt_fname(self, slice_id, label):
        slice_idx = int(slice_id)
        label_idx = {
            'ET': 0,
            'TC': 1,
            'WT': 2,
        }[label]
        fname = f'gt_{slice_idx}_{label}_{self.random_hash}.png'
        fname = pth.join(self.cache_dir, fname)
        if not pth.exists(fname):
            ground_truth = self.example['seg'][0].cpu().numpy()
            plt.imsave(fname, seg_cmap(ground_truth[label_idx, :, :, slice_idx], label_idx, colorscheme='gt'))
        return fname

    def pred_fname(self, slice_id, label, colorscheme='pred'):
        slice_idx = int(slice_id)
        label_idx = {
            'ET': 0,
            'TC': 1,
            'WT': 2,
            'CSF': 3,
            'GM': 4,
            'WM': 5,
        }[label]
        if label_idx > 2:
            assert self.model == '2d_unet'
        fname = f'pred_{slice_idx}_{label}_{self.random_hash}.png'
        fname = pth.join(self.cache_dir, fname)
        if not pth.exists(fname):
            seg = self.get_y_pred(slice_idx, label_idx)
            plt.imsave(fname, seg_cmap(seg, label_idx, colorscheme))
        return fname

    def counter_fname(self, slice_id, label, counter_input, modality, colorscheme='counterfactual'):
        slice_idx = int(slice_id)
        label_idx = {
            'ET': 0,
            'TC': 1,
            'WT': 2,
        }[label]
        fname = f'counter_inpaint_{slice_idx}_{label}_{modality}_{self.random_hash}.png'
        fname = pth.join(self.cache_dir, fname)
        if not pth.exists(fname):
            if self.model == 'unet':
                raise Exception('not supported')
            elif self.model == '2d_att_unet':
                key = slice_id
                img, counter_slice_id = counter_input
                assert counter_slice_id == slice_id
                seg = self.predict_slice_attunet(img)
                seg = seg[label_idx]
            elif self.model == '2d_unet':
                key = slice_id
                img, counter_slice_id = counter_input
                assert counter_slice_id == slice_id
                seg = self.predict_slice_2dunet(img)
                seg = seg[label_idx]
            plt.imsave(fname, seg_cmap(seg, label_idx, colorscheme))
        return fname


    def predict_slice_2dunet(self, img):
        if self.model != '2d_unet':
            raise Exception('Not using the right model. Should use the unet '
                            'from demo_unet/')
        img = img.to(self.unet_2d.device)
        logits = self.unet_2d.unet(img)
        probs = logits # TODO: why? F.sigmoid(logits)
        probs = probs[0] # remove batch dimension

        # 'ET', 'TC', 'WT', 'CSF', 'GM', 'WM'
        seg = (probs > 0.5)
        return seg.cpu().numpy()


    def predict_slice_attunet(self, img):
        if self.model != '2d_att_unet':
            raise Exception('Not using a 2d segmentation model. Predict as '
                            'one volume during initialization.')
        img = img.to(self.attunet.device)
        probs = self.attunet.unet(img)
        probs = probs[0] # remove batch dimension

        # 'Background', 'NCR/NET' , 'ED' , 'ET'
        seg = (probs > 0.3)

        # 'NCR/NET' , 'ED' , 'ET'
        # l1, l2, l4
        seg = seg[1:]

        # see https://www.med.upenn.edu/cbica/brats2020/data.html and for mapping
        # reverse conversion:
        # l2 = WT - TC                                                               
        # l1 = TC - ET                                                               
        # l4 = ET
        # forward conversion:
        # ET = l4
        # TC = l1 + ET
        # WT = l2 + TC
        # indices in seg after removing background dim:
        # l4 - 2
        # l2 - 1
        # l1 - 0

        ET = seg[2]
        TC = seg[0] + ET
        WT = seg[1] + TC

        # [3, 240, 240] w/ first 3 ET,TC,WT in 1-hot float32
        seg = torch.stack([ET, TC, WT])
        seg = seg.to(torch.float).to('cpu').numpy()
        # then, select the right modality to create [240, 240]... pass that to seg_cmap
        return seg

    def counterfactual(self, slice_id, label, counter_input):
        slice_idx = int(slice_id)
        label_idx = {
            'ET': 0,
            'TC': 1,
            'WT': 2,
        }[label]

        # ground truth segmentation
        gt = self.example['seg'][0].cpu().numpy()
        gt = gt[label_idx, :, :, slice_idx]
        img_shape = gt.shape
        gt = gt.flatten()

        # predicted segmentation
        # TODO: fix the root problem... going into eval mode results in weird
        # outputs... but there is stochasticity in the network, so 
        torch.random.manual_seed(8)
        pred = self.get_y_pred(slice_idx, label_idx)
        pred = pred.flatten()

        # counterfactual segmentation
        img, counter_slice_id = counter_input
        assert counter_slice_id == slice_id
        if self.model == 'unet':
            raise Exception('not supported')
        elif self.model == '2d_att_unet':
            counter = self.predict_slice_attunet(img)
        elif self.model == '2d_unet':
            counter = self.predict_slice_2dunet(img)
        counter = counter[label_idx]
        counter = counter.flatten()

        assert set(list(np.unique(gt)) + list(np.unique(pred)) + list(np.unique(counter))) <= {0., 1.}

        if gt.sum() != 0:
            before_dice = distance.dice(gt, pred)
            after_dice = distance.dice(gt, counter)
            diff = after_dice - before_dice
        else:
            before_dice = 0
            after_dice = 0
            diff = 0

        cmap = plt.cm.get_cmap('coolwarm')
        norm = matplotlib.colors.Normalize(vmin=-1., vmax=1., clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        diff_rgba = mapper.to_rgba(diff)
        stats = {
            'before_dice': before_dice,
            'after_dice': after_dice,
            'diff_dice': diff,
            'diff_color': matplotlib.colors.rgb2hex(diff_rgba),
        }

        # there's no spatial information, but still show as a (uniform) spatial heatmap
        heatmap = np.ones(img_shape)
        # diff in [-1, 1]
        heatmap *= diff
        # center at 0.5 in [0, 1] to meet imsave expectations
        heatmap = (heatmap + 1) / 2

        fname = f'counter_heatmap_{slice_idx}_{label}_{self.random_hash}.png'
        fname = pth.join(self.cache_dir, fname)
        # plot with diverging map centered at 0.5
        plt.imsave(fname, heatmap, vmin=0, vmax=1, cmap=cmap)
        return fname, stats


    def predict(self):
        if self.model != 'unet':
            raise Exception('Not using a 3d segmentation model. Predict as '
                            'slices are requested instead of once beforehand '
                            'as performed here.')
        self.net.eval()
        x, y, batch = prepare_batch(self.example)
        y_pred = self.net(x)
        y_pred_probs = torch.sigmoid(y_pred)
        pred_probs = y_pred_probs.cpu().detach().numpy()
        preds = (pred_probs > self.pred_thresh).astype(pred_probs.dtype)
        self.pred_probs = pred_probs
        self.preds = preds

        if 'seg' in self.example:
            self.example['y'] = self.example['seg']
        self.example['y_pred'] = (y_pred_probs > self.pred_thresh).to(y_pred_probs)
        self.example['y_pred_probs'] = y_pred_probs

    def get_y_pred(self, slice_idx, label_idx=None):
        if self.model == 'unet':
            # TODO: should be self.preds, but make sure this doesn't mess up caching
            if self.pred is None:
                self.predict()
            seg = self.preds[0, :, :, :, slice_idx]
        elif self.model == '2d_att_unet':
            img = self.example['vis_image'][:, :, :, :, slice_idx]
            seg = self.predict_slice_attunet(img)
        elif self.model == '2d_unet':
            img = self.example['vis_image'][:, :, :, :, slice_idx]
            seg = self.predict_slice_2dunet(img)
        if label_idx is not None:
            seg = seg[label_idx]
        return seg

    def get_box_from_tumor(self, slice_id, tumor_type):
        label_idx = {
            'ET': 0,
            'TC': 1,
            'WT': 2,
        }[tumor_type]
        slice_idx = int(slice_id)

        ground_truth = self.example['seg'][0].cpu().numpy()
        ground_truth = ground_truth[label_idx, :, :, slice_idx]
        # no label
        if (ground_truth == 0).all():
            return None
        col_inds = np.where(ground_truth.sum(axis=0))[0]
        x = col_inds.min()
        w = (col_inds.max() + 1) - x
        row_inds = np.where(ground_truth.sum(axis=1))[0]
        y = row_inds.min()
        h = (row_inds.max() + 1) - y

        img_height, img_width = ground_truth.shape
        box = [x/img_width, y/img_height, w/img_width, h/img_height]
        return box

    def preprocess_for_inpainter(self, slice_id, modality):
        slice_idx = int(slice_id)

        eps = 1e-5
        slice_size = (256, 256)

        nifti_file = self.val_ds.data[0][modality]

        try:
            vol = nib.load(nifti_file)
        except nib.filebasedimages.ImageFileError as e:
            warnings.warn(str(e))

        vol = vol.get_data()
        # normalize by constant
        img_max = self.MAX_CONST
        vol = np.clip(vol, 0, self.MAX_CONST)
        if vol.max() > img_max:
            warnings.warn(f'img max is {vol.max()}, but normalizing by {img_max}')
        # normalize by individual max
        #img_max = vol.max() + eps

        img = vol[:, :, slice_idx]
        img = img.astype(np.float)
        img = img / img_max
        img = skimage.transform.resize(img, slice_size)
        assert len(img.shape) == 2, 'assumes single channel for now'
        img = np.tile(img[:, :, None], 3)

        return img

        #img = (img * 255).astype(np.uint8)
        #fname = f'in_inpaint_{slice_idx}_{self.random_hash}.png'
        #fname = pth.join(self.cache_dir, fname)
        #skimage.io.imsave(fname, img)
        #return fname


    def deprocess_from_inpainter(self, inpainted_img):

        img = inpainted_img.astype(np.float)
        img = img * self.MAX_CONST
        return img


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def scale_lightness(rgb, scale_l):
    # scale_l from 0 to 2
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


def get_seg_color(label_idx, colorscheme, with_alpha=True):
    # custom colormap with different color for each label
    scale = {
        'gt': 0.7,
        'pred': 1.0,
        'counterfactual': 1.3,
    }[colorscheme]
    # ET, TC, WT
    if label_idx == 0:
        # c69255
        color = (198/256, 146/256, 85/256)
    elif label_idx == 1:
        # a291e1
        color = (162/256, 145/256, 225/256)
        if colorscheme == 'counterfactual':
            scale = 1.15
    elif label_idx == 2:
        # 5ea5c5
        color = (94/256, 165/256, 197/256)
    # CSF, GM, WM
    elif label_idx == 3:
        # DB162F
        color = (219/256, 22/256, 47/256)
        scale = scale + .3
    elif label_idx == 4:
        # 383961
        color = (56/256, 57/256, 97/256)
        scale = scale + .3
    elif label_idx == 5:
        # 38423B
        color = (56/256, 66/256, 59/256)
        scale = scale + .3
    color = scale_lightness(color, scale)
    if with_alpha:
        color += (1,)
    return color


def seg_cmap(seg, label_idx, colorscheme):
    assert set(np.unique(seg)).issubset(set([0., 1.]))

    color = get_seg_color(label_idx, colorscheme)
    colors = np.array([
        [0, 0, 0, 0],
        color,
    ])
    cmap = ListedColormap(colors)

    # normal color map
    #cmap = matplotlib.cm.get_cmap('viridis')

    img = cmap(seg)

    # find background mask
    mask = (seg == 0)
    mask = mask[:, :, None].repeat(4, axis=2)
    mask[:, :, :3] = False

    # set background alpha to 0
    img[mask] = 0.
    return img


def prepare_batch(batch):
    batch = dict(batch)
    #for k, tensor in batch.items():
    #    if torch.is_tensor(tensor):
    #        batch[k] = tensor.to(device)
    x = batch['image']
    if "seg" in batch:
        seg = batch["seg"]
    else:
        seg = None
    return x, seg, batch
