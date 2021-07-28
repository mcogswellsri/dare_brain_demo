import os
import glob
import pickle as pkl

import torch
import numpy as np
import nibabel as nib
import scipy.stats
from tqdm import tqdm

import eval
import layers_list
from custom_transforms import brats_label_to_raw

from monai.losses import DiceLoss

import skimage.morphology as morph

class VisNotFoundError(Exception):
    pass

def get_file(pattern):
    files = list(glob.glob(pattern))
    if len(files) != 1:
        raise VisNotFoundError(f'not found {pattern}')
    return files[0]

#_cache = {}
def load_vis(vis_key, layer_key, ex_id, args):
    key = (vis_key, layer_key, ex_id)
    #if key in _cache:
    #    return _cache[key]

    vis_dir = os.path.join(args.model_dir,
                           f'an_vis_niftis_{vis_key}_{layer_key}',
                           'result_dataset',
                           ex_id)
    vis_name = eval.vis_key_name_mapping[vis_key]
    layer_name = dict(layers_list.unet_layers)[layer_key]
    vis_fname = get_file(os.path.join(vis_dir,
        f'*_{vis_name.replace(" ", "-")}_{layer_name.replace(" ", "-")}.nii.gz'))
    vis = nib.load(vis_fname).get_fdata().transpose((3, 0, 1, 2))
    # BCHWD
    vis = torch.tensor(vis[None])
    #_cache[key] = vis
    return vis


def find_border(seg):
    assert len(seg.shape) == 3
    result = morph.dilation(seg, morph.cube(3))
    return np.maximum(0, result - seg)



# overlap with background
with open('background_mask.pkl', 'rb') as f:
    background_mask = pkl.load(f)
background_mask = torch.tensor(background_mask[None, None])


_bkg_cache = {}
def get_background_like(map1):
    key = map1.shape, map1.dtype, map1.device
    if key in _bkg_cache:
        return _bkg_cache[key]
    result = background_mask.expand(*map1.shape).to(map1)
    _bkg_cache[key] = result
    return result



def _single_rank_compare(img1, img2):
    assert len(img1.shape) == 5
    assert len(img2.shape) == 5
    assert img1.shape[0] == 1
    assert img1.shape[1] == img2.shape[1]
    result = []
    for c in range(img1.shape[1]):
        corr = scipy.stats.spearmanr(img1[0, c].numpy().flatten(),
                                     img2[0, c].numpy().flatten())
        result.append(corr)
    result = [result]
    return torch.tensor(result).to(img1)


def rank_compare(img1, img2):
    result = _single_rank_compare(img1, img2)
    result = result.to(img1)
    result_back = _single_rank_compare(get_background_like(img1), img2)
    result_back = result_back.to(img1)
    assert len(result.shape) == 3
    return torch.stack([result, result_back], dim=3)


_full_cache = {}
def _full_like(map1, val):
    key = map1.shape, map1.dtype, map1.device, val
    if key in _full_cache:
        return _full_cache[key]
    result = torch.full_like(map1, val)
    _full_cache[key] = result
    return result

# use the loss function instead of the metric because it's soft so I can ignore thresholding
iou_loss = DiceLoss(jaccard=True, reduction='none')
def iou_compare(map1, map2):
    result = 1 - iou_loss(map1, map2)
    result0_0 = 1 - iou_loss(_full_like(map1, 0.0), map2)
    result0_5 = 1 - iou_loss(_full_like(map1, 0.5), map2)
    result1_0 = 1 - iou_loss(_full_like(map1, 1.0), map2)
    result_bkg = 1 - iou_loss(get_background_like(map1), map2)
    assert len(result.shape) == 2
    return torch.stack([result, result0_0, result0_5, result1_0, result_bkg], dim=2)


def metric_one_volume(ex_dir):
    examples = [] # one per vis

    ex_id = os.path.basename(ex_dir)
    # load gt and add l1,l2
    gt_fname = get_file(os.path.join(ex_dir, '*_gt.nii.gz'))
    gt = nib.load(gt_fname).get_fdata().transpose((3, 0, 1, 2))
    extra_gt = brats_label_to_raw(gt, onehot=True)
    # just labels 1 and 2 since label 4 is ET (channel 0 of gt)
    extra_gt = extra_gt[:2]
    # channels = ET, TC, WT, l1, l2
    gt = np.concatenate([gt, extra_gt], axis=0)
    gt_border = [find_border(gti) for gti in gt]
    gt_border = np.stack(gt_border)
    # BCHWD
    gt = torch.tensor(gt[None])
    gt_border = torch.tensor(gt_border[None])

    # load pred
    pred_fname = get_file(os.path.join(ex_dir, '*_pred.nii.gz'))
    pred = nib.load(pred_fname).get_fdata().transpose((3, 0, 1, 2))
    # BCHWD
    pred = torch.tensor(pred[None])

    # ET, TC, WT
    # "1 - " because this is the loss function
    #print('pred')
    pred_iou = iou_compare(pred, gt[:, :3])
    pred_corr = rank_compare(pred, gt[:, :3])


    for vis_key, layer_key in all_keys:
        examples.append({
            'id': ex_id,
            'vis_key': vis_key,
            'layer_key': layer_key,
            'pred_iou': pred_iou.numpy(),
            'pred_corr': pred_corr.numpy(),
            'vis_vis_iou': {},
            'vis_vis_corr': {}
        })
        ex = examples[-1]
        # load vis
        try:
            vis = load_vis(vis_key, layer_key, ex_id, args)
            vis = vis.expand(-1, 5, -1, -1, -1)
            #print('gt')
            vis_iou = iou_compare(vis, gt)
            vis_corr = rank_compare(vis, gt)
            ex['vis_gt_iou'] = vis_iou.numpy()
            ex['vis_gt_corr'] = vis_corr.numpy()
            #print('border')
            vis_border_iou = iou_compare(vis, gt_border)
            vis_border_corr = rank_compare(vis, gt_border)
            ex['vis_gt_border_iou'] = vis_border_iou.numpy()
            ex['vis_gt_border_corr'] = vis_border_corr.numpy()
            #print('background')
            background_iou = iou_compare(vis[:, 0:1], background_mask)
            background_corr = rank_compare(vis[:, 0:1], background_mask)
            ex['vis_background_iou'] = background_iou.numpy()
            ex['vis_background_corr'] = background_corr.numpy()
        except VisNotFoundError:
            ex['vis_gt_iou'] = None
            ex['vis_gt_corr'] = None
            ex['vis_gt_border_iou'] = None
            ex['vis_gt_border_corr'] = None
            ex['vis_background_iou'] = None
            ex['vis_background_corr'] = None

        #key = (vis_key, layer_key)
        #for other_vis_key, other_layer_key in all_keys:
        #    other_key = other_vis_key, other_layer_key
        #    if key == other_key:
        #        val = None
        #    else:
        #        try:
        #            other_vis = load_vis(other_vis_key, other_layer_key, ex_id, args)
        #            other_iou = iou_compare(vis, other_vis)
        #            other_corr = rank_compare(vis, other_vis)
        #            ex['vis_vis_iou'][other_key] = other_iou.numpy()
        #            ex['vis_vis_corr'][other_key] = other_corr.numpy()
        #        except VisNotFoundError:
        #            val = None
        #    ex['vis_vis_iou'][other_key] = None
        #    ex['vis_vis_corr'][other_key] = None

    #_cache = {}
    return examples


def main(args):
    #global _cache
    global all_keys
    result_dir = os.path.join(args.model_dir, args.result_dname)
    pred_gt_dataset_dir = os.path.join(args.model_dir, 'an_vis_niftis_entire_cls0_model.0', 'result_dataset')

    vis_keys = list(eval.vis_key_name_mapping.keys())
    layer_keys = list(dict(layers_list.unet_layers).keys())
    all_keys = [(vis_key, layer_key) for vis_key in vis_keys
                                     for layer_key in layer_keys]

    examples = []

    #for ex_dir in tqdm(glob.glob(os.path.join(pred_gt_dataset_dir, '*'))):
    from joblib import Parallel, delayed
    ex_dirs = tqdm(glob.glob(os.path.join(pred_gt_dataset_dir, '*')))
    result = Parallel(n_jobs=10)(delayed(metric_one_volume)(ex_dir) for ex_dir in ex_dirs)
    examples = sum(result, [])


    results = {
        #'vis_key': vis_key,
        #'vis_name': vis_name,
        #'layer_key': layer_key,
        #'layer_name': layer_name,
        'examples': examples,
    }

    result_file = os.path.join(result_dir, 'analysis.pkl')
    with open(result_file, 'wb') as f:
        pkl.dump(results, f)




if __name__ == "__main__":
    import options
    args = options.parser.parse_args()
    main(args)
