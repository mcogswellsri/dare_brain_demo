import shutil
import os
import sys
import logging
import json
from pprint import pprint

import numpy as np
import torch
import monai
import ignite

from monai.losses import DiceLoss
from monai.handlers import MeanDice
from monai.metrics.meandice import DiceMetric
from monai.data import NiftiSaver
from ignite.engine import Engine

import options
from data_utils import load_data, resize
from net_utils import load_net
from train import prepare_batch, prepare_output
from custom_transforms import brats_label_to_raw

import vis_utils
import layers_list

import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns



vis_key_name_mapping = {
    'entire_cls0': 'ET GradCAM',
    'entire_cls0_neg': 'ET Neg GradCAM',
    'entire_cls1': 'TC GradCAM',
    'entire_cls1_neg': 'TC Neg GradCAM',
    'entire_cls2': 'WT GradCAM',
    'entire_cls2_neg': 'WT Neg GradCAM',
    'fn_cls0': 'ET FN GradCAM',
    'fn_cls1': 'TC FN GradCAM',
    'fn_cls2': 'WT FN GradCAM',
    'fp_cls0': 'ET FP GradCAM',
    'fp_cls1': 'TC FP GradCAM',
    'fp_cls2': 'WT FP GradCAM',
    'tn_cls0': 'ET TN GradCAM',
    'tn_cls1': 'TC TN GradCAM',
    'tn_cls2': 'WT TN GradCAM',
    'tp_cls0': 'ET TP GradCAM',
    'tp_cls1': 'TC TP GradCAM',
    'tp_cls2': 'WT TP GradCAM',
    'entire_cls0_alphas': 'ET GradCAM Alphas',
    'entire_cls1_alphas': 'TC GradCAM Alphas',
    'entire_cls2_alphas': 'WT GradCAM Alphas',
    'fn_cls0_alphas': 'ET FN GradCAM Alphas',
    'fn_cls1_alphas': 'TC FN GradCAM Alphas',
    'fn_cls2_alphas': 'WT FN GradCAM Alphas',
    'fp_cls0_alphas': 'ET FP GradCAM Alphas',
    'fp_cls1_alphas': 'TC FP GradCAM Alphas',
    'fp_cls2_alphas': 'WT FP GradCAM Alphas',
    'tn_cls0_alphas': 'ET TN GradCAM Alphas',
    'tn_cls1_alphas': 'TC TN GradCAM Alphas',
    'tn_cls2_alphas': 'WT TN GradCAM Alphas',
    'tp_cls0_alphas': 'ET TP GradCAM Alphas',
    'tp_cls1_alphas': 'TC TP GradCAM Alphas',
    'tp_cls2_alphas': 'WT TP GradCAM Alphas',
    'uncertain_cls0': 'ET Uncertain GradCAM',
    'uncertain_cls1': 'TC Uncertain GradCAM',
    'uncertain_cls2': 'WT Uncertain GradCAM',
}

err_key_name_mapping = {
    'fn_cls0': 'ET FN',
    'fn_cls1': 'TC FN',
    'fn_cls2': 'WT FN',
    'fp_cls0': 'ET FP',
    'fp_cls1': 'TC FP',
    'fp_cls2': 'WT FP',
    'tn_cls0': 'ET TN',
    'tn_cls1': 'TC TN',
    'tn_cls2': 'WT TN',
    'tp_cls0': 'ET TP',
    'tp_cls1': 'TC TP',
    'tp_cls2': 'WT TP',
    'fn_cls0_alphas': 'ET FN',
    'fn_cls1_alphas': 'TC FN',
    'fn_cls2_alphas': 'WT FN',
    'fp_cls0_alphas': 'ET FP',
    'fp_cls1_alphas': 'TC FP',
    'fp_cls2_alphas': 'WT FP',
    'tn_cls0_alphas': 'ET TN',
    'tn_cls1_alphas': 'TC TN',
    'tn_cls2_alphas': 'WT TN',
    'tp_cls0_alphas': 'ET TP',
    'tp_cls1_alphas': 'TC TP',
    'tp_cls2_alphas': 'WT TP',
    'uncertain_cls0': 'ET Uncertain',
    'uncertain_cls1': 'TC Uncertain',
    'uncertain_cls2': 'WT Uncertain',
}


def seg_cmap(seg):
    assert set(np.unique(seg)).issubset(set([0., 1.]))
    # normal color map
    cmap = matplotlib.cm.get_cmap('viridis')
    img = cmap(seg)

    # find background mask
    mask = (seg == 0)
    mask = mask[:, :, None].repeat(4, axis=2)
    mask[:, :, :3] = False

    # set background alpha to 0
    img[mask] = 0.
    return img


def main(args):
    #assert args.vis_page or args.vis_nifti or args.contest_submission or args.save_metrics or args.survival_features or args.alpha_analysis
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    monai.utils.set_determinism(seed=0)
    result_dir = os.path.join(args.model_dir, args.result_dname)
    os.makedirs(result_dir, exist_ok=True)
    result_dataset_dir = os.path.join(result_dir, 'result_dataset')
    os.makedirs(result_dataset_dir, exist_ok=True)

    # load data
    val_ds, val_loader = load_data(args, include_train=False)
    has_ground_truth = ("seg" in val_ds[0])

    # create net, loss, optimizer
    net, loss_fn = load_net(args, include_loss=True)
    assert not args.checkpoint_epoch is None
    checkpoint = os.path.join(args.model_dir,
                              'checkpoints',
                              f'net_checkpoint_{args.checkpoint_epoch}.pth')
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['net'])
    device = torch.device("cuda:0")
    net = net.to(device)
    vis_meta = {'examples': []}
    # add things to this list to actually compute them
    vis_keys = [
        'entire_cls0', 'entire_cls0_neg',
        'entire_cls1', 'entire_cls1_neg',
        'entire_cls2', 'entire_cls2_neg',
        'fn_cls0',
        'fn_cls1',
        'fn_cls2',
        'fp_cls0',
        'fp_cls1',
        'fp_cls2',
        'tn_cls0',
        'tn_cls1',
        'tn_cls2',
        'tp_cls0',
        'tp_cls1',
        'tp_cls2',
        'entire_cls0_alphas',
        'entire_cls1_alphas',
        'entire_cls2_alphas',
        'fn_cls0_alphas',
        'fn_cls1_alphas',
        'fn_cls2_alphas',
        'fp_cls0_alphas',
        'fp_cls1_alphas',
        'fp_cls2_alphas',
        'tn_cls0_alphas',
        'tn_cls1_alphas',
        'tn_cls2_alphas',
        'tp_cls0_alphas',
        'tp_cls1_alphas',
        'tp_cls2_alphas',
        'uncertain_cls0',
        'uncertain_cls1',
        'uncertain_cls2',
    ]
    if args.vis_key is not None:
        vis_keys = [args.vis_key]
    layers = layers_list.unet_layers
    if args.layer_key is not None:
        unet_layers = dict(layers_list.unet_layers)
        layers = [(args.layer_key, unet_layers[args.layer_key])]

    def inference(engine, batch):
        net.eval()
        x, y, batch = prepare_batch(batch, device, args)
        y_pred = net(x)
        y_pred_probs = torch.sigmoid(y_pred)
        # for metrics
        output = prepare_output(batch, y, y_pred, None)
        batch['y_pred_probs'] = y_pred_probs
        batch['y_pred'] = (y_pred_probs > args.pred_thresh).to(y_pred_probs)

        # visualize
        vis_results = []
        target_fn_lookup = {
            # target function, negative, only report alphas
            'entire_cls0': (vis_utils.target_entire_cls(0), False, False),
            'entire_cls0_neg': (vis_utils.target_entire_cls(0), True, False),
            'entire_cls1': (vis_utils.target_entire_cls(1), False, False),
            'entire_cls1_neg': (vis_utils.target_entire_cls(1), True, False),
            'entire_cls2': (vis_utils.target_entire_cls(2), False, False),
            'entire_cls2_neg': (vis_utils.target_entire_cls(2), True, False),
            'fn_cls0': (vis_utils.target_partial('fn', 0), True, False),
            'fn_cls1': (vis_utils.target_partial('fn', 1), True, False),
            'fn_cls2': (vis_utils.target_partial('fn', 2), True, False),
            'fp_cls0': (vis_utils.target_partial('fp', 0), True, False),
            'fp_cls1': (vis_utils.target_partial('fp', 1), True, False),
            'fp_cls2': (vis_utils.target_partial('fp', 2), True, False),
            'tn_cls0': (vis_utils.target_partial('tn', 0), True, False),
            'tn_cls1': (vis_utils.target_partial('tn', 1), True, False),
            'tn_cls2': (vis_utils.target_partial('tn', 2), True, False),
            'tp_cls0': (vis_utils.target_partial('tp', 0), True, False),
            'tp_cls1': (vis_utils.target_partial('tp', 1), True, False),
            'tp_cls2': (vis_utils.target_partial('tp', 2), True, False),
            'entire_cls0_alphas': (vis_utils.target_entire_cls(0), False, True),
            'entire_cls1_alphas': (vis_utils.target_entire_cls(1), False, True),
            'entire_cls2_alphas': (vis_utils.target_entire_cls(2), False, True),
            'fn_cls0_alphas': (vis_utils.target_partial('fn', 0), True, True),
            'fn_cls1_alphas': (vis_utils.target_partial('fn', 1), True, True),
            'fn_cls2_alphas': (vis_utils.target_partial('fn', 2), True, True),
            'fp_cls0_alphas': (vis_utils.target_partial('fp', 0), True, True),
            'fp_cls1_alphas': (vis_utils.target_partial('fp', 1), True, True),
            'fp_cls2_alphas': (vis_utils.target_partial('fp', 2), True, True),
            'tn_cls0_alphas': (vis_utils.target_partial('tn', 0), True, True),
            'tn_cls1_alphas': (vis_utils.target_partial('tn', 1), True, True),
            'tn_cls2_alphas': (vis_utils.target_partial('tn', 2), True, True),
            'tp_cls0_alphas': (vis_utils.target_partial('tp', 0), True, True),
            'tp_cls1_alphas': (vis_utils.target_partial('tp', 1), True, True),
            'tp_cls2_alphas': (vis_utils.target_partial('tp', 2), True, True),
            'uncertain_cls0': (vis_utils.target_partial('uncertain', 0), True, False),
            'uncertain_cls1': (vis_utils.target_partial('uncertain', 1), True, False),
            'uncertain_cls2': (vis_utils.target_partial('uncertain', 2), True, False),
        }
        for vis_key in vis_keys:
            vname = vis_key_name_mapping[vis_key]
            target_fn, neg, alphas = target_fn_lookup[vis_key]

            if alphas:
                _layers = [('', 'Input')]
            else:
                _layers = layers

            vis_result = vis_utils.gradcam(net,
                                           batch,
                                           _layers,
                                           target_fn,
                                           neg=neg,
                                           alphas=alphas)

            vis_results.append((vis_key, vname, vis_result))

        # save to file
        dl = DiceLoss()
        input_img = x.detach().cpu().numpy()
        pred_probs = y_pred_probs.cpu().detach().numpy()
        preds = (pred_probs > args.pred_thresh).astype(pred_probs.dtype)
        if has_ground_truth:
            ground_truth = y.cpu().numpy()
        for i, path in enumerate(batch['t1_meta_dict']['filename_or_obj']):
            relpath = os.path.relpath(path, args.val_data)
            out_name = os.path.split(relpath)[0]
            out_dir = os.path.join(result_dataset_dir, out_name)
            os.makedirs(out_dir, exist_ok=True)

            if args.contest_submission:
                print(f'saving example {i} prediction')
                y_pred_thresh = (y_pred_probs[i] > args.pred_thresh).cpu().numpy()
                raw_label = brats_label_to_raw(y_pred_thresh)
                nifti_saver = NiftiSaver(result_dataset_dir, '')
                meta = {
                    'affine': batch['t1_meta_dict']['affine'][i],
                    'original_affine': batch['t1_meta_dict']['original_affine'][i],
                    'spatial_shape': batch['t1_meta_dict']['spatial_shape'][i],
                    'filename_or_obj': out_name,
                }
                nifti_saver.save(raw_label, meta)
                # take the '_' out of the filename (it's baked in to monai)
                src_fname = os.path.join(out_dir, out_name + '_.nii.gz')
                dst_fname = os.path.join(out_dir, out_name + '.nii.gz')
                shutil.move(src_fname, dst_fname)

            if args.survival_features:
                print(f'saving example {i} survival features')
                whole_brain_size = (x != 0).sum(dim=(2,3,4))[i, 0]

                pred_size = batch['y_pred'].sum(dim=(2,3,4))
                et_pred_size = pred_size[i, 0]
                tc_pred_size = pred_size[i, 1]
                wt_pred_size = pred_size[i, 2]

                gt_size = batch['seg'].sum(dim=(2,3,4))
                et_gt_size = gt_size[i, 0]
                tc_gt_size = gt_size[i, 1]
                wt_gt_size = gt_size[i, 2]

                subj_id = batch['subj_id'][i]

                survival_fname = os.path.join(result_dataset_dir, f'survival_features_{subj_id}.pt')
                features = {
                    'subj_id': subj_id,
                    'whole_brain_size': whole_brain_size,
                    'et_pred_size': et_pred_size,
                    'tc_pred_size': tc_pred_size,
                    'wt_pred_size': wt_pred_size,
                    'et_gt_size': et_gt_size,
                    'tc_gt_size': tc_gt_size,
                    'wt_gt_size': wt_gt_size,
                }
                torch.save(features, survival_fname)

            if args.alpha_analysis:
                print(f'saving example {i} alpha features')
                whole_brain_size = (x != 0).sum(dim=(2,3,4))[i, 0]
                whole_brain_size = whole_brain_size.detach().cpu().numpy()

                pred_size = batch['y_pred'].sum(dim=(2,3,4))
                et_pred_size = pred_size[i, 0].detach().cpu().numpy()
                tc_pred_size = pred_size[i, 1].detach().cpu().numpy()
                wt_pred_size = pred_size[i, 2].detach().cpu().numpy()

                gt_size = batch['seg'].sum(dim=(2,3,4))
                et_gt_size = gt_size[i, 0].detach().cpu().numpy()
                tc_gt_size = gt_size[i, 1].detach().cpu().numpy()
                wt_gt_size = gt_size[i, 2].detach().cpu().numpy()

                subj_id = batch['subj_id'][i]

                dice = DiceMetric(sigmoid=False, logit_thresh=args.pred_thresh, reduction='none')(batch['y_pred'], batch['seg'])
                mean_dice = dice.mean(dim=1)[i]
                et_dice = dice[i, 0]
                tc_dice = dice[i, 1]
                wt_dice = dice[i, 2]

                features = {
                    'subj_id': subj_id,
                    'whole_brain_size': whole_brain_size,
                    'et_pred_size': et_pred_size,
                    'tc_pred_size': tc_pred_size,
                    'wt_pred_size': wt_pred_size,
                    'et_gt_size': et_gt_size,
                    'tc_gt_size': tc_gt_size,
                    'wt_gt_size': wt_gt_size,
                    'alphas': {},
                    'errors': {},
                    'mean_dice': mean_dice,
                    'et_dice': et_dice,
                    'tc_dice': tc_dice,
                    'wt_dice': wt_dice,
                }

                for vkey, vname, vis_result in vis_results:
                    if 'alphas' not in vkey:
                        continue
                    lkey, lname = ('', 'Input')
                    vis = vis_result[lkey][i:i+1]
                    alpha = vis[:, :, 0, 0, 0]
                    features['alphas'][vkey] = alpha

                    if 'error_type' in vis_result:
                        error_type = vis_result['error_type']
                        error_mask = vis_result['error_mask']
                        features['errors'][error_type] = {
                            'counts': error_mask.sum(axis=(0,2,3,4)),
                            'ratios': error_mask.sum(axis=(0,2,3,4)) / whole_brain_size,
                        }

                fname = os.path.join(result_dataset_dir, f'alpha_data_{subj_id}.pt')
                torch.save(features, fname)

            if args.vis_page:
                gray = matplotlib.cm.get_cmap('gray')
                ex = {
                    'ex_name': out_name,
                    'input': {},
                    'preds': {},
                    'dice': {},
                }
                vis_meta['examples'].append(ex)

                slice_ratio = pred_probs[i, 0].sum(axis=(0,1)).argmax() / preds.shape[-1]
                slice = int(slice_ratio * input_img.shape[-1])
                ex['slice'] = slice
                ex['slice_ratio'] = slice_ratio

                # input
                t1_path = os.path.join(out_dir, out_name + '_t1.jpg')
                plt.imsave(t1_path, input_img[i, 0, :, :, slice], cmap=gray)
                ex['input']['t1'] = os.path.relpath(t1_path, result_dir)

                t2_path = os.path.join(out_dir, out_name + '_t2.jpg')
                plt.imsave(t2_path, input_img[i, 1, :, :, slice], cmap=gray)
                ex['input']['t2'] = os.path.relpath(t2_path, result_dir)

                t2flair_path = os.path.join(out_dir, out_name + '_t2flair.jpg')
                plt.imsave(t2flair_path, input_img[i, 2, :, :, slice], cmap=gray)
                ex['input']['t2flair'] = os.path.relpath(t2flair_path, result_dir)

                t1ce_path = os.path.join(out_dir, out_name + '_t1ce.jpg')
                plt.imsave(t1ce_path, input_img[i, 3, :, :, slice], cmap=gray)
                ex['input']['t1ce'] = os.path.relpath(t1ce_path, result_dir)

                # output
                et_path = os.path.join(out_dir, out_name + '_et_pred.png')
                plt.imsave(et_path, seg_cmap(preds[i, 0, :, :, slice]))
                ex['preds']['et'] = os.path.relpath(et_path, result_dir)

                tc_path = os.path.join(out_dir, out_name + '_tc_pred.png')
                plt.imsave(tc_path, seg_cmap(preds[i, 1, :, :, slice]))
                ex['preds']['tc'] = os.path.relpath(tc_path, result_dir)

                wt_path = os.path.join(out_dir, out_name + '_wt_pred.png')
                plt.imsave(wt_path, seg_cmap(preds[i, 2, :, :, slice]))
                ex['preds']['wt'] = os.path.relpath(wt_path, result_dir)

                if has_ground_truth:
                    ex['ground_truth'] = {}
                    # ground truth
                    et_gt_path = os.path.join(out_dir, out_name + '_et_gt.png')
                    plt.imsave(et_gt_path, seg_cmap(ground_truth[i, 0, :, :, slice]))
                    ex['ground_truth']['et'] = os.path.relpath(et_gt_path, result_dir)
                    ex['dice']['et'] = dl(y_pred_probs[i:i+1,0:1], y[i:i+1,0:1]).item()

                    tc_gt_path = os.path.join(out_dir, out_name + '_tc_gt.png')
                    plt.imsave(tc_gt_path, seg_cmap(ground_truth[i, 1, :, :, slice]))
                    ex['ground_truth']['tc'] = os.path.relpath(tc_gt_path, result_dir)
                    ex['dice']['tc'] = dl(y_pred_probs[i:i+1,1:2], y[i:i+1,1:2]).item()

                    wt_gt_path = os.path.join(out_dir, out_name + '_wt_gt.png')
                    plt.imsave(wt_gt_path, seg_cmap(ground_truth[i, 2, :, :, slice]))
                    ex['ground_truth']['wt'] = os.path.relpath(wt_gt_path, result_dir)
                    ex['dice']['wt'] = dl(y_pred_probs[i:i+1,2:3], y[i:i+1,2:3]).item()

                # visualizations
                ex['visualizations'] = []
                ex['error_keys'] = []
                ex['error_masks'] = []

            target_shape = input_img.shape[2:]
            
            # NOTE: this is different than the raw data because of the pre-processing
            save_other_nifti = True #bool(args.vis_nifti)
            if save_other_nifti:
                meta = {
                    #'affine': batch['t1_meta_dict']['affine'][i],
                    #'original_affine': batch['t1_meta_dict']['original_affine'][i],
                    # do not resize to original shape
                    #'spatial_shape': vis.shape,
                    # resize to original shape
                    'spatial_shape': target_shape,
                    'filename_or_obj': out_name,
                }
                # pred
                nifti_saver = NiftiSaver(result_dataset_dir, 'pred')
                nifti_saver.save(preds[i], meta)
                # ground truth
                if has_ground_truth:
                    nifti_saver = NiftiSaver(result_dataset_dir, 'gt')
                    nifti_saver.save(ground_truth[i], meta)


            vis_rev_map = {}
            for vkey, vname, vis_result in vis_results:

                _, _, alphas = target_fn_lookup[vkey]
                if alphas:
                    _layers = [('', 'Input')]
                else:
                    _layers = layers

                for lkey, lname in _layers:

                    vis = vis_result[lkey][i:i+1]
                    vis_rev_map[(vkey, lkey)] = vis
                    if args.vis_nifti:
                        vis_upsampled = resize(vis, target_shape)
                        nifti_saver = NiftiSaver(result_dataset_dir,
                                        f'{vname.replace(" ", "-")}_{lname.replace(" ", "-")}')
                        meta = {
                            #'affine': batch['t1_meta_dict']['affine'][i],
                            #'original_affine': batch['t1_meta_dict']['original_affine'][i],
                            # do not resize to original shape
                            #'spatial_shape': vis.shape,
                            # resize to original shape
                            'spatial_shape': target_shape,
                            'filename_or_obj': out_name,
                        }
                        nifti_saver.save(vis_upsampled, meta)


                    if args.vis_page:
                        slice = int(slice_ratio * vis.shape[-1])
                        if alphas:
                            # 4 inputs in BraTS
                            modalities = [0, 1, 2, 3]
                        else:
                            modalities = [0]
                        for modality in modalities:
                            if alphas:
                                path = os.path.join(out_dir, out_name + f'_{vkey}_{lkey}_m{modality}.jpg')
                                plt.imsave(path, vis[0, modality, :, :, slice], vmin=0, vmax=1)
                            else:
                                path = os.path.join(out_dir, out_name + f'_{vkey}_{lkey}.jpg')
                                plt.imsave(path, vis[0, :, :, slice], vmin=0, vmax=1)

                            ex['visualizations'].append({
                                'vkey': vkey,
                                'vis_type': vname,
                                'lkey': lkey,
                                'layer': lname,
                                'path': os.path.relpath(path, result_dir),
                            })
                            if alphas:
                                ex['visualizations'][-1]['modality'] = modality

                            if vkey not in ex['error_keys'] and 'error_type' in vis_result:
                                error_type = vis_result['error_type']
                                error_mask = vis_result['error_mask']
                                slice = int(slice_ratio * error_mask.shape[-1])
                                error_path = os.path.join(out_dir, out_name + f'_{vkey}_{lkey}_{error_type}.png')
                                if 'cls0' in vkey:
                                    clsi = 0
                                elif 'cls1' in vkey:
                                    clsi = 1
                                elif 'cls2' in vkey:
                                    clsi = 2
                                plt.imsave(error_path, seg_cmap(error_mask[i, clsi, :, :, slice]))
                                ex['error_keys'].append(vkey)
                                ex['error_masks'].append({
                                    'vkey': vkey,
                                    'name': err_key_name_mapping[vkey],
                                    'path': os.path.relpath(error_path, result_dir),
                                })

                            # difference between error gradcam and regular gradcam
                            if 'error_type' in vis_result and not alphas:
                                vkey_ = vkey + '_diff'
                                vname_ = vname + ' Diff'
                                org_vis_ = vis_rev_map[(vkey.replace(error_type, 'entire'), lkey)]
                                diff_ = np.abs(org_vis_ - vis)
                                path_ = os.path.join(out_dir, out_name + f'_{vkey_}_{lkey}.jpg')
                                slice = int(slice_ratio * diff_.shape[-1])
                                plt.imsave(path_, diff_[0, :, :, slice], vmin=0, vmax=1)
                                ex['visualizations'].append({
                                    'vkey': vkey_,
                                    'vis_type': vname_,
                                    'lkey': lkey,
                                    'layer': lname,
                                    'path': os.path.relpath(path_, result_dir),
                                })

        return output
    evaluator = Engine(inference)

    # metrics
    if has_ground_truth:
        vallm = ignite.metrics.Loss(loss_fn,
                      output_transform=lambda out: (out["y_pred"], out["y"]))
        vallm.attach(evaluator, "val_loss")
        valmd = MeanDice(sigmoid=True, logit_thresh=args.pred_thresh,
             output_transform=lambda out: (out["y_pred"], out["y"]))
        valmd.attach(evaluator, "val_Mean_Dice")
        vald1 = MeanDice(sigmoid=True, logit_thresh=args.pred_thresh,
             output_transform=lambda out: (out["y_pred"][:, 0:1], out["y"][:, 0:1]))
        vald1.attach(evaluator, "val_Dice_ET")
        vald2 = MeanDice(sigmoid=True, logit_thresh=args.pred_thresh,
             output_transform=lambda out: (out["y_pred"][:, 1:2], out["y"][:, 1:2]))
        vald2.attach(evaluator, "val_Dice_TC")
        vald3 = MeanDice(sigmoid=True, logit_thresh=args.pred_thresh,
             output_transform=lambda out: (out["y_pred"][:, 2:3], out["y"][:, 2:3]))
        vald3.attach(evaluator, "val_Dice_WT")

    state = evaluator.run(val_loader, 1)
    print(state)
    pprint(state.metrics)

    if args.save_metrics:
        metric_file = os.path.join(result_dir, 'metrics.json')
        with open(metric_file, 'w') as f:
            json.dump(state.metrics, f, indent=4)

    if args.vis_page:
        meta_file = os.path.join(result_dir, 'vis_meta.json')
        with open(meta_file, 'w') as f:
            json.dump(vis_meta, f)

    args_file = os.path.join(result_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    args = options.parser.parse_args()
    main(args)
