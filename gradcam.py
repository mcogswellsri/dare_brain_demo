import argparse
import os
import os.path as pth
import tempfile

import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('./monai_model/')
import vis_utils
import layers_list
import net_utils

from brats_example import modality_to_idx


class GradCAMMonai:
    def __init__(self, params, example):
        self.example = example

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.model = 'unet'
        args.up_mode = 'resize'
        self.net = net_utils.load_net(args)
        args.checkpoint_epoch = 1000
        args.model_dir = '/usr/monai_model_data/models/unet_resize2/'
        checkpoint = pth.join(args.model_dir,
                              'checkpoints',
                              f'net_checkpoint_{args.checkpoint_epoch}.pth')
        checkpoint = torch.load(checkpoint)
        self.net.load_state_dict(checkpoint['net'])

        self.batch = example.example
        self.layers_key_to_name = dict(layers_list.unet_layers)
        self.layers_name_to_key = {v: k for k, v in self.layers_key_to_name.items()}
        self.cache = {}

    def get_target_fn(self, vis_key):
        # TODO: refactor into a function
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
        return target_fn_lookup[vis_key]

    def get_lname(self, gcam_params):
        layer = gcam_params['layer']
        lnum, sub_stage = layer.split('_')
        if sub_stage == 'early':
            lname = f'{lnum} high res'
        elif sub_stage == 'late':
            lname = f'{lnum} final'
        return lname

    def compute_gradcam(self, gcam_params):
        target_cls = gcam_params['target_cls']
        target_region = gcam_params['target_region']
        layer = gcam_params['layer']
        vis_key = f'{target_region}_cls{target_cls}'
        norm_type = gcam_params['norm_type']
        if norm_type == 'pixelwise':
            vis_key += '_npixel'
        lname = self.get_lname(gcam_params)
        lkey = self.layers_name_to_key[lname]
        layers = [(lkey, lname)]
        target_fn, neg, alphas = self.get_target_fn(vis_key)
        assert not alphas
        vis_result = vis_utils.gradcam(self.net,
                                       self.batch,
                                       layers,
                                       target_fn,
                                       neg=neg,
                                       alphas=alphas)
        self.vis_result = vis_result

    def get_gradcam(self, gcam_params):
        target_cls = gcam_params['target_cls']
        target_region = gcam_params['target_region']
        layer = gcam_params['layer']
        norm_type = gcam_params['norm_type']
        key = f'{target_region}_{target_cls}_{layer}_{norm_type}'
        if key not in self.cache:
            print('computing')
            self.compute_gradcam(gcam_params)
            self.cache[key] = self.vis_result
        return self.cache[key]

    def slice_fname(self, slice_id, gcam_params):
        slice_idx = int(slice_id)

        vis_result = self.get_gradcam(gcam_params)
        lname = self.get_lname(gcam_params)
        lkey = self.layers_name_to_key[lname]
        print(lkey)
        vis = vis_result[lkey][0:1]
        slice_ratio = vis.shape[-1] / self.example.example['image'].shape[-1]
        adjusted_slice_idx = int(slice_ratio * slice_idx)

        file = tempfile.NamedTemporaryFile(delete=False)
        #path = os.path.join(out_dir, out_name + f'_{vkey}_{lkey}.jpg')
        plt.imsave(file, vis[0, :, :, adjusted_slice_idx], vmin=0, vmax=1)
        return file.name






class GradCAMAttUnet:
    def __init__(self, params, example):
        self.example = example
        self.params = params

        # TODO: this expects self.attunet to be attached externally... it should
        # be handled in a way consistent with how the Monai model is handled
        self.batch = example.example
        if params.model == '2d_att_unet':
            self.layers_key_to_name = dict(layers_list.att_unet_layers)
        elif params.model == '2d_unet':
            self.layers_key_to_name = dict(layers_list.unet_2d_layers)
        self.layers_name_to_key = {v: k for k, v in self.layers_key_to_name.items()}
        self.cache = {}

        # the att unet uses the raw BraTS label space instead of ET/TC/WT
        vis_utils.set_label_space('raw')
        # the att unet is 2d, not 3d
        vis_utils.set_agg_dims((1, 2))

    def get_target_fn(self, vis_key):
        # TODO: refactor into a function
        target_fn_lookup = {
            # target function, negative, only report alphas
            'entire_cls0': (vis_utils.target_entire_cls(0), False, False),
            'entire_cls0_neg': (vis_utils.target_entire_cls(0), True, False),
            'entire_cls1': (vis_utils.target_entire_cls(1), False, False),
            'entire_cls1_neg': (vis_utils.target_entire_cls(1), True, False),
            'entire_cls2': (vis_utils.target_entire_cls(2), False, False),
            'entire_cls2_neg': (vis_utils.target_entire_cls(2), True, False),
            'entire_cls0_npixel': (vis_utils.target_entire_cls(0), False, 'pixelwise'),
            'entire_cls1_npixel': (vis_utils.target_entire_cls(1), False, 'pixelwise'),
            'entire_cls2_npixel': (vis_utils.target_entire_cls(2), False, 'pixelwise'),
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
        return target_fn_lookup[vis_key]

    def get_lname(self, gcam_params):
        layer = gcam_params['layer']
        if layer == 'Input':
            return 'Input'
        lnum, sub_stage = layer.split('_')
        if sub_stage == 'early':
            lname = f'{lnum} high res'
        elif sub_stage == 'late':
            lname = f'{lnum} final'
        return lname

    def compute_gradcam(self, gcam_params, slice_idx):
        target_cls = gcam_params['target_cls']
        target_region = gcam_params['target_region']
        layer = gcam_params['layer']
        vis_key = f'{target_region}_cls{target_cls}'
        norm_type = gcam_params['norm_type']
        if norm_type == 'pixelwise':
            vis_key += '_npixel'

        lname = self.get_lname(gcam_params)
        lkey = self.layers_name_to_key[lname]
        layers = [(lkey, lname)]
        print(f'using layers: {layers}')

        target_fn, neg, alphas = self.get_target_fn(vis_key)
        #assert not alphas
        batch = {
            # TODO: parameterize device?
            'image': self.batch['image'][:, :, :, :, slice_idx].to('cuda'),
            'seg': self.batch['seg'][:, :, :, :, slice_idx].to('cuda'),
            'y_pred': torch.from_numpy(self.example.get_y_pred(slice_idx)).to('cuda'),
        }
        if 'fp' in vis_key or 'fn' in vis_key:
            raise Exception('not yet supported for att unet because logits need to be re-mapped')
        torch.random.manual_seed(77)
        if self.params.model == '2d_att_unet':
            net = self.attunet.unet
        elif self.params.model == '2d_unet':
            net = self.unet_2d.unet
        vis_result = vis_utils.gradcam(net,
                                       batch,
                                       layers,
                                       target_fn,
                                       neg=neg,
                                       alphas=alphas)
        self.vis_result = vis_result

    def get_gradcam(self, gcam_params, slice_idx):
        target_cls = gcam_params['target_cls']
        target_region = gcam_params['target_region']
        layer = gcam_params['layer']
        modality = gcam_params['modality']
        norm_type = gcam_params['norm_type']
        key = f'{target_region}_{target_cls}_{layer}_{norm_type}_{slice_idx}_{modality}'
        if key not in self.cache:
            print('computing')
            self.compute_gradcam(gcam_params, slice_idx)
            self.cache[key] = self.vis_result
        return self.cache[key]

    def slice_fname(self, slice_id, gcam_params):
        slice_idx = int(slice_id)

        vis_result = self.get_gradcam(gcam_params, slice_idx)
        lname = self.get_lname(gcam_params)
        lkey = self.layers_name_to_key[lname]
        print(lkey)
        vis = vis_result[lkey][0:1]

        if gcam_params['norm_type'] == 'pixelwise':
            mod_idx = modality_to_idx[gcam_params['modality']]
            vis = vis[:, mod_idx]
        file = tempfile.NamedTemporaryFile(delete=False)
        #path = os.path.join(out_dir, out_name + f'_{vkey}_{lkey}.jpg')
        plt.imsave(file, vis[0, :, :], vmin=0, vmax=1)
        return file.name






class GradCAM2dUnet:
    def __init__(self, params, example):
        self.example = example

        # TODO: this expects self.attunet to be attached externally... it should
        # be handled in a way consistent with how the Monai model is handled
        self.batch = example.example
        self.layers_key_to_name = dict(layers_list.unet_2d_layers)
        self.layers_name_to_key = {v: k for k, v in self.layers_key_to_name.items()}
        self.cache = {}

        # the 2d unet is 2d, not 3d
        vis_utils.set_agg_dims((1, 2))

    def get_target_fn(self, vis_key):
        # TODO: refactor into a function
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
        return target_fn_lookup[vis_key]

    def get_lname(self, gcam_params):
        layer = gcam_params['layer']
        lnum, sub_stage = layer.split('_')
        if sub_stage == 'early':
            lname = f'{lnum} high res'
        elif sub_stage == 'late':
            lname = f'{lnum} final'
        return lname

    def compute_gradcam(self, gcam_params, slice_idx):
        target_cls = gcam_params['target_cls']
        target_region = gcam_params['target_region']
        layer = gcam_params['layer']
        vis_key = f'{target_region}_cls{target_cls}'

        lname = self.get_lname(gcam_params)
        lkey = self.layers_name_to_key[lname]
        layers = [(lkey, lname)]
        print(f'using layers: {layers}')

        target_fn, neg, alphas = self.get_target_fn(vis_key)
        assert not alphas
        batch = {
            # TODO: parameterize device?
            'image': self.batch['image'][:, :, :, :, slice_idx].to('cuda'),
        }
        vis_result = vis_utils.gradcam(self.unet_2d.unet,
                                       batch,
                                       layers,
                                       target_fn,
                                       neg=neg,
                                       alphas=alphas)
        self.vis_result = vis_result

    def get_gradcam(self, gcam_params, slice_idx):
        target_cls = gcam_params['target_cls']
        target_region = gcam_params['target_region']
        layer = gcam_params['layer']
        key = f'{target_region}_{target_cls}_{layer}_{slice_idx}'
        if key not in self.cache:
            print('computing')
            self.compute_gradcam(gcam_params, slice_idx)
            self.cache[key] = self.vis_result
        return self.cache[key]

    def slice_fname(self, slice_id, gcam_params):
        slice_idx = int(slice_id)

        vis_result = self.get_gradcam(gcam_params, slice_idx)
        lname = self.get_lname(gcam_params)
        lkey = self.layers_name_to_key[lname]
        print(lkey)
        vis = vis_result[lkey][0:1]

        file = tempfile.NamedTemporaryFile(delete=False)
        #path = os.path.join(out_dir, out_name + f'_{vkey}_{lkey}.jpg')
        plt.imsave(file, vis[0, :, :], vmin=0, vmax=1)
        return file.name
