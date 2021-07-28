#!/usr/bin/env python
from subprocess import Popen

import eval
import layers_list


for vis_key in eval.vis_key_name_mapping:
    for layer_key, _ in layers_list.unet_layers:
        ep = 20
        cmd = ('python eval.py --model unet --up-mode resize '
              f'--vis-page 0 --vis-nifti 1 --contest-submission 0 '
              f'--vis-key {vis_key} --layer-key {layer_key} '
              f'--result-dname an_vis_niftis_ep{ep}_{vis_key}_{layer_key} '
              f'--checkpoint-epoch {ep} data/models/unet_resize2/')
        Popen(cmd, shell=True).wait()

        cmd = ('python analyze.py '
              f'--vis-key {vis_key} --layer-key {layer_key} '
              f'--result-dname an_vis_niftis_ep{ep}_{vis_key}_{layer_key} '
              f'data/models/unet_resize2/')
        Popen(cmd, shell=True).wait()

