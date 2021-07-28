import torch
from flask import Flask, request, send_from_directory, send_file, jsonify
from flask import Response
import jsonrpcserver
from jsonrpcserver import methods
import json
from flask_cors import CORS
import time
import datetime
import math
import random
import hashlib
import os
import os.path as pth
import sys
import requests
import numpy as np
import torchvision
import torchvision.models
import torchvision.utils
import torchvision.datasets.folder
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
from io import BytesIO
import argparse
import string
import random
import cv2
import pprint
import matplotlib.pyplot as plt

#Command line options
parser=argparse.ArgumentParser(description='')
# Model
parser.add_argument('--port', type=int, default=5010)
parser.add_argument('--cache_mode', default='disk', choices=['none', 'disk'])
params=parser.parse_args();
params.argv=sys.argv;


# data config
params.image_dir = '/usr/brats/MICCAI_BraTS2020_TrainingData/BraTS20_Training_003/'
params.thumbnail_dir = '/usr/data_ro/BraTS20_Training_thumbnails/'

# TODO: remove... static datasets of independent slices
# fastmri
#params.image_dir = '/usr/data_ro/fastMRI_brain_images_all/images/t1/'
# brats
#params.image_dir = '/usr/data_ro/MICCAI_BraTS2020_TrainingData_dev_preprocessed_for_demo/images/t1/'

# general config
params.num_examples = 20

# inpainting config
#params.checkpoint_dir = '/usr/data/experiments/exp2.2/fastmri_gen_log/'
#params.checkpoint_dir = '/usr/data/experiments/exp2.4/fastmri_gen_log/'
params.checkpoint_dirs = {
    't1': '/usr/data/experiments/exp4.10.1/fastmri_gen_log/',
    't1ce': '/usr/data/experiments/exp4.11.1.0.0.0/fastmri_gen_log/',
    't2': '/usr/data/experiments/exp4.12.1.0/fastmri_gen_log/',
    't2flair': '/usr/data/experiments/exp4.13.1/fastmri_gen_log/',
}

modalities = ['t1', 't1ce', 't2', 't2flair']

# segment model config (unet, 2d_att_unet)
#params.model = 'unet' # 3d monai model
params.model = '2d_att_unet'
#params.model = '2d_unet'

app = Flask(__name__, static_url_path='')
CORS(app)

# for cachine internally on the server
import flask_caching
if params.cache_mode == 'none':
    app.config['CACHE_TYPE'] = 'null'
elif params.cache_mode == 'disk':
    app.config['CACHE_TYPE'] = 'filesystem'
    app.config['CACHE_DIR'] = '/usr/brats_cache'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 0
    app.config['CACHE_THRESHOLD'] = 0
disk_cache = flask_caching.Cache(app)

# for HTTP caching in the browser
import flask_cachecontrol as cc
flask_cache_control = cc.FlaskCacheControl()
flask_cache_control.init_app(app)



import brats_example
from inpainter import Inpainter
from brats_example import BraTSExample
from gradcam import GradCAMMonai, GradCAMAttUnet
from att_unet import AttUNetWrapper

brats_base = '/usr/brats/MICCAI_BraTS2020_TrainingData/'
# examples found to have significant failures
entries = [
    'BraTS20_Training_035',
    'BraTS20_Training_050',
    'BraTS20_Training_097',
    'BraTS20_Training_134',
    'BraTS20_Training_154',
    'BraTS20_Training_194',
    'BraTS20_Training_224',
    'BraTS20_Training_247',
    'BraTS20_Training_258',
    'BraTS20_Training_302',
    'BraTS20_Training_303',
    'BraTS20_Training_307',
    'BraTS20_Training_308',
    'BraTS20_Training_315',
    'BraTS20_Training_327',
]
#entries = [
#    'BraTS20_Training_003',
#    'BraTS20_Training_005',
#    'BraTS20_Training_008',
#    'BraTS20_Training_011',
#    'BraTS20_Training_016',
#    'BraTS20_Training_018',
#]
current_entry_id = 0

def load(params):
    entry_id = 0
    app.brats = {}
    app.brats['examples'] = {}
    app.brats['gradcams'] = {}
    app.brats['inpainters'] = {}
    print('dict created...')

def load_entry(params, entry_id):
    if entry_id not in app.brats['examples']:
        exam_id = entries[entry_id]
        params.image_dir = pth.join(brats_base, exam_id) + '/'
        example = BraTSExample(params)
        if params.model == 'unet':
            assert False, 'This is broken because gradcam has not loaded yet'
            example.net = app.brats['gradcam'].net
            example.predict()
        elif params.model == '2d_att_unet':
            load_attunet(params)
            example.attunet = app.brats['attunet']
        elif params.model == '2d_unet':
            load_2dunet(params)
            example.unet_2d = app.brats['unet_2d']
        app.brats['examples'][entry_id] = example
    return app.brats['examples'][entry_id]

def get_example(entry_id):
    return load_entry(params, entry_id)

def get_gradcam(entry_id):
    return load_gradcam(params, entry_id)

def load_inpainter(params, modality):
    params.checkpoint_dir = params.checkpoint_dirs[modality]
    inpainter = Inpainter(params)
    app.brats['inpainters'][modality] = inpainter

def load_gradcam(params, entry_id):
    if entry_id not in app.brats['gradcams']:
        example = get_example(entry_id)
        if params.model == 'unet':
            gradcam = GradCAMMonai(params, example)
        elif params.model == '2d_att_unet':
            gradcam = GradCAMAttUnet(params, example)
            # already done in load_entry
            #load_attunet(params)
            gradcam.attunet = app.brats['attunet']
        elif params.model == '2d_unet':
            gradcam = GradCAMAttUnet(params, example)
            gradcam.unet_2d = app.brats['unet_2d']
        app.brats['gradcams'][entry_id] = gradcam
    return app.brats['gradcams'][entry_id]


from unet_2d_wrapper import UNet2dWrapper

def load_2dunet(params):
    unet_2d = UNet2dWrapper()
    app.brats['unet_2d'] = unet_2d

def load_attunet(params):
    attunet = AttUNetWrapper()
    app.brats['attunet'] = attunet

def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


import lru_cache
lru_image=lru_cache.new(300);
lru_vqa=lru_cache.new(300);

p_use_cache=0.5;
def load_image(slice_id, modality, return_id=False):
    # TODO: remove hard coding and make this work for extranal images?
    #img_id = pth.basename(imurl)
    #if imurl in lru_image and float(torch.rand(1))<p_use_cache:
    #    image=lru_image[imurl].copy();
    #else:
    #    response=requests.get(imurl);
    #    image=Image.open(BytesIO(response.content));
    #    lru_image[imurl]=image;
    #    image=image.copy();

    return image, img_id


inpaint_mapping = {}

def get_box_id(box):
    if box is None:
        x, y, w, h = (0, 0, 0, 0)
    else:
        x, y, w, h = box
    bid = f'{x:.3f}-{y:.3f}-{w:.3f}-{h:.3f}'
    #w = str(hash(box))
    #bid = hashlib.md5(w.encode()).hexdigest()[:10]
    print(f'getting box id for {box}: {bid}')
    return bid


def remove_box(entry_id, slice_id, modality, box):
    # load inputs
    in_fname = get_example(entry_id).image_fname(slice_id, modality)
    image = cv2.imread(in_fname)

    # output info
    id=random_string(12);
    out_fname = f'counterfactual/{id}.jpg'
    # TODO: the first id should be the actual exam id, not 0
    box_id = get_box_id(box)
    url = f'counter_slices/{entry_id}/{slice_id}/{box_id}/{modality}'

    # compute inpaint if there is a box
    if box is None:
        out_fname = in_fname
        raw_out = image[None]
    else:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        print('remove box')

        if modality not in app.brats['inpainters']:
            load_inpainter(params, modality)
        inpainter = app.brats['inpainters'][modality]
        mask = inpainter.make_mask(image, x, y, w, h)
        imout, raw_out = inpainter.inpaint(image, mask)
        cv2.imwrite(out_fname, imout)

    # save everything
    print(f'sending url {url}')
    key = (entry_id, slice_id, box_id, modality)
    print(f'caching {key} as {out_fname}')
    inpaint_mapping[key] = (out_fname, raw_out)
    return {'imurl': url};

gradcam_mapping = {}

def gradcam(entry_id, slice_id, modality, target_cls, target_region, layer, norm_type):
    if norm_type == 'pixelwise':
        layer = 'Input'
    gcam_params = {
        'target_cls': {
            'ET': 0, 'TC': 1, 'WT': 2,
        }[target_cls],
        'target_region': target_region,
        'layer': layer,
        'modality': modality,
        'norm_type': norm_type,
    }
    fname = get_gradcam(entry_id).slice_fname(slice_id, gcam_params)
    return fname

@methods.add
def get_colors():
    labels = [
        (0, 'ET'),
        (1, 'TC'),
        (2, 'WT'),
    ]
    colorschemes = [
        'gt',
        'pred',
        'counterfactual',
    ]
    colors = {}
    for label_idx, label in labels:
        for colorscheme in colorschemes:
            color = brats_example.get_seg_color(label_idx, colorscheme, with_alpha=False)
            r, g, b = [int(c * 255) for c in color]
            hex = f'#{r:02x}{g:02x}{b:02x}'
            colors[f'{colorscheme}_{label}'] = hex
    return colors


def get_inpaint(entry_id, slice_id, box, mod):
    if ('inpainters' not in app.brats) or (mod not in app.brats['inpainters']):
        load_inpainter(params, mod)
    box_id = get_box_id(box)
    key = (entry_id, slice_id, box_id, mod)
    if key not in inpaint_mapping:
        # caches key in inpaint_mapping
        remove_box(entry_id, slice_id, mod, box)
    return inpaint_mapping[key]


def counterfactual(entry_id, slice_id, tumor_type, changed_modality):
    slice_idx = int(slice_id)
    inputs = {}
    box = get_example(entry_id).get_box_from_tumor(slice_id, tumor_type)
    box_id = get_box_id(box)
    for mod in modalities:
        mod_idx = get_example(entry_id).modality_to_idx[mod]
        max_val = get_example(entry_id).example['vis_image'][0, mod_idx, :, :, slice_idx].max()
        scale = float(max_val / 255.)
        if mod == changed_modality:
            _, ipt = get_inpaint(entry_id, slice_id, box, mod)
            ipt = ipt.transpose(0, 3, 1, 2)
            ipt = ipt[:, 0:1]
            ipt = torch.from_numpy(ipt).to(torch.float)
            inputs[mod] = ipt * scale
        else:
            in_fname = get_example(entry_id).image_fname(slice_id, changed_modality)
            image = cv2.imread(in_fname)
            image = torch.from_numpy(image).to(torch.float)
            inputs[mod] = image[None, None, :, :, 0] * scale

    ipt = torch.cat([
        inputs['t1'],
        inputs['t2'],
        inputs['t2flair'],
        inputs['t1ce'],
    ], dim=1)
    return (ipt, slice_id)


@methods.add
def list_ims():
    slices = list(map(str, range(0, 155, 2))) # app.brats['example'].list_slices()
    #ims=os.listdir(params.image_dir)
    #ims=[os.path.join('val',x) for x in ims]
    #ims = ims[:params.num_examples]
    return {'ims':slices};


@methods.add
def list_entries():
    # TODO: load appropriately
    return {'entries': entries};



@app.route('/slices/<int:entry_id>')
def send_best_slice(entry_id):
    modality = 't1'
    slice_id = 30
    entry_id = current_entry_id
    return send_slice(entry_id, slice_id, modality)


def _send_image(dname, fname):
    in_file = pth.join(dname, fname)
    start, ext = fname.split('.')
    out_fname = start + '_rot90' + '.' + ext
    out_file = pth.join(dname, out_fname)
    img = Image.open(in_file)
    img = img.rotate(90)
    img.save(out_file)
    return send_from_directory(dname, out_fname)



from io import BytesIO

def _send_image(dname, fname):
    in_file = pth.join(dname, fname)
    img = Image.open(in_file)
    img = img.rotate(270)
    bio = BytesIO()
    img.save(bio, 'png')
    bio.seek(0)
    return send_file(bio, mimetype='image/png')



@app.route('/slices/<int:entry_id>/<path:slice_id>', defaults={'modality': 't1'})
@app.route('/slices/<int:entry_id>/<path:slice_id>/<modality>')
@disk_cache.cached()
def send_slice(entry_id, slice_id, modality):
    fname = get_example(entry_id).image_fname(slice_id, modality)
    dname, fname = pth.split(fname)
    return _send_image(dname, fname)

@app.route('/thumbnails/<int:entry_id>')
@disk_cache.cached()
def send_thumbnail(entry_id):
    exam_id = entries[entry_id]
    dname = pth.join(params.thumbnail_dir, exam_id)
    fname = 'thumbnail.png'
    return _send_image(dname, fname)

@app.route('/gradcam/<int:entry_id>/<path:slice_id>/<modality>/<target_cls>/<target_region>/<layer>/<norm_type>')
@disk_cache.cached()
def send_gradcam(entry_id, slice_id, modality, target_cls, target_region, layer, norm_type):
    fname = gradcam(entry_id, slice_id, modality, target_cls, target_region, layer, norm_type)
    dname, fname = pth.split(fname)
    return _send_image(dname, fname)

@app.route('/segment/<source>/<tumor_type>/<int:entry_id>/<path:slice_id>', defaults={'modality': 't1'})
@app.route('/segment/<source>/<tumor_type>/<int:entry_id>/<path:slice_id>/<modality>')
@disk_cache.cached()
def send_segment(source, tumor_type, entry_id, slice_id, modality):
    if source == 'gt':
        fname = get_example(entry_id).gt_fname(slice_id, tumor_type)
    elif source == 'pred':
        fname = get_example(entry_id).pred_fname(slice_id, tumor_type)
    elif source == 'counter':
        counter_input = counterfactual(entry_id, slice_id, tumor_type, modality)
        print(counter_input[0].sum())
        fname = get_example(entry_id).counter_fname(slice_id, tumor_type, counter_input, modality, colorscheme='counterfactual')
    dname, fname = pth.split(fname)
    return _send_image(dname, fname)

@app.route('/counterfactual/<ctype>/<tumor_type>/<int:entry_id>/<path:slice_id>', defaults={'modality': 't1'})
@app.route('/counterfactual/<ctype>/<tumor_type>/<int:entry_id>/<path:slice_id>/<modality>')
@disk_cache.cached()
def send_counterfactual(ctype, tumor_type, entry_id, slice_id, modality):
    counter_input = counterfactual(entry_id, slice_id, tumor_type, modality)
    counterfactual_fname, stats = get_example(entry_id).counterfactual(slice_id, tumor_type, counter_input)
    dname, fname = pth.split(counterfactual_fname)
    if ctype == 'image':
        return _send_image(dname, fname)
    elif ctype == 'stats':
        return jsonify(stats)


def get_box_image(box, slice_size):
    mask = np.zeros(slice_size + (4,))
    if box is None:
        return mask
    x, y, w, h = box
    y1 = int(y * mask.shape[0])                                              
    y2 = int((y+h) * mask.shape[0])                                          
    x1 = int(x * mask.shape[1])                                              
    x2 = int((x+w) * mask.shape[1])                                          
    # red
    mask[y1:y2, x1:x2, 0] = 1.
    # alpha
    mask[y1:y2, x1:x2, 3] = 1.
    return mask  


@app.route('/counter_slices/<int:entry_id>/<path:slice_id>/<modality>/<tumor_type>')
@disk_cache.cached()
def send_counterfactual_slice(entry_id, slice_id, modality, tumor_type):
    key = (entry_id, slice_id, modality, tumor_type)
    print(f'reading {key}')
    box = get_example(entry_id).get_box_from_tumor(slice_id, tumor_type)
    fname, _ = get_inpaint(entry_id, slice_id, box, modality)
    dname, fname = pth.split(fname)
    return _send_image(dname, fname)

@app.route('/counter_boxes/<int:entry_id>/<path:slice_id>/<modality>/<tumor_type>')
@disk_cache.cached()
def send_counterfactual_box(entry_id, slice_id, modality, tumor_type):
    key = (entry_id, slice_id, modality, tumor_type)
    print(f'reading {key}')
    box = get_example(entry_id).get_box_from_tumor(slice_id, tumor_type)
    slice_size = get_example(entry_id).example['vis_image'].shape[2:4]
    box_image = get_box_image(box, slice_size)
    id=random_string(12);
    fname = f'counterfactual/{id}.png'
    plt.imsave(fname, box_image)
    dname, fname = pth.split(fname)
    return _send_image(dname, fname)

@app.route('/brats_demo/', methods=['GET','POST'])
def demo_page():
    return send_from_directory('./','brats_demo.html')

@app.route('/colorbar.png')
def send_colorbar():
    return send_from_directory('./', 'colorbar.png')

@app.route('/div_colorbar.png')
def send_div_colorbar():
    return send_from_directory('./', 'div_colormap.png')

#Serving functions
@app.route('/api/', methods=['POST'])
def api():
    req = request.get_data().decode()
    response = jsonrpcserver.dispatch(req)
    return Response(str(response), response.http_status,mimetype='application/json')




if __name__ == "__main__":
    load(params)
    app.run(host='0.0.0.0', threaded=False, port=params.port, debug=True);
