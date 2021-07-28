#!/usr/bin/env python
# coding: utf-8

from network import R2AttU_Net,AttU_Net
import torch
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import argparse

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./model_weights')
parser.add_argument('--weight_pkl', type=str, default='AttU_Net-150-0.0001-2-0.3042.-droupout-config1-p02.pkl')
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--input_data', type=str, default='./data/BraTS_sample_input.pkl')
parser.add_argument('--output_data', type=str, default='./data/BraTS_sample_output.pkl')
config = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def plot_channels(data,start_indx=0,vmin=None,vmax=None,title=None, channel_labels = None,filename=None):
    fig = plt.figure(figsize=(20,100))
    ax = None
    if data.shape[0] == 1:
        data_plot = data[0]
    else:
        data_plot = data
    for i in range(start_indx,4):
        ax = fig.add_subplot(1, 4, i+1, sharex=ax, sharey=ax)
        if title:
            if channel_labels:
                ttl = title+' {}'.format(channel_labels[i])
            else:
                ttl = title+' channel {}'.format(i)
            ax.title.set_text(ttl)
        if vmin and vmax:
            h = ax.imshow(data_plot[i],cmap='gray')
        else:
            h = ax.imshow(data_plot[i],vmin=vmin, vmax=vmax,cmap='gray')
        ax.autoscale(True)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(h, cax=cax)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        plt.savefig('output.png', dpi=300, bbox_inches="tight")


### build the model

unet_path = os.path.join(config.model_path,config.weight_pkl)
if not os.path.exists(unet_path):
    raise ValueError('weight file {} does not exist.'.format(unet_path))

print('model path: {}'.format(unet_path))
unet = AttU_Net(img_ch=4,output_ch=4,dropout=True,dropout_rate=0.2)
unet.load_state_dict(torch.load(unet_path))
unet.to(device)
print('models pushed to the device.')

output_dir = config.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('loading data from {} {} ...'.format(config.input_data,config.output_data))
if not os.path.exists(config.input_data):
    raise ValueError('data file {} does not exist.'.format(config.input_data))
if not os.path.exists(config.output_data):
    raise ValueError('data file {} does not exist.'.format(config.output_data))
test_inps = pickle.load(open(config.input_data,'rb'))
test_lbls = pickle.load(open(config.output_data,'rb'))


frame_ind = 75  # other interesting frames: 85,95,105,115
print('running frame {}'.format(frame_ind))

inp = test_inps[0,frame_ind:frame_ind+1,...]
lbl = test_lbls[0,frame_ind:frame_ind+1,...]

import pdb; pdb.set_trace()

unet_seg = unet.forward(torch.tensor(inp).type(torch.float).to(device))
		    
# plot inputs
input_filename = os.path.join(output_dir,'input_{}.png'.format(str(frame_ind).zfill(3)))
input_modes = ['T1','T2','T1ce','Flair']
plot_channels(inp,title='Input',channel_labels = input_modes, filename=input_filename)
print('input saved to {}'.format(input_filename))

# plot segmentation output
output_filename = os.path.join(output_dir,'output_{}.png'.format(str(frame_ind).zfill(3)))
output_labels = ['Background', 'NCR/NET' , 'ED' , 'ET']
plot_channels(unet_seg.cpu().detach().numpy(),vmin=0,vmax=1,title='Seg.',channel_labels = output_labels, filename=output_filename)
print('output saved to {}'.format(output_filename))
