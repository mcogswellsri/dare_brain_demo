import os
import pickle
import argparse
import tempfile
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import sys
sys.path.append('./demo_unet/')
from pytorch_unet import UNet

torch.autograd.set_detect_anomaly(True)



class UNet2dWrapper():

    def __init__(self):
        self.model_path = './demo_unet'
        self.weight_pkl = 'BRATS_unet_0.0001_100_40_epochs.pkl'
        self.device = torch.device('cuda')
        self.build_model()

    def build_model(self):
        ### build the model
        unet_path = os.path.join(self.model_path,self.weight_pkl)
        if not os.path.exists(unet_path):
            raise ValueError('weight file {} does not exist.'.format(unet_path))

        print('model path: {}'.format(unet_path))
        # 4 input modalities, 6 output labels
        self.unet = UNet(4, 6)
        self.unet.load_state_dict(torch.load(unet_path))
        self.unet.to(self.device)
        print('models pushed to the device.')

    def predict(self):
        print('loading data from {} {} ...'.format(self.input_data,config.output_data))
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

        inp = torch.tensor(inp).type(torch.float).to(self.device)
        unet_seg = self.unet.forward(inp)
