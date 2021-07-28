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
sys.path.append('./attUNet_BraTS_Segmentation_2D/')
from network import R2AttU_Net,AttU_Net

torch.autograd.set_detect_anomaly(True)



class AttUNetWrapper():

    def __init__(self):
        self.model_path = './attUNet_BraTS_Segmentation_2D/model_weights'
        self.weight_pkl = 'AttU_Net-150-0.0001-2-0.3042.-droupout-config1-p02.pkl'
        # TODO: allow cpu?
        self.device = torch.device('cuda')
        self.build_model()


    def build_model(self):
        ### build the model

        unet_path = os.path.join(self.model_path,self.weight_pkl)
        if not os.path.exists(unet_path):
            raise ValueError('weight file {} does not exist.'.format(unet_path))

        print('model path: {}'.format(unet_path))
        self.unet = AttU_Net(img_ch=4,output_ch=4,dropout=True,dropout_rate=0.2)
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
