#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

"""
This script read the picture to tensor, and uses resize, centerCrop, Normalize transforms.
Last the pictures will convert to a ndarray and save it, for the purpose of reducing time of loading
dataset time in train or test.
"""


import os, sys, pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as tvtsfm
from functools import partial


dataPath = '/home/cheny/DataSet/miniImagenet'
miniImagePath = dataPath + '/images'
splitPath = dataPath + '/splits'
miniImageNpPath = dataPath + '/npdata'

splits = ['train', 'val', 'test']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




tsfm = tvtsfm.Compose([
    tvtsfm.Resize(84),
    tvtsfm.CenterCrop(84),
    tvtsfm.ToTensor(),
    tvtsfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(path, spt='test', tsfms=tsfm):
    name = path.split('.')[0]
    path = os.path.join(miniImagePath, path)
    image = Image.open(path)
    image = tsfms(image)#.to(device)
    image = image.numpy()#.cpu().numpy()
    save_arr_path = os.path.join(miniImageNpPath, spt, name + '.npy')
    np.save(save_arr_path, image)
    return None




for spt in splits:
    split_csv_path = os.path.join(splitPath, spt + '.csv')
    split_csv = pd.read_csv(split_csv_path)
    file_name = split_csv['filename'].values
    # print(split_csv_path)
    preprocess = partial(preprocess, spt=spt, tsfms=tsfm)
    images = list(map(preprocess, file_name))

