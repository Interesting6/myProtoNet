#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import torch

import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

from loadData import load_ds_dl
from protoNet import myProtoNet, encoding_map, get_model

from myutils import Averager, ensure_path, save_model


torch.manual_seed(1234)

test_class = 5
test_support = 5
test_query = 15
n_episode = 1000



omniglot_test_dl = load_ds_dl('omniglot', 'test', test_class,
                              n_episode, test_support, test_query)


modelPath = './save/models/min-loss.pth'
model = get_model()
model.load_state_dict(torch.load(modelPath))
model.eval()

average_loss = Averager()
average_acc = Averager()
for n_batch, batch_sample in  enumerate(omniglot_test_dl, 1):
    _, outs = model.loss(batch_sample)
    average_loss.add(outs['loss'])
    average_acc.add(outs['accuracy'])
    print('batch {}:'.format(n_batch) +
          ' loss:{:0.6f}'.format(outs['loss']) +
          '; accuracy:{:0.4f}'.format(outs['accuracy']))

average_loss = average_loss.item()
average_acc = average_acc.item()
print('\non test:, average loss={:.4f} average acc={:.4f}\n'.format( average_loss, average_acc))





