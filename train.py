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
# 要保证每次输出的一样，必须在当前文件初始化所有，其他文件不能显示地初始化，只能在函数中初始化输出才不会影响。

learning_rate = 1e-3
weight_decay = 0.0
decay_every = 20

optim_config = { 'lr': learning_rate,
                'weight_decay': weight_decay }


# model = get_model()
EncodingMap = encoding_map()
model = myProtoNet(EncodingMap)
optimer = optim.Adam(model.parameters(), **optim_config)
scheduler = lr_scheduler.StepLR(optimer, step_size=decay_every, gamma=0.5)


savePath = './save/models'
ensure_path(savePath)

max_epoch = 10000
epoch = 0
best_loss = np.inf
wait = 0
train_patience = 200
state = True

omniglot_train_dl = load_ds_dl()
omniglot_val_dl = load_ds_dl(req_dataset='val', n_way=5, n_query=15, )

while epoch < max_epoch and state:

    model.train()
    scheduler.step()

    t_avr_loss, t_avr_acc = Averager(), Averager()
    for n_batch, samples,  in enumerate(omniglot_train_dl, 1):

        optimer.zero_grad()
        loss_, outputs = model.loss(samples)
        loss_.backward()
        optimer.step()

        t_avr_loss.add(outputs['loss']); t_avr_acc.add(outputs['accuracy'])
        print('epoch {}'.format(epoch) + ' batch {}:'.format(n_batch) +
              ' loss:{:0.6f}'.format(outputs['loss']) +
              '; accuracy:{:0.4f}'.format(outputs['accuracy']))
        loss_ = None;  outputs = None
    t_avr_loss, t_avr_acc = t_avr_loss.item(), t_avr_acc.item()


    model.eval()
    v_avr_loss, v_avr_acc = Averager(), Averager()
    for n_batch, samples in enumerate(omniglot_val_dl, 1):
        loss_, outputs = model.loss(samples)

        v_avr_loss.add(outputs['loss']);  v_avr_acc.add(outputs['accuracy'])
        loss_ = None; outputs=None
    v_avr_loss, v_avr_acc = v_avr_loss.item(), v_avr_acc.item()
    print('\nepoch {}, on val, loss={:.4f} acc={:.4f}\n'.format(epoch, v_avr_loss, v_avr_acc))


    if v_avr_loss < best_loss:
        best_loss = v_avr_loss
        print("==> best model (loss = {:0.6f}), saving model...\n".format(best_loss))
        save_model('min-loss', model, savePath)
        wait = 0
    else:
        wait += 1

    if wait > train_patience:
        print("==> patience {:d} exceeded\n".format(train_patience))
        state = False
        save_model('epoch-last', model, savePath)

    epoch += 1


