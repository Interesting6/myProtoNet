#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from myutils import euclidean_dist




class Flatten(nn.Module):
    def __init__(self,):
        super(Flatten, self).__init__()

    def forward(self, x):
        # x.size(0): number of a batch set x
        return x.view(x.size(0), -1)



def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def encoding_map(x_dim=(1, 28, 28), hid_dim=64, out_dim=64):
    return nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, out_dim),
        Flatten()
    )




class myProtoNet(nn.Module):
    def __init__(self, encoding_map, ):
        super(myProtoNet, self).__init__()
        self.encoder = encoding_map

    def loss(self, batch_sample):
        xs = batch_sample['support_set']  # (n_class, n_support, 1, w, h)
        xq = batch_sample['query_set']

        n_class = xs.size(0)
        assert n_class==xq.size(0)
        n_support, n_query = xs.size(1), xq.size(1)

        y_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()

        x = torch.cat((xs.view(n_class*n_support, *xs.size()[2:]),
                       xq.view(n_class*n_query, *xq.size()[2:]) ), dim=0)
        # x: (n_class*(n_support+n_query), 1, 28, 28) 将所有样本放在一起。

        z = self.encoder(x) # z: (n_class*(n_support+n_query), 64)


        # prediction
        z_proto = z[:n_class*n_support].view(n_class, n_support, -1).mean(1) # (n_class, 64)
        zq = z[n_class*n_support:]
        dists = euclidean_dist(zq, z_proto) # (n_class*n_query, n_class)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1) # (n_class, n_query, n_class)
        _, y_hat = log_p_y.max(2)  # (n_class, n_query ) 每行为一类中的query样本的标签
        acc_ = torch.eq(y_hat, y_inds.squeeze()).float().mean()

        # calculate loss
        loss_ = -log_p_y.gather(dim=2, index=y_inds) # shape same as y_inds: (n_class, n_query, 1)
        loss_ = loss_.squeeze().view(-1).mean() # (n_class, n_query)->(n_class*n_query)->(1)

        return loss_, {
            'loss': loss_.item(),
            'accuracy': acc_.item(),
        }


def get_model():
    EncodingMap = encoding_map()
    return myProtoNet(EncodingMap)



