#!/usr/bin/env python

"""Ensemble by self-attention and mean pooling"""

import torch
import torch.nn as nn
import re


class EnsembleConcatenation(nn.Module):
    def __init__(self, labels, input_dims, proj_dim=300, names=None,
                 device='cpu'):
        super(EnsembleConcatenation, self).__init__()
        self.proj_dim = proj_dim
        self.labels = [re.sub(r'[^A-Za-z]', '', label)
                       for label in labels]
        self.input_dims = input_dims
        self.device = device
        self.set_shapes = False
        self.names = names
        self.softmax = nn.Softmax(dim=0)

        self.linear = nn.Linear(in_features=sum(self.input_dims),
                                out_features=self.proj_dim,
                                bias=True)

        self.prediction = nn.ModuleDict({
            k: nn.Linear(in_features=self.proj_dim,
                         out_features=1,
                         bias=True,)
            for k in self.labels})

    def forward(self, *x):
        projs = torch.tensor([])
        projs = projs.to(self.device)

        preds = torch.tensor([])
        preds = preds.to(self.device)

        for proj in x:
            projs = torch.concat((projs, proj), dim=-1)

        projs = self.linear(projs)

        relu = nn.ReLU()
        projs = relu(projs)

        m = nn.Sigmoid()
        for label in self.labels:
            pred = self.prediction[label](projs)
            preds = torch.cat((preds, pred), dim=-1)

        preds = m(preds)
        return preds
