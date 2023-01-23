#!/usr/bin/env python

"""Ensemble by self-attention and mean pooling"""

import torch
import torch.nn as nn
import re
import logging


class ScaledDotProductAttention(nn.Module):
    def __init__(self, features, device='cpu'):
        super(ScaledDotProductAttention, self).__init__()
        self.features = features
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.query_transform = nn.Linear(in_features=self.features,
                                         out_features=self.features,
                                         bias=False)
        self.key_transform = nn.Linear(in_features=self.features,
                                       out_features=self.features,
                                       bias=False)
        self.value_transform = nn.Linear(in_features=self.features,
                                         out_features=self.features,
                                         bias=False)

    def forward(self, query, key, value):
        q_t = self.query_transform(query)
        k_t = self.key_transform(key)
        v_t = self.value_transform(value)
        attn = torch.einsum("bij, bkj -> bik", q_t, k_t)
        dim = list(k_t.shape)[-1]
        s_dim = 1./torch.sqrt(torch.Tensor([dim]))
        s_dim = s_dim.to(self.device)
        attn = torch.einsum("bij, k -> bij",
                            attn, s_dim)
        attn = self.softmax(attn)
        proj = torch.einsum("bii, bij -> bij", attn, v_t)
        return proj


class EnsembleSelfAttn(nn.Module):
    def __init__(self, labels, input_dims, proj_dim=300, names=None,
                 device='cpu'):
        super(EnsembleSelfAttn, self).__init__()
        self.proj_dim = proj_dim
        self.labels = [re.sub(r'[^A-Za-z]', '', label)
                       for label in labels]
        self.input_dims = input_dims
        self.device = device
        self.set_shapes = False
        self.names = names
        self.softmax = nn.Softmax(dim=0)
        self.self_attn = ScaledDotProductAttention(features=self.proj_dim,
                                                   device=self.device)
        self.prediction = nn.ModuleDict({
            k: nn.Linear(in_features=self.proj_dim,
                         out_features=1,
                         bias=True,)
            for k in self.labels})

        self.projections = nn.ModuleDict(
                {name: nn.Linear(in_features=dim,
                                 out_features=self.proj_dim,
                                 bias=True)
                 for name, dim in zip(self.names, self.input_dims)})

    def forward(self, *x):
        projs = torch.tensor([])
        projs = projs.to(self.device)

        preds = torch.tensor([])
        preds = preds.to(self.device)

        for i, name in enumerate(self.projections):
            proj = torch.unsqueeze(self.projections[name](x[i]), dim=1)
            projs = torch.concat((projs, proj), dim=1)

        logging.info(projs.shape)

        relu = nn.ReLU()
        projs = relu(projs)
        projs = self.self_attn(projs, projs, projs)
        projs = self.mean_pool(projs)

        logging.info(projs.shape)

        m = nn.Sigmoid()
        for label in self.labels:
            pred = self.prediction[label](projs)
            preds = torch.cat((preds, pred), dim=-1)

        preds = m(preds)
        return preds

    def mean_pool(self, projections):
        return torch.mean(projections, dim=1)
