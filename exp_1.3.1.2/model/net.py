#!/home/workboots/VirtualEnvs/aiml/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-01 15:34:37.375743601 +0530
# Modify: 2022-03-01 15:34:37.385743601 +0530

"""Prediction model of case offences using HAN."""

import re

import torch
import torch.nn as nn

from .han import HAN

from pathlib import Path


class HANPrediction(nn.Module):
    """HAN-based multi-label prediction model."""

    def __init__(self, input_size, hidden_dim, labels, device):
        super(HANPrediction, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.labels = [re.sub(r'[^A-Za-z]+', '', label)
                       for label in labels]

        self.device = device

        self.han = HAN(input_size=self.input_size,
                       hidden_dim=self.hidden_dim,
                       device=self.device)

        self.prediction = nn.ModuleDict({
            k: nn.Linear(in_features=2*self.hidden_dim,
                         out_features=1,
                         bias=True)
            for k in self.labels})

    def forward(self, x):
        preds = torch.tensor([])
        preds = preds.to(self.device)
        output = self.han(x)
        m = nn.Sigmoid()
        for k in self.labels:
            pred = self.prediction[k](output)
            preds = torch.cat((preds, pred), dim=-1)

        preds = m(preds)
        return preds
