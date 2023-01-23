#!/usr/bin/env python

# Birth: 2022-10-27 13:20:33.303485821 +0530
# Modify: 2022-10-27 21:21:49.499368470 +0530

"""Simple multi-label prediction model with embedding inputs"""

from collections import OrderedDict
import re

import torch
import torch.nn as nn

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharaya@gmail.com"


class SimpleMultiLabelPrediction(nn.Module):
    """Simple multi-label prediction model with embedding inputs"""

    def __init__(self, input_dim, labels, device):
        super(SimpleMultiLabelPrediction, self).__init__()
        self.input_dim = input_dim
        self.labels = [re.sub(r'[^A-Za-z]', '', label)
                       for label in labels]
        self.device = device

        self.projection = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features=self.input_dim,
                                  out_features=1024,
                                  bias=True)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(in_features=1024,
                                  out_features=512,
                                  bias=True)),
            ("relu2", nn.ReLU()),
            ("linear3", nn.Linear(in_features=512,
                                  out_features=256,
                                  bias=True)),
            ("relu3", nn.ReLU()),
            ("linear4", nn.Linear(in_features=256,
                                  out_features=128,
                                  bias=True)),
            ("relu4", nn.ReLU())
            ]))

        self.prediction = nn.ModuleDict({
                k: nn.Linear(in_features=128,
                             out_features=1,
                             bias=True)
                for k in self.labels})

    def forward(self, x):
        preds = torch.tensor([])
        preds = preds.to(self.device)
        proj = self.projection(x)

        for label in self.labels:
            pred = self.prediction[label](proj)
            preds = torch.cat((preds, pred), dim=-1)

        m = nn.Sigmoid()
        preds = m(preds)

        return preds
