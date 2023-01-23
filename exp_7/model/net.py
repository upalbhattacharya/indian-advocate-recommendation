#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-09-17 23:55:22.914298455 +0530
# Modify: 2022-09-18 00:21:59.829254157 +0530

"""BERT-based model for multi-label classification."""

import re

import torch
import torch.nn as nn

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class EmbedMultiLabel(nn.Module):

    """General embedding-based model for multi-label classification"""

    def __init__(self, labels, device, input_dim,
                 embed_dim=300, mode="train"):
        super(EmbedMultiLabel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device
        self.labels = [re.sub(r'[^A-Za-z]', '', label)
                       for label in labels]
        self.mode = mode
        # Keeping the tokenizer here makes the model better behaved
        # as opposed to using it in the DataLoader
        self.embed = nn.Linear(in_features=self.input_dim,
                               out_features=self.embed_dim,
                               bias=True)

        self.prediction = nn.ModuleDict({
            k: nn.Linear(in_features=self.embed_dim,
                         out_features=1,
                         bias=True,)
            for k in self.labels})

    def forward(self, x):
        preds = torch.tensor([])
        preds = preds.to(self.device)
        relu = nn.ReLU()

        embedding = self.embed(x)
        embedding = relu(embedding)
        # Retaining only the [CLS] token
        m = nn.Sigmoid()
        for label in self.labels:
            pred = self.prediction[label](embedding)
            preds = torch.cat((preds, pred), dim=-1)

        preds = m(preds)
        if self.mode == "train":
            return preds
        else:
            return preds, embedding
