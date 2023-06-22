#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-05-17 09:38:52.711351951 +0530
# Modify: 2022-05-17 16:31:13.567617509 +0530

"""BERT-based model for multi-label classification."""

import re

import torch
import torch.nn as nn

from transformers import LongformerModel, LongformerTokenizer

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class LongformerMultiLabel(nn.Module):

    """Longformer-based model for multi-label classification"""

    def __init__(self, labels, device, hidden_size=768, max_length=4096,
                 longformer_model_name="allenai/longformer-base-4096",
                 truncation_side="right", mode="train"):
        super(LongformerMultiLabel, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.max_length = max_length
        self.labels = [re.sub(r'[^A-Za-z]', '', label)
                       for label in labels]
        self.longformer_model_name = longformer_model_name
        self.longformer_model = LongformerModel.from_pretrained(self.longformer_model_name)
        self.truncation_side = truncation_side
        self.mode = mode
        # Keeping the tokenizer here makes the model better behaved
        # as opposed to using it in the DataLoader
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained(
                                        self.longformer_model_name,
                                        truncation_side=self.truncation_side)

        self.prediction = nn.ModuleDict({
            k: nn.Linear(in_features=self.hidden_size,
                         out_features=1,
                         bias=True,)
            for k in self.labels})

    def process(self, x):
        tokenized = self.longformer_tokenizer(x, truncation=True, padding="longest",
                                              max_length=self.max_length,
                                              return_tensors="pt")
        return tokenized

    def forward(self, x):
        tokenized = self.process(x)
        tokenized = tokenized.to(self.device)
        preds = torch.tensor([])
        preds = preds.to(self.device)

        encoding = self.longformer_model(**tokenized)
        # Retaining only the [CLS] token
        cls = encoding.last_hidden_state[:, 0, :]
        m = nn.Sigmoid()
        for label in self.labels:
            pred = self.prediction[label](cls)
            preds = torch.cat((preds, pred), dim=-1)

        preds = m(preds)
        if self.mode == "train":
            return preds
        else:
            return preds, cls