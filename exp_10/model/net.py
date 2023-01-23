#!/usr/bin/env python

# Birth: 2022-10-28 10:14:11.765739581 +0530
# Modify: 2022-10-28 22:08:49.143751133 +0530

"""Simple Multi-Task Prediction Model with two prediction outputs"""

import re
from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class SimpleMultiTaskMultiLabelPrediction(nn.Module):
    """Simple Multi-Task Prediction Model with two prediction outputs"""

    def __init__(self,
                 labels: dict[str, list[str]],
                 max_length: int = 4096,
                 truncation_side: str = "right",
                 model_name: str = "allenai/longformer-base-4096",
                 mode: str = "train",
                 device: str = "cuda"):
        super(SimpleMultiTaskMultiLabelPrediction, self).__init__()

        self._labels = {k: [re.sub(r"[^A-Za-z]", "", label)
                            for label in v]
                        for k, v in labels.items()}

        if truncation_side not in ["right", "left"]:
            raise ValueError(("Truncation must be 'right' or 'left'. "
                              f"Found {truncation_side}"))
        self._truncation_side = truncation_side
        self._max_length = max_length
        if mode not in ["train", "generate"]:
            raise ValueError(("Mode must be 'train' or 'generate'. "
                              f"Found {mode}"))
        self._mode = mode
        self._model_name = model_name
        self.device = device

        self._tokenizer = AutoTokenizer.from_pretrained(
                                self._model_name,
                                truncation_side=self._truncation_side)

        # Model layers
        self.embedding = AutoModel.from_pretrained(self._model_name)

        self.projection_adv = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features=768,
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
            ("relu4", nn.ReLU()),
            ]))

        self.projection_area = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features=768,
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
            ("relu4", nn.ReLU()),
            ]))

        self.prediction_adv = nn.ModuleDict({
             k: nn.Linear(in_features=128,
                          out_features=1,
                          bias=True)
             for k in self._labels["adv"]})

        self.prediction_area = nn.ModuleDict({
             k: nn.Linear(in_features=128,
                          out_features=1,
                          bias=True)
             for k in self._labels["area"]})

    def _process(self, x):
        tokenized = self._tokenizer(x, truncation=True,
                                    padding="longest",
                                    max_length=self._max_length,
                                    return_tensors="pt")
        return tokenized

    def forward(self, x: list[str]) -> Union[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        tokenized = self._process(x)
        tokenized = tokenized.to(self.device)

        preds_adv = torch.tensor([])
        preds_adv = preds_adv.to(self.device)

        preds_area = torch.tensor([])
        preds_area = preds_area.to(self.device)

        embedding = self.embedding(**tokenized)
        # Using only [CLS] token for prediction
        cls = embedding.last_hidden_state[:, 0, :]
        m = nn.Sigmoid()

        projs_adv = self.projection_adv(cls)
        projs_area = self.projection_area(cls)

        for label in self._labels["adv"]:
            pred_adv = self.prediction_adv[label](projs_adv)
            preds_adv = torch.cat((preds_adv, pred_adv), dim=-1)

        for label in self._labels["area"]:
            pred_area = self.prediction_area[label](projs_area)
            preds_area = torch.cat((preds_area, pred_area), dim=-1)

        preds_adv = m(preds_adv)
        preds_area = m(preds_area)

        if self._mode == "generate":
            return preds_adv, preds_area, cls

        return preds_adv, preds_area
