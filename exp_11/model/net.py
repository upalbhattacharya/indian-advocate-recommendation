#!usr/bin/env python
# Birth: 2022-11-29 23:17:16.397115050 +0530
# Modify: 2022-11-30 12:46:56.524294408 +0530

"""Databank-based advocate prediction"""

import os
import re
import sys
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "../")
import utils

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


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
        attn = torch.einsum("bij, dkj -> bdik", q_t, k_t)
        dim = list(k_t.shape)[-1]
        s_dim = 1./torch.sqrt(torch.Tensor([dim]))
        s_dim = s_dim.to(self.device)
        attn = torch.einsum("bdij, k -> bdij",
                            attn, s_dim)
        attn = self.softmax(attn)
        proj = torch.einsum("bdii, dij -> bij", attn, v_t)

        return proj


class DatabankModel(nn.Module):
    def __init__(
            self,
            databank_path: Union[str, torch.Tensor],
            labels: list,
            proj_dim: int = 300,
            hidden_dim: int = 768,
            max_length: int = 4096,
            model_name: str = "allenai/longformer-base-4096",
            truncation_side: str = "right",
            device: str = "cpu",
            embed_model_ckpt: str = None):
        super(DatabankModel, self).__init__()
        self.device = device
        self._hidden_dim = hidden_dim
        self._proj_dim = proj_dim
        self._labels = [re.sub(r'[^A-Za-z]', '', label)
                        for label in labels]
        self._databank_path = databank_path
        self._max_length = max_length
        self._truncation_side = truncation_side
        self._model_name = model_name
        self._embed_model_ckpt = embed_model_ckpt
        self.embed_model = AutoModel.from_pretrained(self._model_name)
        if self._embed_model_ckpt is not None:
            _ = utils.load_checkpoint(self.embed_model_ckpt, self.embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
                                        self._model_name,
                                        truncation_side=self._truncation_side)
        self.projection = nn.Linear(
                in_features=self._hidden_dim,
                out_features=self._proj_dim,
                bias=True)

        self.attn = ScaledDotProductAttention(
                features=self._proj_dim,
                device=self.device)
        self.prediction = nn.ModuleDict({
            k: nn.Linear(
                in_features=self._proj_dim * 2,
                out_features=1,
                bias=True)
            for k in self._labels})

        self.device = device
        self.databank = self.create_databank()

    def create_databank(self):
        if type(self._databank_path) == str:
            databank = []
            for fl in os.listdir(self._databank_path):
                data = torch.load(os.path.join(self._databank_path, fl))
                databank.append(data)

            databank = torch.concat(databank, dim=0)
        else:
            databank = self._databank_path
        databank = databank.to(self.device)
        return databank

    def process(self, x):
        tokenized = self.tokenizer(
                x, truncation=True, padding="longest",
                max_length=self._max_length,
                return_tensors="pt")
        return tokenized

    def forward(self, x):
        tokenized = self.process(x)
        tokenized = tokenized.to(self.device)
        preds = torch.tensor([])
        preds = preds.to(self.device)
        proj = self.embed_model(**tokenized)
        proj = proj.last_hidden_state[:, 0, :]
        proj = torch.unsqueeze(proj, dim=1)
        embed = self.projection(proj)
        relu = nn.ReLU()
        embed = relu(embed)
        attn_embed = self.attn(embed, self.databank, self.databank)
        concat_embed = torch.concat((embed, attn_embed), dim=-1)

        m = nn.Sigmoid()
        for label in self._labels:
            pred = self.prediction[label](concat_embed)
            preds = torch.cat((preds, pred), dim=-1)

        preds = m(preds)
        return preds
