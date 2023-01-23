#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-11-03 16:17:08.573981365 +0530
# Modify: 2022-11-03 17:06:56.786022721 +0530

"""SBERT model for embedding generation."""

from typing import Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class SBertEmbedding(nn.Module):

    """SBERT model for embedding generation"""

    def __init__(
            self,
            device: str,
            hidden_size: int = 768,
            max_length: int = 512,
            model_name: str = "sentence-transformers/all-distilroberta-v1",
            truncation_side: str = "right",
            mode: str = "train",
            pooling: str = "mean"):

        super(SBertEmbedding, self).__init__()
        self._hidden_size = hidden_size
        self.device = device
        self._max_length = max_length
        self._model_name = model_name
        self._sbert_model = AutoModel.from_pretrained(self._model_name)
        self._truncation_side = truncation_side
        self._mode = mode
        self._pooling = pooling
        # Keeping the tokenizer here makes the model better behaved
        # as opposed to using it in the DataLoader
        self._sbert_tokenizer = AutoTokenizer.from_pretrained(
                                        self._model_name,
                                        truncation_side=self._truncation_side)

    def _process(self, x: Union[list[str], str]):
        tokenized = self.sbert_tokenizer(x, truncation=True, padding="longest",
                                         max_length=self.max_length,
                                         return_tensors="pt")
        return tokenized

    def _mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                                            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        tokenized = self._process(x)
        tokenized = tokenized.to(self.device)
        encoding = self._sbert_model(**tokenized)

        if self._pooling == "mean":
            embeddings = self._mean_pooling(
                    encoding,
                    tokenized["attention_mask"])
        else:
            # CLS
            embeddings = encoding.last_hidden_state[:, 0, :]
        return embeddings
