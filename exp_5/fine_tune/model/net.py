#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-07-21 16:48:02.886297359 +0530
# Modify: 2022-07-21 19:44:03.709002814 +0530

"""SBERT model for embedding generation."""

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

    def __init__(self, device, hidden_size=768, max_length=512,
                 sbert_model_name="sentence-transformers/",
                 truncation_side="right", mode="train", pooling="mean"):
        super(SBertEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.max_length = max_length
        self.sbert_model_name = sbert_model_name
        self.sbert_model = AutoModel.from_pretrained(self.sbert_model_name)
        self.truncation_side = truncation_side
        self.mode = mode
        self.pooling = pooling
        # Keeping the tokenizer here makes the model better behaved
        # as opposed to using it in the DataLoader
        self.sbert_tokenizer = AutoTokenizer.from_pretrained(
                                        self.sbert_model_name,
                                        truncation_side=self.truncation_side)

    def process(self, x):
        tokenized = self.sbert_tokenizer(x, truncation=True, padding="longest",
                                         max_length=self.max_length,
                                         return_tensors="pt")
        return tokenized

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                                            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        tokenized = self.process(x)
        tokenized = tokenized.to(self.device)
        encoding = self.sbert_model(**tokenized)

        if self.pooling == "mean":
            embeddings = self.mean_pooling(encoding,
                                           tokenized["attention_mask"])
        else:
            # CLS
            embeddings = encoding.last_hidden_state[:, 0, :]
        return embeddings
