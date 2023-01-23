#!/usr/bin/env python
# Birth: 2022-07-23 12:35:45.111002924 +0530
# Modify: 2022-07-23 12:36:06.047708319 +0530
"""Different losses for SBERT."""

import torch
import torch.nn as nn

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class LossModel(nn.Module):
    """Append an appropriate loss to the model."""

    def __init__(self, model, loss_fn=nn.MSELoss(),
                 similarity_fn=torch.cosine_similarity):
        super(LossModel, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.similarity_fn = similarity_fn

    def forward(self, x, labels):
        doc_1, doc_2 = x
        embed_1 = self.model(doc_1)
        embed_2 = self.model(doc_2)
        similarity = self.similarity_fn(embed_1, embed_2)
        loss = self.loss_fn(similarity, labels.view(-1))

        return loss
