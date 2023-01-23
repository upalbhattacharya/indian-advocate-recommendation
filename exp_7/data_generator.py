#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-09-17 23:55:22.670297935 +0530
# Modify: 2022-09-18 02:11:43.534691739 +0530

"""Data loader for BertMultiLabel"""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class MultiLabelDataset(Dataset):
    def __init__(self, data_paths, targets_paths, unique_labels=None,
                 input_type="np", mode="train"):
        self.data_paths = data_paths
        self.targets_paths = targets_paths
        self.input_type = input_type
        self.targets_dict = self.get_targets()
        if unique_labels is None:
            self.unique_labels = self.get_unique_labels()
        else:
            with open(unique_labels, 'r') as f:
                self.unique_labels = f.readlines()
            self.unique_labels = list(filter(None, map(lambda x: x.strip("\n"),
                                                       self.unique_labels)))

        self.text_paths = self.get_fullpaths()
        # IDs needed for __getitem__
        self.idx = {i: k for i, k in enumerate(self.text_paths)}

        self.mode = mode

    def __len__(self):
        return len(self.text_paths.keys())

    def __getitem__(self, idx):
        data = self.load_data(self.text_paths[self.idx[idx]])
        target = self.fetch_target(self.idx[idx])

        if self.mode == "train":
            return data, target
        else:
            flname = os.path.splitext(os.path.basename(self.idx[idx]))[0]
            return flname, data

    def get_fullpaths(self):

        doc_paths = {}
        for path in self.data_paths:
            for doc_idx in os.listdir(path):
                idx = os.path.splitext(doc_idx)[0]
                doc_paths[idx] = os.path.join(path, doc_idx)

        return doc_paths

    def get_targets(self) -> dict:
        """Get targets of documents from targets paths.

        Returns
        -------
        targets: dict
            Dictionary containing the targets of each document.
        """
        targets = {}
        for path in self.targets_paths:
            with open(path, 'r') as f:
                target = json.load(f)
            targets.update(target)

        return targets

    def fetch_target(self, doc):
        """Return target tensors for given batch

        Parameters
        ----------
        batch : list
            List of document IDs for a batch.

        Returns
        -------
        targets : torch.nn.Tensor
            Tensor containing the target tensors.
        """
        target = torch.tensor([int(label in self.targets_dict[doc])
                               for label in self.unique_labels])

        return target

    def load_data(self, path):
        if self.input_type == "np":
            embed = np.load(path, allow_pickle=True)
            # Handle bad numpy saved sparse data
            if len(embed.shape) == 0:
                # TODO: upalbhattacharya:
                # Efficiently use sparse tensors. Can cause memory overhead
                embed = embed.tolist()[0]
                embed = embed.todense()
            embed = torch.from_numpy(embed)
            embed = embed.float()
            embed = torch.squeeze(embed)
        elif self.input_type == "pt":
            embed = torch.load(path)
            embed = torch.squeeze(embed)
        else:
            raise FileNotFoundError(f"{path} does not exist")
        return embed

    def get_unique_labels(self):

        # Keeping it as a list for ordering ??
        unique_labels = list(set([label
                                  for labels in self.targets_dict.values()
                                  for label in labels]))

        # Extra step to ensure consistency with test dataset
        unique_labels = sorted(unique_labels)

        return unique_labels
