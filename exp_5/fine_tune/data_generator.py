#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-07-21 16:48:02.702290180 +0530
# Modify: 2022-07-22 00:09:20.168224143 +0530

"""Data loaders for SBERT"""

import json
import os
from itertools import combinations

import torch
from torch.utils.data import Dataset

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class EmbeddingGenerationDataset(Dataset):
    def __init__(self, data_paths, targets_paths, unique_labels=None,
                 mode="train"):
        self.data_paths = data_paths
        self.targets_paths = targets_paths
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

        self.targets_dict = self.get_targets()
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
        with open(path, 'r') as f:
            data = f.read()
        return data

    def get_unique_labels(self):

        # Keeping it as a list for ordering ??
        unique_labels = list(set([label
                                  for labels in self.targets_dict.values()
                                  for label in labels]))

        # Extra step to ensure consistency with test dataset
        unique_labels = sorted(unique_labels)

        return unique_labels


class SBertTrainerDataset(Dataset):
    def __init__(self, data_paths, targets_paths, unique_labels=None,
                 mode="train", similarity="jaccard"):
        self.data_paths = data_paths
        self.targets_paths = targets_paths
        self.targets_dict = self.get_targets()
        if unique_labels is None:
            self.unique_labels = self.get_unique_labels()
        else:
            with open(unique_labels, 'r') as f:
                self.unique_labels = f.readlines()
            self.unique_labels = list(filter(None, map(lambda x: x.strip("\n"),
                                                       self.unique_labels)))

        self.text_paths = self.get_fullpaths()
        self.all_combinations = [(idx_1, idx_2)
                                 for idx_1, idx_2 in combinations(
                                            range(len(self.text_paths)), 2)]
        # IDs needed for __getitem__
        self.idx = {i: k for i, k in enumerate(self.text_paths)}
        self.mode = mode
        self.similarity = similarity

    def __len__(self):
        return len(self.all_combinations)

    def __getitem__(self, idx):
        idx_1, idx_2 = self.all_combinations[idx]
        doc_1 = self.load_data(self.text_paths[self.idx[idx_1]])
        doc_2 = self.load_data(self.text_paths[self.idx[idx_2]])
        target_1 = self.fetch_target(self.idx[idx_1])
        target_2 = self.fetch_target(self.idx[idx_2])

        if self.similarity == "jaccard":
            sim = self.jaccard(target_1, target_2)

        if self.mode == "train":
            return doc_1, doc_2, sim

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
        target = [label for label in self.unique_labels
                  if label in self.targets_dict[doc]]

        return target

    def load_data(self, path):
        with open(path, 'r') as f:
            data = f.read()
        return data

    def get_unique_labels(self):

        # Keeping it as a list for ordering ??
        unique_labels = list(set([label
                                  for labels in self.targets_dict.values()
                                  for label in labels]))

        # Extra step to ensure consistency with test dataset
        unique_labels = sorted(unique_labels)

        return unique_labels

    def jaccard(self, it_a, it_b):

        return (len(set(it_a).intersection(set(it_b))) *
                1./len(set(it_a).union(set(it_b))))
