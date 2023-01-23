#!/usr/bin/env python

"""Custom Dataset for embeddings"""

import json
import os
from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class SimpleDataset(Dataset):
    """Custom Dataset for loading embeddings/text"""

    def __init__(self, data_paths, targets_paths, unique_labels=None):
        self._data_paths = data_paths
        self._targets_paths = targets_paths
        self._targets_dict = self._get_targets()
        if unique_labels is not None:
            with open(unique_labels, 'r') as f:
                self._unique_labels = f.readlines()
            self._unique_labels = list(filter(
                    None, map(lambda x: x.strip("\n"), self._unique_labels)))
        else:
            self._unique_labels = self._get_unique_labels()
        self._embed_paths = self._get_fullpaths()
        self._idx = {i: k for i, k in enumerate(self._embed_paths)}

    @property
    def unique_labels(self):
        """Get unique labels"""
        return self._unique_labels

    def __len__(self):
        return len(self._embed_paths.keys())

    def __getitem__(self, idx):

        target = self._fetch_target(self._idx[idx])
        data = self._fetch_data(self._embed_paths[self._idx[idx]])

        return data, target

    def _get_targets(self):
        """Get targets from targets_paths"""
        targets = {}
        for target_path in self._targets_paths:
            with open(target_path, 'r') as f:
                target = json.load(f)
            targets.update(target)
        return targets

    def _get_unique_labels(self):
        """Get unique labels if not specified"""

        labels = set(list(chain.from_iterable(self._targets_dict.values())))
        labels = sorted(labels)
        return labels

    def _get_fullpaths(self):
        """Get full paths of all embeddings"""
        embed_paths = {}
        for path in self._data_paths:
            for fl in os.listdir(path):
                flname = os.path.splitext(fl)[0]
                embed_paths[flname] = os.path.join(path, fl)
        return embed_paths

    def _fetch_target(self, fl):
        target = torch.tensor([int(label in self._targets_dict[fl])
                               for label in self._unique_labels])
        return target

    def _fetch_data(self, fl):
        ext = os.path.splitext(fl)[-1]
        if "npy" in ext:
            data = np.load(fl, allow_pickle=True)
            if len(data.shape) == 0:
                data = data.tolist()[0]
                data = data.todense()
            data = torch.from_numpy(data)
            if len(data.shape) == 1:
                data = torch.unsqueeze(data, dim=0)
        elif "pt" in ext:
            data = torch.load(fl)
            if len(data.shape) == 1:
                data = torch.unsqeeze(data, dim=0)
        elif "txt" in ext:
            with open(fl, 'r') as f:
                data = f.read()
        else:
            raise ValueError(f"'{ext}' filetype of file {fl} incompatible ")
        return data
