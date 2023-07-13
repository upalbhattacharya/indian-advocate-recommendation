#!/usr/bin/env python
# Birth: 2022-10-28 13:10:35.413914085 +0530
# Modify: 2022-11-03 18:21:38.857962240 +0530

"""Dataset for Multi-Task Setup"""

import json
import os
from itertools import chain
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = ""
__email__ = "upal.bhattacharya@gmail.com"


class MultiTaskDataset(Dataset):
    """Dataset for Multi-Task Setup"""

    def __init__(self,
                 data_paths: list[str],
                 targets_paths: dict[str, list[str]],
                 unique_labels: dict[str, list[str]]) -> None:

        super(MultiTaskDataset, self).__init__()
        self._data_paths = data_paths

        if not isinstance(targets_paths, dict):
            raise TypeError(("'targets_paths' must be of type 'dict'. "
                            f"Found {type(targets_paths)}"))

        if not all([k in ["adv", "area"] for k in targets_paths.keys()]):
            raise ValueError(("Only targets_paths keys 'adv' and 'area' "
                              f"expected. Got {targets_paths.keys()}"))

        self._targets_paths = targets_paths
        self._targets_dict = self._get_targets()

        if not isinstance(unique_labels, dict):
            raise TypeError(("'unique_labels' must be of type 'dict'. "
                            f"Found {type(unique_labels)}"))

        if not all([k in ["adv", "area"] for k in unique_labels.keys()]):
            raise ValueError(("Only unique_labels keys 'adv' and 'area' "
                              f"expected. Got {unique_labels.keys()}"))

        self._unique_labels = self._get_unique_labels(unique_labels)
        self._input_paths = self._get_fullpaths()
        self._idx = {i: k for i, k in enumerate(self._input_paths.keys())}

    def __len__(self) -> int:
        return len(self._input_paths.keys())

    def __getitem__(self, idx: int) -> tuple[
            Union[str, torch.Tensor],
            torch.Tensor, torch.Tensor]:
        data = self._fetch_data(self._input_paths[self._idx[idx]])
        target_adv, target_area = self._fetch_targets(self._idx[idx])
        flname = os.path.splitext(os.path.basename(self.idx[idx]))[0]
        return data, target_adv, target_area, flname

    @property
    def unique_labels(self) -> dict[str, str]:
        """Gettr for unique_labels"""
        return self._unique_labels

    def _get_targets(self) -> dict[str, dict[str, Union[list[str], None]]]:

        targets_paths = {}
        targets_paths_adv = {}
        targets_paths_area = {}

        for path in self._targets_paths["adv"]:
            with open(path, 'r') as f:
                target = json.load(f)
            targets_paths_adv.update(target)

        for path in self._targets_paths["area"]:
            with open(path, 'r') as f:
                target = json.load(f)
            targets_paths_area.update(target)

        for key in targets_paths_adv.keys():
            targets_paths[key] = {
                    'adv': targets_paths_adv.get(key, []),
                    'area': targets_paths_area.get(key, [])
                    }

        return targets_paths

    def _get_unique_labels(self,
                           unique_labels: dict[
                            str, Union[str, None]]) -> dict[str, list[str]]:
        u_labels = {}
        for key in unique_labels.keys():
            if unique_labels[key] is not None:
                with open(unique_labels[key], 'r') as f:
                    labels = f.readlines()
                labels = list(filter(None,
                                     map(lambda x: x.strip("\n"), labels)))
                u_labels[key] = labels
            else:
                labels = set(list(chain.from_iterable(
                    list(filter(None, map(lambda x: self._targets_dict[x][key],
                             self._targets_dict.keys()))))))
                u_labels[key] = sorted(labels)
        return u_labels

    def _get_fullpaths(self) -> dict[str, str]:
        input_paths = {}
        for path in self._data_paths:
            for fl in os.listdir(path):
                idx = os.path.splitext(fl)[0]
                input_paths[idx] = os.path.join(path, fl)
        return input_paths

    def _fetch_data(self, fl: str) -> Union[str, torch.Tensor]:
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

    def _fetch_targets(self, fl: str) -> tuple[Union[torch.Tensor, None],
                                               Union[torch.Tensor, None]]:
        target_adv = torch.tensor([
            int(label in self._targets_dict[fl]["adv"])
            for label in self._unique_labels["adv"]])

        target_area = torch.tensor([
            int(label in self._targets_dict[fl]["area"])
            for label in self._unique_labels["area"]])

        return target_adv, target_area
