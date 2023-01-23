#!/usr/bin/env pythot
# -*- encoding: utf-8 -*-
# Birth: 2022-11-03 16:17:08.281980901 +0530
# Modify: 2022-11-03 21:12:12.170122589 +0530

"""Data loaders for SBERT"""

import json
import logging
import os
from collections import defaultdict
from itertools import combinations
from random import sample, shuffle
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class SBertTrainerDataset(Dataset):
    """Dataset generator for SBERT. Creates sample pairs following a sampling
    strategy"""

    def __init__(
            self,
            data_paths: list[str],
            targets_paths: list[str],
            unique_labels: Union[None, str] = None,
            similarity: str = "jaccard",
            sample: str = "equal",
            least: int = 5000,
            bins: int = 10,
            min_sim: float = 0.0,
            max_sim: float = 1.0,
            num: int = -1):
        """
        Initialization

        Parameters
        ----------
        data_paths: list[str]
            Paths to load data from
        target_paths: list[str]
            Paths to load targets from
        unique_labels: Union[None, str], default None
            List of unique labels to consider as relevant targets
        similarity: str, default "jaccard"
            Similarity metric to use
        sample: str, default "equal"
            Sampling strategy
        least: int, default 5000
            Least number of elements in a bin to be considered for binning
        bins: int, default 10
            Number of bins to create when sampling
        min_sim: float, default 0.0
            Minimum similarity value, used for creating bins. Ignored whe
            sampling is not specified.
        max_sim: float, default 1.0
            Maximum similarity value
        num: Union[None, int], default None
            Number of data points to consider before making pairs.
        """
        self._data_paths = data_paths
        self._targets_paths = targets_paths
        self._similarity = similarity
        self._sample = sample
        self._least = least
        self._bins = bins
        self._min_sim = min_sim
        self._max_sim = max_sim
        self._num = num

        self._targets_dict = self._get_targets()
        self._text_paths = self._get_fullpaths()

        if unique_labels is None:
            self._unique_labels = self._get_unique_labels()
        else:
            with open(unique_labels, 'r') as f:
                self._unique_labels = f.readlines()
            self._unique_labels = list(
                    filter(None, map(lambda x: x.strip("\n"),
                                     self._unique_labels)))

        if self._similarity == "jaccard":
            self._sim_func = self._jaccard

        self._idx = {i: k for i, k in enumerate(self._text_paths)}

        # Getting all pairs
        logging.info("[DATASET] Generating all combinations and similarities")

        # Computing similarity scores and getting bins
        score = self._pair_sim_scores()
        self._sim_scores, self._all_combinations, self._bin_group_idxs = score

        if self._sample == "equal":
            self._sim_scores, self._all_combinations = self._equal_sample()

    @property
    def unique_labels(self):
        return self._unique_labels

    def __len__(self):
        return len(self._all_combinations)

    def __getitem__(self, idx):
        idx_1, idx_2 = self._all_combinations[idx]
        doc_1 = self._load_data(self._text_paths[self._idx[idx_1]])
        doc_2 = self._load_data(self._text_paths[self._idx[idx_2]])
        sim = self._sim_scores[idx]
        return doc_1, doc_2, sim

    def _get_fullpaths(self):

        doc_paths = {}
        logging.info("[DATASET] Getting all file paths")
        for path in self._data_paths:
            logging.info(f"[DATASET] Loading file paths from {path}")
            for doc_idx in tqdm(os.listdir(path)):
                idx = os.path.splitext(doc_idx)[0]
                doc_paths[idx] = os.path.join(path, doc_idx)

        if self._num != -1:
            doc_paths = {
                    k: v
                    for k, v in list(doc_paths.items())[:self._num]}

        return doc_paths

    def _get_targets(self) -> dict:
        """Get targets of documents from targets paths.

        Returns
        -------
        targets: dict
            Dictionary containing the targets of each document.
        """
        targets = {}
        for path in self._targets_paths:
            with open(path, 'r') as f:
                target = json.load(f)
            targets.update(target)

        return targets

    def _fetch_target(self, doc: str):
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
        target = [label for label in self._unique_labels
                  if label in self._targets_dict[doc]]

        return target

    def _load_data(self, path: str):
        with open(path, 'r') as f:
            data = f.read()
        return data

    def _get_unique_labels(self):

        # Keeping it as a list for ordering ??
        unique_labels = list(set([label
                                  for labels in self._targets_dict.values()
                                  for label in labels]))

        # Extra step to ensure consistency with test dataset
        unique_labels = sorted(unique_labels)

        return unique_labels

    def _jaccard(self, it_a, it_b):
        # Forcing float32 to prevent double conversion
        return torch.tensor([[len(set(it_a).intersection(set(it_b))) *
                            1./len(set(it_a).union(set(it_b)))]],
                            dtype=torch.float32)

    def _pair_sim_scores(self):
        sim_scores = {}
        all_combinations = {}

        bins = np.arange(self._min_sim, self._max_sim,
                         (self._max_sim - self._min_sim) * 1./self._bins)
        bin_group_idxs = defaultdict(list)

        for i, (idx_1, idx_2) in tqdm(enumerate(combinations(
                                          range(len(self._text_paths)), 2))):
            all_combinations[i] = (idx_1, idx_2)
            target_1 = self._fetch_target(self._idx[idx_1])
            target_2 = self._fetch_target(self._idx[idx_2])
            sim_scores[i] = self._sim_func(target_1, target_2)
            bin = np.digitize(sim_scores[i], bins=bins)[0][0]
            bin_group_idxs[bin].append(i)
        return sim_scores, all_combinations, bin_group_idxs

    def _equal_sample(self):
        sizes = list(map(lambda x: len(self._bin_group_idxs[x]),
                         self._bin_group_idxs))
        logging.info(f"[DATASET] bins sizes are {sizes}")
        least = max(min(sizes), self._least)
        selected = []
        for k, v in self._bin_group_idxs.items():
            sample_count = min(len(v), least)
            logging.info(
                    f"[DATASET] Sampling {sample_count} items for bin {k}")
            selected.extend(sample(v, sample_count))

        shuffle(selected)
        logging.info(f"[DATASET] Training on {len(selected)} pairs")
        sim_scores = []
        all_combinations = []
        for idx in selected:
            sim_scores.append(self._sim_scores[idx])
            all_combinations.append(self._all_combinations[idx])
        return sim_scores, all_combinations


class EmbeddingGenerationDataset(Dataset):
    def __init__(
            self,
            data_paths: list[str],
            targets_paths: list[str],
            unique_labels: Union[None, str] = None,
            mode: str = "train"):

        super(EmbeddingGenerationDataset, self).__init__()
        self._data_paths = data_paths
        self._targets_paths = targets_paths
        self._targets_dict = self._get_targets()
        if unique_labels is None:
            self._unique_labels = self._get_unique_labels()
        else:
            with open(unique_labels, 'r') as f:
                self._unique_labels = f.readlines()
            self._unique_labels = list(
                    filter(None, map(lambda x: x.strip("\n"),
                                     self._unique_labels)))

        self._text_paths = self._get_fullpaths()
        # IDs needed for __getitem__
        self._idx = {i: k for i, k in enumerate(self._text_paths)}

        self._mode = mode

    @property
    def unique_labels(self):
        return self._unique_labels

    def __len__(self):
        return len(self._text_paths.keys())

    def __getitem__(self, idx):
        data = self._load_data(self._text_paths[self._idx[idx]])
        target = self._fetch_target(self._idx[idx])

        if self._mode == "train":
            return data, target
        else:
            flname = os.path.splitext(os.path.basename(self._idx[idx]))[0]
            return flname, data

    def _get_fullpaths(self):

        doc_paths = {}
        for path in self._data_paths:
            for doc_idx in os.listdir(path):
                idx = os.path.splitext(doc_idx)[0]
                doc_paths[idx] = os.path.join(path, doc_idx)

        return doc_paths

    def _get_targets(self) -> dict:
        """Get targets of documents from targets paths.

        Returns
        -------
        targets: dict
            Dictionary containing the targets of each document.
        """
        targets = {}
        for path in self._targets_paths:
            with open(path, 'r') as f:
                target = json.load(f)
            targets.update(target)

        return targets

    def _fetch_target(self, doc):
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
        target = torch.tensor([int(label in self._targets_dict.get(doc, []))
                               for label in self._unique_labels])

        return target

    def _load_data(self, path):
        with open(path, 'r') as f:
            data = f.read()
        return data

    def _get_unique_labels(self):

        # Keeping it as a list for ordering ??
        unique_labels = list(set([label
                                  for labels in self._targets_dict.values()
                                  for label in labels]))

        # Extra step to ensure consistency with test dataset
        unique_labels = sorted(unique_labels)

        return unique_labels
