#!/usr/bin/env pythot
# -*- encoding: utf-8 -*-
# Birth: 2022-07-21 16:48:02.702290180 +0530
# Modify: 2022-07-25 13:05:35.545682643 +0530

"""Data loaders for SBERT"""

import json
import logging
import os
from itertools import combinations, groupby
from random import sample, shuffle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
    """Dataset generator for SBERT. Creates sample pairs following a sampling
    strategy"""

    def __init__(self, data_paths, targets_paths, unique_labels=None,
                 similarity="jaccard", sample="equal", least=5000,
                 steps=10, min_sim=0.0, max_sim=1.0):
        """
        Initialization

        Parameters
        ----------
        data_paths: list
            Paths to load data from
        target_paths: list
            Paths to load targets from
        unique_labels: list, default None
            List of unique labels to consider as relevant targets
        similarity: str, default "jaccard"
            Similarity metric to use
        sample: str, default "equal"
            Sampling strategy
        least: int, default 5000
            Least number of elements in a bin to be considered for binning
        steps: int, default 10
            Number of bins to create when sampling
        min_sim: float, default 0.0
            Minimum similarity value, used for creating bins. Ignored whe
            sampling is not specified.
        max_sim: float, default 1.0
            Maximum similarity value
        """
        self.data_paths = data_paths
        self.targets_paths = targets_paths
        self.similarity = similarity
        self.sample = sample
        self.least = least
        self.steps = steps
        self.min_sim = min_sim
        self.max_sim = max_sim

        self.targets_dict = self.get_targets()
        self.text_paths = self.get_fullpaths()

        if unique_labels is None:
            self.unique_labels = self.get_unique_labels()
        else:
            with open(unique_labels, 'r') as f:
                self.unique_labels = f.readlines()
            self.unique_labels = list(filter(None, map(lambda x: x.strip("\n"),
                                                       self.unique_labels)))

        if self.similarity == "jaccard":
            self.sim_func = self.jaccard

        # Getting all pairs
        logging.info("[DATASET] Generating all combinations")
        #  self.all_combinations = [(idx_1, idx_2)
                                 #  for idx_1, idx_2 in combinations(
                                            #  range(len(self.text_paths)), 2)]

        self.all_combinations = [(idx_1, idx_2)
                                 for idx_1, idx_2 in combinations(
                                            range(100), 2)]
        self.idx = {i: k for i, k in enumerate(self.text_paths)}
        # Computing similarity scores (used for sampling)
        self.sim_scores = self.pair_sim_scores()

        if self.sample == "equal":

            logging.info(f"[DATASET] Retaining pairs by {self.sample} sampling")
            self.sim_scores, self.all_combinations = self.equal_sample()

    def __len__(self):
        return len(self.all_combinations)

    def __getitem__(self, idx):
        idx_1, idx_2 = self.all_combinations[idx]
        doc_1 = self.load_data(self.text_paths[self.idx[idx_1]])
        doc_2 = self.load_data(self.text_paths[self.idx[idx_2]])
        sim = self.sim_scores[idx]
        return doc_1, doc_2, sim

    def get_fullpaths(self):

        doc_paths = {}
        logging.info("[DATASET] Getting all file paths")
        for path in self.data_paths:
            logging.info(f"[DATASET] Loading file paths from {path}")
            for doc_idx in tqdm(os.listdir(path)):
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
        # Forcing float32 to prevent double conversion
        return torch.tensor([[len(set(it_a).intersection(set(it_b))) *
                            1./len(set(it_a).union(set(it_b)))]],
                            dtype=torch.float32)

    def pair_sim_scores(self):
        sim_scores = []
        for i, (idx_1, idx_2) in enumerate(self.all_combinations):
            target_1 = self.fetch_target(self.idx[idx_1])
            target_2 = self.fetch_target(self.idx[idx_2])
            sim_scores.append(self.sim_func(target_1, target_2))
        return sim_scores

    def equal_sample(self):
        bins = np.arange(self.min_sim, self.max_sim,
                         (self.max_sim - self.min_sim) * 1./self.steps)
        bin_idxs = np.digitize(self.sim_scores, bins=bins)
        logging.info("[DATASET] Creating bins for sampling")
        bin_groups = {
                key: [item[0] for item in group]
                for key, group in tqdm(groupby(sorted(enumerate(bin_idxs),
                                       key=lambda x: x[1]), lambda x: x[1]))
                 }
        sizes = list(map(lambda x: len(bin_groups[x]), bin_groups))
        logging.info(f"[DATASET] bins sizes are {sizes}")
        least = max(min(sizes), self.least)
        selected = []
        for k, v in bin_groups.items():
            selected.extend(sample(v, min(len(v), least)))

        shuffle(selected)
        sim_scores = []
        all_combinations = []
        #  for idx in selected:
            #  sim_scores.append[self.sim_scores[idx]]
            #  all_combinations.append[self.all_combinations[idx]]
        sim_scores = [self.sim_scores[idx] for idx in selected]
        all_combinations = [self.all_combinations[idx] for idx in selected]
        return sim_scores, all_combinations
