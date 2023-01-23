#!usr/bin/env python

"""Custom dataset and batch generator for ensemble methods"""

import json
import os
from collections import defaultdict
from itertools import chain
from random import sample

import numpy as np
import torch
from torch.utils.data import Dataset


class EnsembleDataset(Dataset):
    def __init__(self, embed_paths, target_path, unique_labels=None,
                 mode="train"):
        self.embed_paths = embed_paths
        self.target_path = target_path
        self.mode = mode
        self.targets_dict = self.get_targets()
        if unique_labels is None:
            self.unique_labels = self.get_unique_labels()
        else:
            with open(unique_labels, 'r') as f:
                self.unique_labels = f.readlines()
            self.unique_labels = list(filter(None, map(lambda x: x.strip("\n"),
                                                       self.unique_labels)))

        self.idx = self.get_idx()
        self.full_paths = self.get_full_paths()

    def __len__(self):
        return len(os.listdir(self.embed_paths[0]))

    def get_targets(self):
        with open(self.target_path, 'r') as f:
            targets = json.load(f)
        return targets

    def get_unique_labels(self):
        unique_labels = list(set(chain.from_iterable(
                                            self.targets_dict.values())))
        unique_labels = sorted(unique_labels)

        return unique_labels

    def get_idx(self):
        idx = {i: os.path.splitext(path)[0]
               for i, path in enumerate(os.listdir(self.embed_paths[0]))}
        return idx

    def get_full_paths(self):
        full_paths = {}
        for idx in self.idx.values():
            paths = {}
            for i, embed_path in enumerate(self.embed_paths):
                if os.path.exists(os.path.join(embed_path, f"{idx}.pt")):
                    paths[i] = os.path.join(embed_path, f"{idx}.pt")
                elif os.path.exists(os.path.join(embed_path, f"{idx}.npy")):
                    paths[i] = os.path.join(embed_path, f"{idx}.npy")
                else:
                    raise FileNotFoundError(
                            f"Embedding for {idx} not found in {embed_path}")
            full_paths[idx] = paths
        return full_paths

    def fetch_target(self, doc):
        target = torch.tensor([int(label in self.targets_dict[doc])
                               for label in self.unique_labels])

        return target

    def load_data(self, path):
        ext = os.path.splitext(path)[-1]
        if "npy" in ext:
            data = np.load(path, allow_pickle=True)
            # Handle bad numpy saved sparse data
            if len(data.shape) == 0:
                # TODO: upalbhattacharya:
                # Efficiently use sparse tensors. Can cause memory overhead
                data = data.tolist()[0]
                data = data.todense()
            data = torch.from_numpy(data)
            if len(data.shape) == 1:
                data = torch.unsqueeze(data, dim=0)
        elif "pt" in ext:
            data = torch.load(path)
            if len(data.shape) == 1:
                data = torch.unsqueeze(data, dim=0)
        else:
            raise FileNotFoundError(f"{path} does not exist")
        return data

    def __getitem__(self, idx):
        data = []
        for path in self.full_paths[self.idx[idx]].values():
            data.append(self.load_data(path))

        target = self.fetch_target(self.idx[idx])

        if self.mode == "train":
            return data, target
        else:
            return data, self.idx[idx]


class EnsembleDataLoader():
    def __init__(self, dataset, batch_size=16, mode="train"):
        super(EnsembleDataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        self.num_items = len(dataset)
        self.full_batches = self.num_items // self.batch_size
        self.left_num = self.num_items % self.batch_size

    def __call__(self):
        idx = list(range(0, self.num_items))
        for batch in range(self.full_batches):
            items_data = []
            items_targets = []
            selected_idxs = sample(idx, self.batch_size)
            for i in selected_idxs:
                data, targets = self.dataset[i]
                items_data.append(data)
                items_targets.append(targets)
            idx = list(set(idx) - set(selected_idxs))
            items_data, items_targets = self.collate(items_data, items_targets)
            yield items_data, items_targets
        else:
            if self.left_num != 0:
                items_data = []
                items_targets = []
                for i in idx:
                    data, targets = self.dataset[i]
                    items_data.append(data)
                    items_targets.append(targets)
                items_data, items_targets = self.collate(items_data, items_targets)
                yield items_data, items_targets

    def collate(self, data, targets):
        embeds = defaultdict(list)
        stacked_embeds = []
        for embed in data:
            for i in range(len(embed)):
                emb = embed[i]
                embeds[i].append(emb)

        for embed in embeds.values():
            concat = torch.concat(embed, dim=0)
            stacked_embeds.append(concat)

        if self.mode == "train":
            stacked_targets = (np.stack(targets, axis=0) if type(targets[0]) ==
                               np.ndarray else torch.stack(targets, dim=0))
        else:
            stacked_targets = targets

        return stacked_embeds, stacked_targets
