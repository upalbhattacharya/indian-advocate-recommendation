#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-10-25 16:51:01.466432365 +0530
# Modify: 2022-10-25 17:11:30.940935280 +0530

"""Training and evaluation for EnsembleSelfAttn"""

import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_lr_finder import TrainDataLoaderIter, LRFinder

import utils
from data_generator import LongformerMultiLabelDataset
from evaluate import evaluate
from metrics import metrics
from model.net import LongformerMultiLabel

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"

class CustomLRFinder(LRFinder):
    def __init(self, *args, **kwargs):
        super(CustomLRFinder, self).__init__(*args, **kwargs)

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        total_loss = None  # for late initialization
                                    
        self.optimizer.zero_grad()
        for i in range(accumulation_steps):
            inputs, labels = next(train_iter)

            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs.float(), labels.float())

            # Loss should be averaged in each step
            loss /= accumulation_steps
            # Backward pass
            loss.backward()
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

            self.optimizer.step()
            return total_loss.item()


class CustomDataLoaderIter(TrainDataLoaderIter):
    def __init__(self, data_loader, auto_reset=True):
        super(CustomDataLoaderIter, self).__init__(data_loader)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            inputs, labels = batch
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            inputs, labels = batch
        return inputs, labels


def find_lr(model, optimizer, loss_fn, data_loader, args, params):
    fig, ax = plt.subplots()
    lr_finder = CustomLRFinder(model, optimizer, loss_fn, device=args.device)
    lr_finder.range_test(data_loader, end_lr=10, num_iter=100,
                         accumulation_steps=400, smooth_f=0.05)
    ax = lr_finder.plot(ax=ax) # to inspect the loss-learning rate graph
    fig.savefig(os.path.join(args.save_path, "lr_plot.png"))
    lr_finder.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_paths", nargs="+", type=str,
                        default=["data/"],
                        help="Directory containing cases.")
    parser.add_argument("-t", "--targets_paths", nargs="+", type=str,
                        default=["targets/targets.json"],
                        help="Path to target files.")
    parser.add_argument("-x", "--exp_dir", default="experiments/",
                        help=("Directory to load parameters "
                              " from and save metrics and model states"))
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of model")
    parser.add_argument("-p", "--params", default="params.json",
                        help="Name of params file to load from exp+_dir")
    parser.add_argument("-de", "--device", type=str, default="cuda",
                        help="Device to train on.")
    parser.add_argument("-id", "--device_id", type=int, default=0,
                        help="Device ID to run on if using GPU.")
    parser.add_argument("-r", "--restore_file", default=None,
                        help="Restore point to use.")
    parser.add_argument("-ul", "--unique_labels", type=str, default=None,
                        help="Labels to use as targets.")
    parser.add_argument("-lm", "--longformer_model_name", type=str,
                        default="longformer-base-uncased",
                        help="Longformer variant to use as model.")
    parser.add_argument("-s", "--save_path", type=str,
                        help="Path to save generated lr plot")

    args = parser.parse_args()

    # Setting logger
    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}"))

    # Selecting correct device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    if args.device == "cuda":
        args.device = f"cuda:{args.device_id}"

    logging.info(f"Device is {args.device}.")

    logging.info("Final arguments are:")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # Loading parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(params_path), f"No params file at {params_path}"
    params = utils.Params(params_path)

    # Setting seed for reproducability
    torch.manual_seed(47)
    if "cuda" in args.device:
        torch.cuda.manual_seed(47)

    train_paths = []
    for path in args.data_paths:
        train_paths.append(os.path.join(path, "train"))

    # Datasets
    train_dataset = LongformerMultiLabelDataset(
                                    data_paths=train_paths,
                                    targets_paths=args.targets_paths,
                                    unique_labels=args.unique_labels)

    logging.info(f"[DATASET] Using {len(train_dataset.unique_labels)} targets")

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                              shuffle=True)

    model = LongformerMultiLabel(labels=train_dataset.unique_labels,
                           device=args.device,
                           hidden_size=params.hidden_dim,
                           max_length=params.max_length,
                           longformer_model_name=args.longformer_model_name,
                           truncation_side=params.truncation_side)

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCELoss(reduction='sum')

    find_lr(model, optimizer, loss_fn, train_loader, args, params)

    logging.info("="*80)


if __name__ == "__main__":
    main()
