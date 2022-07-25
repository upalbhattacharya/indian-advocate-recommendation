#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-07-21 16:48:02.990301418 +0530
# Modify: 2022-07-25 14:58:57.856169430 +0530

"""Training and evaluation for BertMultiLabel"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from data_generator import SBertTrainerDataset
from model.loss_model import LossModel
from model.net import SBertEmbedding

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def train_one_epoch(loss_model, optimizer, data_loader, params, args):

    # Set model to train
    loss_model.train()
    loss_model.float()

    # For the loss of each batch
    loss_batch = []
    batch_loss = 0.0

    # Training Loop
    for i, (doc_1, doc_2, sim) in enumerate(iter(data_loader)):
        logging.info(f"Training on batch {i + 1}.")
        sim = sim.to(args.device)
        # Convert to list from tuples TODO: need to check better alternatives
        x = (list(doc_1), list(doc_2))
        # Data is moved to relevant device in net.py after tokenization
        loss_value = loss_model(x, sim)
        #  logging.info(f"The loss value is {loss_value.item()}")
        #  logging.debug(f"The dtype of loss_value is {loss_value.dtype}")
        loss_value.backward()
        batch_loss += loss_value.item()

        # Sub-batching behaviour to prevent memory overload
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(batch_loss)
            batch_loss = 0.0

        del x
        del sim
        torch.cuda.empty_cache()

    else:
        # Last batch
        if (i + 1) % params.update_grad_every != 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(batch_loss)
            batch_loss = 0.0

    summary_batch = {}
    summary_batch["batch_losses"] = {i: k for i, k in enumerate(loss_batch)}
    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)

    return summary_batch


def train(loss_model, optimizer, train_loader,
          params, exp_dir, name, args,
          restore_file=None):
    # Default start epoch
    start_epoch = 0

    # Load from checkpoint if any
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, f"{restore_file}.pth.tar")

        logging.info(f"Found checkpoint at {restore_path}.")

        start_epoch = utils.load_checkpoint(restore_path, loss_model,
                                            optimizer) + 1

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Logging for epoch {epoch}.")

        _ = train_one_epoch(loss_model, optimizer, train_loader,
                            params, args)

        state = {
            "epoch": epoch + 1,
            "state_dict": loss_model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            }

        utils.save_checkpoint(state, True,
                              os.path.join(exp_dir, "model_states", f"{name}"),
                              (epoch + 1) % params.save_every == 0)

    # For the last epoch

    utils.save_checkpoint(state, True,
                          os.path.join(exp_dir, "model_states", f"{name}"),
                          True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dirs", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and testing cases."))
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
    parser.add_argument("-sbm", "--sbert_model_name", type=str,
                        default="sentence-transformers/all-distilroberta-v1",
                        help="SBert variant to use as model.")

    args = parser.parse_args()

    # Setting logger
    timestr = time.strftime("%Y%m%d%H%M%S")
    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}_{timestr}.log"))

    # Selecting correct device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    if args.device == "cuda":
        args.device = f"cuda:{args.device_id}"

    logging.info(f"Device is {args.device}.")

    # Loading parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(params_path), f"No params file at {params_path}"
    params = utils.Params(params_path)

    # Setting seed for reproducability
    torch.manual_seed(47)
    if "cuda" in args.device:
        torch.cuda.manual_seed(47)

    # Setting data paths
    train_paths = []
    for path in args.data_dirs:
        train_paths.append(os.path.join(path, "train"))

    # Datasets
    logging.info("[DATASET] Generating pairs...")
    train_dataset = SBertTrainerDataset(data_paths=train_paths,
                                        targets_paths=args.targets_paths,
                                        unique_labels=args.unique_labels,
                                        similarity="jaccard",
                                        sample="equal",
                                        least=params.least,
                                        steps=10,
                                        min_sim=0.0,
                                        max_sim=1.0)
    logging.info(f"[DATASET] Found {len(train_dataset)} pairs.")
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                              shuffle=True)

    model = SBertEmbedding(device=args.device,
                           max_length=params.max_length,
                           sbert_model_name=args.sbert_model_name,
                           truncation_side=params.truncation_side)

    loss_model = LossModel(model=model,
                           loss_fn=nn.MSELoss(),
                           similarity_fn=torch.cosine_similarity)

    loss_model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(loss_model.parameters(), lr=params.lr)

    train(loss_model, optimizer, train_loader,
          params, args.exp_dir, args.name, args,
          restore_file=args.restore_file)

    logging.info("="*80)


if __name__ == "__main__":
    main()
