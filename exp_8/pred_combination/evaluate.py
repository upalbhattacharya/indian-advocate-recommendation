#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-10-18 14:58:26.988095903 +0530
# Modify: 2022-10-18 14:58:26.988095903 +0530

"""Evaluation script for EnsembleConcatenation"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from data_generator import EnsembleDataset, EnsembleDataLoader
from metrics import metrics
from model.net import PredVectorCombination


def evaluate(model, loss_fn, data_loader, params, metrics, args, target_names):

    if args.restore_file is not None:

        # Loading trained model
        logging.info(f"Found checkpoint at {args.restore_file}. Loading.")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=0) + 1


    # Set model to eval mode
    model.eval()

    # Accumulate data of batches
    accumulate = utils.Accumulate()
    loss_batch = []

    criterion = loss_fn

    for data, target in data_loader():
        target = target.to(args.device)
        data = [d.float().to(args.device) for d in data]

        y_pred = model(*data)

        loss = criterion(y_pred.float(), target.float())

        outputs_batch = (y_pred.data.cpu().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)
        loss_batch.append(loss.item())

        del data
        del target
        del y_pred
        del outputs_batch
        del targets_batch

    output, targets = accumulate()

    summary_batch = {metric: metrics[metric](output, targets, target_names)
                     for metric in metrics}

    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)

    return summary_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embed_paths", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing test embeddings."))
    parser.add_argument("-en", "--embed_names", nargs="+", type=str,
                        help="Names of embeddings")
    parser.add_argument("-ed", "--embed_dims", nargs="+", type=int,
                        help="Dimensionality of different embeddings")
    parser.add_argument("-t", "--target_path", type=str,
                        default="targets/targets.json",
                        help="Path to target file.")
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

    # Loading parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(params_path), f"No params file at {params_path}"
    params = utils.Params(params_path)

    # Setting seed for reproducability
    torch.manual_seed(47)
    if "cuda" in args.device:
        torch.cuda.manual_seed(47)

    # Datasets
    test_dataset = EnsembleDataset(embed_paths=args.embed_paths,
                                   target_path=args.target_path,
                                   unique_labels=args.unique_labels)

    logging.info(f"Using {len(test_dataset.unique_labels)} targets")
    logging.info(f"Test set contains {len(test_dataset)} datapoints")

    # Dataloaders
    test_loader = EnsembleDataLoader(dataset=test_dataset,
                                     batch_size=params.batch_size)

    model = PredVectorCombination(proj_dim=params.proj_dim,
                                  names=args.embed_names,
                                  device=args.device,
                                  labels=test_dataset.unique_labels,
                                  input_dims=args.embed_dims)

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCELoss(reduction='sum')

    test_stats = evaluate(model, loss_fn, test_loader, params,
                          metrics, args, test_dataset.unique_labels)

    json_path = os.path.join(
            args.exp_dir, "metrics", f"{args.name}", "test",
            "test_f1.json")
    utils.save_dict_to_json(test_stats, json_path)

    logging.info("="*80)


if __name__ == "__main__":
    main()
