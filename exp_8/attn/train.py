#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-09-12 19:41:15.490098971 +0530
# Modify: 2022-09-12 20:25:04.149972027 +0530

"""Training and evaluation for EnsembleSelfAttn"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from data_generator import EnsembleDataset, EnsembleDataLoader
from evaluate import evaluate
from metrics import metrics
from model.net import EnsembleSelfAttn

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def train_one_epoch(model, optimizer, loss_fn, data_loader, params,
                    metrics, target_names, args):

    # Set model to train
    model.train()

    criterion = loss_fn

    # For the loss of each batch
    loss_batch = []
    accumulate = utils.Accumulate()

    # Training Loop
    for i, (data, target) in enumerate(data_loader()):
        logging.info(f"Training on batch {i + 1}.")
        target = target.to(args.device)
        data = [d.float().to(args.device) for d in data]
        y_pred = model(*data)
        loss = criterion(y_pred.float(), target.float())
        loss.backward()

        # Sub-batching behaviour to prevent memory overload
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

        outputs_batch = (y_pred.data.cpu().detach().numpy()
                         > params.threshold).astype(np.int32)

        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)

        del data
        del target
        del outputs_batch
        del targets_batch
        del y_pred
        torch.cuda.empty_cache()

    else:
        # Last batch
        if (i + 1) % params.update_grad_every != 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

    outputs, targets = accumulate()

    summary_batch = {metric: metrics[metric](outputs, targets, target_names)
                     for metric in metrics}
    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)

    return summary_batch


def train_and_evaluate(model, optimizer, loss_fn, train_loader,
                       val_loader, params, metrics, exp_dir, name, args,
                       target_names, restore_file=None):
    # Default start epoch
    start_epoch = 0
    # Best train and val macro f1 variables
    best_train_macro_f1 = 0.0
    best_val_macro_f1 = 0.0

    # Load from checkpoint if any
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, f"{restore_file}.pth.tar")

        logging.info(f"Found checkpoint at {restore_path}.")

        start_epoch = utils.load_checkpoint(restore_path, model, optimizer) + 1

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Logging for epoch {epoch}.")

        _ = train_one_epoch(model, optimizer, loss_fn, train_loader,
                            params, metrics, target_names, args)

        val_stats = evaluate(model, loss_fn, val_loader,
                              params, metrics, args, target_names)

        train_stats = evaluate(model, loss_fn, train_loader,
                               params, metrics, args, target_names)

        # Getting f1 val_stats

        train_macro_f1 = train_stats['f1']['macro_f1']
        is_train_best = train_macro_f1 >= best_train_macro_f1

        val_macro_f1 = val_stats['f1']['macro_f1']
        is_val_best = val_macro_f1 >= best_val_macro_f1

        logging.info(
                (f"Test macro F1: {val_macro_f1:0.5f}\n"
                 f"Train macro F1: {train_macro_f1:0.5f}\n"
                 f"Avg val loss: {val_stats['loss_avg']:0.5f}\n"
                 f"Avg train loss: {train_stats['loss_avg']:0.5f}\n"))

        # Save val_stats
        train_json_path = os.path.join(
                exp_dir, "metrics", f"{name}", "train",
                f"epoch_{epoch + 1}_train_f1.json")
        utils.save_dict_to_json(train_stats, train_json_path)

        val_json_path = os.path.join(
                exp_dir, "metrics", f"{name}", "val",
                f"epoch_{epoch + 1}_val_f1.json")
        utils.save_dict_to_json(val_stats, val_json_path)

        # Saving best stats
        if is_train_best:
            best_train_macro_f1 = train_macro_f1
            train_stats["epoch"] = epoch + 1

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "train",
                    "best_train_f1.json")
            utils.save_dict_to_json(train_stats, best_json_path)

        if is_val_best:
            best_val_macro_f1 = val_macro_f1
            val_stats["epoch"] = epoch + 1

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "val",
                    "best_val_f1.json")
            utils.save_dict_to_json(val_stats, best_json_path)

            logging.info(
                    (f"New best macro F1: {best_val_macro_f1:0.5f} "
                     f"Train macro F1: {train_macro_f1:0.5f} "
                     f"Avg val loss: {val_stats['loss_avg']} "
                     f"Avg train loss: {train_stats['loss_avg']}."))

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            }

        utils.save_checkpoint(state, is_val_best,
                              os.path.join(exp_dir, "model_states", f"{name}"),
                              (epoch + 1) % params.save_every == 0)

    # For the last epoch

    utils.save_checkpoint(state, is_val_best,
                          os.path.join(exp_dir, "model_states", f"{name}"),
                          True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embed_paths", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and valing embeddings."))
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

    # Setting data paths
    train_paths = []
    val_paths = []
    for path in args.embed_paths:
        train_paths.append(os.path.join(path, "train"))
        val_paths.append(os.path.join(path, "val"))

    # Datasets
    train_dataset = EnsembleDataset(embed_paths=train_paths,
                                    target_path=args.target_path,
                                    unique_labels=args.unique_labels)

    val_dataset = EnsembleDataset(embed_paths=val_paths,
                                   target_path=args.target_path,
                                   unique_labels=args.unique_labels)

    logging.info(f"Using {len(train_dataset.unique_labels)} targets")

    # Dataloaders
    train_loader = EnsembleDataLoader(dataset=train_dataset,
                                      batch_size=params.batch_size)

    val_loader = EnsembleDataLoader(dataset=val_dataset,
                                     batch_size=params.batch_size)

    model = EnsembleSelfAttn(proj_dim=params.proj_dim,
                             names=args.embed_names,
                             device=args.device,
                             labels=train_dataset.unique_labels,
                             input_dims=args.embed_dims)

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCELoss(reduction='sum')

    train_and_evaluate(model, optimizer, loss_fn, train_loader,
                       val_loader, params, metrics, args.exp_dir,
                       args.name, args, train_dataset.unique_labels,
                       restore_file=args.restore_file)

    logging.info("="*80)


if __name__ == "__main__":
    main()
