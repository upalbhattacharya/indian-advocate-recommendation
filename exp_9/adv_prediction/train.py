#!/usr/bin/env python

# Birth: 2022-10-27 11:07:17.800581752 +0530
# Modify: 2022-10-28 23:02:41.262921455 +0530

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import utils
from data_generator import SimpleDataset
from evaluate import evaluate
from metrics import metrics
from model.net import SimpleMultiLabelPrediction

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__email__ = "upal.bhattacharya@gmail.com"
__version__ = "1.0"


def train_one_epoch(model, optimizer, criterion, data_loader, params,
                    metrics, target_names, args):

    model.train()

    loss_batch = []
    accumulate = utils.Accumulate()

    # Training Loop
    num_batches = len(data_loader)
    data_loader = iter(data_loader)

    for i in tqdm.tqdm(range(num_batches), mininterval=10, desc="Training"):
        data, target = next(data_loader)
        logging.info(f"Training on batch {i+1}")
        logging.info(f"Input shape: {data.shape}")

        target = target.to(args.device)
        data = data.to(args.device)
        data = torch.squeeze(data)

        y_pred = model(data)
        logging.info(f"Output shape: {y_pred.shape}")
        loss = criterion(y_pred.float(), target.float())
        loss.backward()

        # Sub-batching
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

        outputs_batch = (y_pred.data.cpu().detach().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)
        accumulate.update(outputs_batch, targets_batch)

        # Delete statements might be unnecessary
        del data
        del target
        del y_pred
        del outputs_batch
        del targets_batch
        torch.cuda.empty_cache()

    else:
        # last batch
        if (i + 1) % params.update_grad_every != 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

        outputs, targets = accumulate()

    summary_batch = {metric: metrics[metric](outputs, targets, target_names)
                     for metric in metrics}
    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)
    return summary_batch


def train_and_evaluate(model, optimizer, criterion, train_loader,
                       val_loader, params, metrics, target_names, args):
    start_epoch = 0
    best_train_macro_f1 = 0.0
    best_val_macro_f1 = 0.0

    if args.restore_file is not None:
        restore_path = os.path.join(args.exp_dir, args.restore_file)
        logging.info(f"Found restore point at {restore_path}")
        start_epoch = utils.load_checkpoint(restore_path, model, optimizer) + 1
        # Set to None to allow new states for evaluation
        args.restore_file = None

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Logging for epoch {epoch}.")

        _ = train_one_epoch(model, optimizer, criterion, train_loader,
                            params, metrics, target_names, args)

        val_stats = evaluate(model, criterion, val_loader, params,
                             metrics, target_names, args)

        train_stats = evaluate(model, criterion, train_loader, params,
                               metrics, target_names, args)

        train_macro_f1 = train_stats['prec_rec_f1_sup']['macro_f1']
        val_macro_f1 = val_stats['prec_rec_f1_sup']['macro_f1']

        is_train_best = train_macro_f1 >= best_train_macro_f1
        is_val_best = val_macro_f1 >= best_val_macro_f1

        logging.info(
                (f"val macro F1: {val_macro_f1:0.5f}\n"
                 f"Train macro F1: {train_macro_f1:0.5f}\n"
                 f"Avg val loss: {val_stats['loss_avg']:0.5f}\n"
                 f"Avg train loss: {train_stats['loss_avg']:0.5f}\n"))

        train_json_path = os.path.join(
                    args.exp_dir, "metrics", f"{args.name}", "train",
                    f"epoch_{epoch + 1}_train_f1.json")
        utils.save_dict_to_json(train_stats, train_json_path)

        val_json_path = os.path.join(
                    args.exp_dir, "metrics", f"{args.name}", "val",
                    f"epoch_{epoch + 1}_val_f1.json")
        utils.save_dict_to_json(val_stats, val_json_path)

        if is_train_best:
            best_train_macro_f1 = train_macro_f1
            train_stats["epoch"] = epoch + 1

            train_json_path = os.path.join(
                        args.exp_dir, "metrics", f"{args.name}", "train",
                        "best_train_f1.json")
            utils.save_dict_to_json(train_stats, train_json_path)

        if is_val_best:
            best_val_macro_f1 = val_macro_f1
            val_stats["epoch"] = epoch + 1

            val_json_path = os.path.join(
                        args.exp_dir, "metrics", f"{args.name}", "val",
                        "best_val_f1.json")
            utils.save_dict_to_json(val_stats, val_json_path)

            logging.info(
                    (f"New best macro F1: {best_val_macro_f1:0.5f} "
                     f"Train macro F1: {train_macro_f1:0.5f} "
                     f"Avg val loss: {val_stats['loss_avg']} "
                     f"Avg train loss: {train_stats['loss_avg']}."))

            state = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict()
                    }

            utils.save_checkpoint(state, is_val_best,
                                  os.path.join(args.exp_dir, "model_states",
                                               f"{args.name}"),
                                  (epoch + 1) % params.save_every == 0)

    # Save last epoch stats
    utils.save_checkpoint(state, is_val_best,
                          os.path.join(args.exp_dir, "model_states",
                                       f"{args.name}"), True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dirs", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and validation cases."))
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

    # Setting data paths
    train_paths = []
    val_paths = []
    for path in args.data_dirs:
        train_paths.append(os.path.join(path, "train"))
        val_paths.append(os.path.join(path, "validation"))

    # Datasets
    train_dataset = SimpleDataset(
                            data_paths=train_paths,
                            targets_paths=args.targets_paths,
                            unique_labels=args.unique_labels)

    val_dataset = SimpleDataset(
                            data_paths=val_paths,
                            targets_paths=args.targets_paths,
                            unique_labels=args.unique_labels)

    logging.info(f"Training with {len(train_dataset.unique_labels)} targets")
    logging.info(f"Training on {len(train_dataset)} datapoints")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=params.batch_size,
                            shuffle=True)

    model = SimpleMultiLabelPrediction(
                             labels=train_dataset.unique_labels,
                             device=args.device,
                             input_dim=params.input_dim
                             )

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCELoss(reduction='sum')

    train_and_evaluate(model, optimizer, loss_fn, train_loader,
                       val_loader, params, metrics,
                       train_dataset.unique_labels, args)

    logging.info("="*80)


if __name__ == "__main__":
    main()
