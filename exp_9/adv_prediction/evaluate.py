#!/usr/bin/env python

# Birth: 2022-10-27 11:07:17.800581752 +0530
# Modify: 2022-10-27 21:28:29.039833024 +0530

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

import utils
from metrics import metrics
from data_generator import SimpleDataset
from model.net import SimpleMultiLabelPrediction


def evaluate(model, criterion, data_loader, params, metrics, target_names,
             args):

    if args.restore_file is not None:
        logging.info(f"Loading checkpoint at{args.restore_file}")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=None) + 1

    accumulate = utils.Accumulate()
    loss_batch = []

    model.eval()

    num_batches = len(data_loader)
    data_loader = iter(data_loader)

    for i in tqdm.trange(num_batches, mininters=1, desc="Evaluating"):
        data, target = next(data_loader)
        target = target.to(args.device)
        data = data.to(args.device)
        data = torch.squeeze(data)

        y_pred = model(data)

        loss = criterion(y_pred.float(), target.float())
        outputs_batch = (y_pred.data.cpu().detach().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)
        loss_batch.append(loss.item())

        del data
        del target
        del outputs_batch
        del targets_batch
        torch.cuda.empty_cache()

    outputs, targets = accumulate()

    summary_batch = {metric: metrics[metric](outputs, targets, target_names)
                     for metric in metrics}

    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)
    return summary_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_paths", nargs="+", type=str,
                        default=["data/"],
                        help="Directory containing test cases.")
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

    # Datasets

    test_dataset = SimpleDataset(
                            data_paths=args.data_paths,
                            targets_paths=args.targets_paths,
                            unique_labels=args.unique_labels)

    logging.info(f"Testing with {len(test_dataset.unique_labels)} targets")
    logging.info(f"Testing on {len(test_dataset)} datapoints")

    # Dataloaders
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=True)

    model = SimpleMultiLabelPrediction(
                             labels=test_dataset.unique_labels,
                             device=args.device,
                             input_dim=params.input_dim
                             )

    model.to(args.device)

    # Defining optimizer and loss function
    loss_fn = nn.BCELoss(reduction='sum')

    test_stats = evaluate(model, loss_fn, test_loader, params,
                          metrics, test_dataset.unique_labels, args)

    json_path = os.path.join(
            args.exp_dir, "metrics", f"{args.name}", "test",
            "test_f1.json")
    utils.save_dict_to_json(test_stats, json_path)

    logging.info("="*80)


if __name__ == "__main__":
    main()
