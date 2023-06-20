#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-10-25 11:22:37.744100343 +0530
# Modify: 2022-10-25 12:07:18.847364600 +0530

"""Getting predictions from trained BertMultiLabel model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
from data_generator import BertMultiLabelDataset
from metrics import metrics
from model.net import BertMultiLabel
from torch.utils.data import DataLoader


def get_preds(model, data_loader, params, metrics, args, target_names):
    if args.restore_file is not None:
        # Loading trained model
        logging.info(f"Found checkpoint at {args.restore_file}. Loading.")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=0) + 1
        args.restore_file = None

    # Set model to eval mode
    model.eval()

    preds = {}

    for idx, data in iter(data_loader):
        data = list(data)
        y_pred = model(data)

        outputs_batch = (y_pred.data.cpu().numpy() > params.threshold).astype(
            np.int32
        )
        outputs_batch = outputs_batch[0]
        pred_idx = [i for i, val in enumerate(outputs_batch) if val != 0.0]
        pred = [target_names[j] for j in pred_idx]
        idx = idx[0]
        preds[idx] = pred

        del data
        del y_pred
        del outputs_batch

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_paths",
        nargs="+",
        type=str,
        default=["data/"],
        help="Directory containing test cases.",
    )
    parser.add_argument(
        "-t",
        "--targets_paths",
        nargs="+",
        type=str,
        default=["targets/targets.json"],
        help="Path to target files.",
    )
    parser.add_argument(
        "-x",
        "--exp_dir",
        default="experiments/",
        help=(
            "Directory to load parameters "
            " from and save metrics and model states"
        ),
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="Name of model"
    )
    parser.add_argument(
        "-p",
        "--params",
        default="params.json",
        help="Name of params file to load from exp+_dir",
    )
    parser.add_argument(
        "-de", "--device", type=str, default="cuda", help="Device to train on."
    )
    parser.add_argument(
        "-id",
        "--device_id",
        type=int,
        default=0,
        help="Device ID to run on if using GPU.",
    )
    parser.add_argument(
        "-r", "--restore_file", default=None, help="Restore point to use."
    )
    parser.add_argument(
        "-ul",
        "--unique_labels",
        type=str,
        default=None,
        help="Labels to use as targets.",
    )
    parser.add_argument(
        "-lm",
        "--model_name",
        type=str,
        default="allenai/longformer-base-4096",
        help="Longformer variant to use as model.",
    )

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

    # Datasets
    test_dataset = BertMultiLabelDataset(
        data_paths=args.data_paths,
        targets_paths=args.targets_paths,
        unique_labels=args.unique_labels,
        mode="eval",
    )

    logging.info(f"[DATASET] Using {len(test_dataset.unique_labels)} targets")
    logging.info(f"[DATASET] Test set contains {len(test_dataset)} datapoints")

    # Dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=True
    )

    model = BertMultiLabel(
        labels=test_dataset.unique_labels,
        device=args.device,
        hidden_size=params.hidden_dim,
        max_length=params.max_length,
        model_name=args.model_name,
        truncation_side=params.truncation_side,
    )

    model.to(args.device)

    test_stats = get_preds(
        model,
        test_loader,
        params,
        metrics,
        args,
        test_dataset.unique_labels,
    )

    json_path = os.path.join(
        args.exp_dir, "metrics", f"{args.name}", "test", "preds.json"
    )
    utils.save_dict_to_json(test_stats, json_path)

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
