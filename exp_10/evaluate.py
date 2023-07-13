#!/usr/bin/env python

# Birth: 2022-10-28 10:03:56.902540957 +0530
# Modify: 2022-11-03 18:51:36.915832785 +0530

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import utils
from data_generator import MultiTaskDataset
from metrics import metrics
from model.multi_task_loss import MultiTaskLoss
from model.net import SimpleMultiTaskMultiLabelPrediction
from torch.utils.data import DataLoader


def evaluate(
    model,
    criterion,
    multiTaskLoss,
    data_loader,
    params,
    metrics,
    target_names,
    args,
):
    if args.restore_file is not None:
        logging.info(f"Loading checkpoint at{args.restore_file}")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=None) + 1

    model.eval()
    m = nn.Sigmoid()

    loss_batch = []
    loss_batch_adv = []
    loss_batch_area = []
    accumulate_adv = utils.Accumulate()
    accumulate_area = utils.Accumulate()

    preds_adv = {}
    preds_area = {}
    activations_adv = {}
    activations_area = {}

    num_batches = len(data_loader)
    data_loader = iter(data_loader)

    for i in tqdm.tqdm(range(num_batches), mininterval=1, desc="Evaluating"):
        data, target_adv, target_area, idx = next(data_loader)
        data = list(data)

        target_adv = target_adv.to(args.device)
        target_area = target_area.to(args.device)

        y_pred_adv, y_pred_area = model(data)

        y_pred_adv_acts = m(y_pred_adv)
        y_pred_area_acts = m(y_pred_area)

        # Setting area prediction to zero for cases with no determined areas
        for t in range(len(target_area)):
            if not torch.any(target_area[t].bool()):
                y_pred_area_acts[t] = y_pred_area_acts[t] * target_area[t]

        loss_adv = criterion["adv"](
            y_pred_adv_acts.float(), target_adv.float()
        )
        loss_area = criterion["area"](
            y_pred_area_acts.float(), target_area.float()
        )
        losses = torch.stack((loss_adv, loss_area))
        multi_task_loss = multiTaskLoss(losses)

        loss_batch.append(multi_task_loss.item())
        loss_batch_adv.append(loss_adv.item())
        loss_batch_area.append(loss_area.item())

        outputs_batch_adv = (
            y_pred_adv_acts.detach().data.cpu().numpy() > params.threshold
        ).astype(np.int32)

        targets_batch_adv = (target_adv.detach().data.cpu().numpy()).astype(
            np.int32
        )

        accumulate_adv.update(outputs_batch_adv, targets_batch_adv)

        outputs_batch_area = (
            y_pred_area_acts.detach().data.cpu().numpy() > params.threshold
        ).astype(np.int32)

        targets_batch_area = (target_area.detach().data.cpu().numpy()).astype(
            np.int32
        )

        accumulate_area.update(outputs_batch_area, targets_batch_area)

        pred_idx_adv = [
            [i for i, val in enumerate(pred) if val != 0.0]
            for pred in outputs_batch_adv
        ]
        pred_names_adv = [
            [target_names["adv"][j] for j in pred] for pred in pred_idx_adv
        ]
        for i, pred in zip(idx, pred_names_adv):
            preds_adv[i] = pred

        for i, acts in zip(idx, y_pred_adv_acts.detach().data.cpu().numpy()):
            activations_adv[i] = {
                target: acts[j] for j, target in enumerate(target_names["adv"])
            }

        pred_idx_area = [
            [i for i, val in enumerate(pred) if val != 0.0]
            for pred in outputs_batch_area
        ]
        pred_names_area = [
            [target_names["area"][j] for j in pred] for pred in pred_idx_area
        ]
        for i, pred in zip(idx, pred_names_area):
            preds_area[i] = pred

        for i, acts in zip(idx, y_pred_area_acts.detach().data.cpu().numpy()):
            activations_area[i] = {
                target: acts[j]
                for j, target in enumerate(target_names["area"])
            }
        # Delete statements might be unnecessary
        del data
        del target_adv
        del target_area
        del y_pred_adv
        del y_pred_area
        del y_pred_adv_acts
        del y_pred_area_acts
        del outputs_batch_adv
        del targets_batch_adv
        del outputs_batch_area
        del targets_batch_area
        torch.cuda.empty_cache()

    outputs_adv, targets_adv = accumulate_adv()
    outputs_area, targets_area = accumulate_area()

    summary_batch = {
        "adv": {
            metric: metrics[metric](
                outputs_adv, targets_adv, target_names["adv"]
            )
            for metric in metrics
        },
        "area": {
            metric: metrics[metric](
                outputs_area, targets_area, target_names["area"]
            )
            for metric in metrics
        },
    }
    summary_batch["adv"]["preds"] = preds_adv
    summary_batch["adv"]["activations"] = pd.DataFrame.from_dict(
        activations_adv, orient="index"
    )
    summary_batch["area"]["preds"] = preds_area
    summary_batch["area"]["activations"] = pd.DataFrame.from_dict(
        activations_area, orient="index"
    )
    summary_batch["loss_avg"] = sum(loss_batch) * 1.0 / len(loss_batch)
    summary_batch["loss_adv"] = sum(loss_batch_adv) * 1.0 / len(loss_batch_adv)
    summary_batch["loss_area"] = (
        sum(loss_batch_area) * 1.0 / len(loss_batch_area)
    )
    return summary_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_paths",
        nargs="+",
        type=str,
        default=["data/"],
        help="Directory containing evaluation cases",
    )
    parser.add_argument(
        "-tc",
        "--targets_paths_areas",
        nargs="+",
        type=str,
        default=["targets/targets.json"],
        help="Path to target files for areas.",
    )
    parser.add_argument(
        "-ta",
        "--targets_paths_advs",
        nargs="+",
        type=str,
        default=["targets/targets.json"],
        help="Path to target files for advocates.",
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
        "-en",
        "--embed_model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of pre-trained model to load",
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
        "-ulc",
        "--unique_labels_areas",
        type=str,
        default=None,
        help="Labels to use as targets for areas.",
    )
    parser.add_argument(
        "-ula",
        "--unique_labels_advs",
        type=str,
        default=None,
        help="Labels to use as targets for advocates.",
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

    targets_paths = {
        "adv": args.targets_paths_advs,
        "area": args.target_paths_areas,
    }

    unique_labels = {
        "adv": args.unique_labels_advs,
        "area": args.target_paths_areas,
    }

    # Datasets
    test_dataset = MultiTaskDataset(
        data_paths=args.data_paths,
        targets_paths=targets_paths,
        unique_labels=unique_labels,
    )

    logging.info(
        (
            "testing with "
            f"{len(test_dataset.unique_labels['adv'])} adv targets"
        )
    )
    logging.info(
        (
            "testing with "
            f"{len(test_dataset.unique_labels['area'])} area targets"
        )
    )
    logging.info(f"testing on {len(test_dataset)} datapoints")

    # Dataloaders
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = SimpleMultiTaskMultiLabelPrediction(
        labels=test_dataset.unique_labels,
        max_length=params.max_length,
        truncation_side=params.truncation_side,
        model_name=args.embed_model_name,
        mode="train",
        device=args.device,
    )

    model.to(args.device)

    model.to(args.device)

    is_regression = torch.Tensor([False, False])
    multiTaskLoss = MultiTaskLoss(is_regression=is_regression, reduction="sum")
    multiTaskLoss.to(args.device)

    loss_area = nn.BCELoss(reduction="sum")
    loss_adv = nn.BCELoss(reduction="sum")
    loss_fn = {"area": loss_area, "adv": loss_adv}

    test_stats = evaluate(
        model,
        loss_fn,
        multiTaskLoss,
        test_loader,
        params,
        metrics,
        test_dataset.unique_labels,
        args,
    )

    test_adv_acts = test_stats["adv"]["activations"]
    del test_stats["adv"]["activations"]
    test_area_acts = test_stats["area"]["activations"]
    del test_stats["area"]["activations"]

    json_path = os.path.join(
        args.exp_dir, "metrics", f"{args.name}", "test", "test_f1.json"
    )
    utils.save_dict_to_json(test_stats, json_path)

    acts_adv_path = os.path.join(
        args.exp_dir,
        "activations",
        f"{args.name}",
        "test",
        "test_adv_activations.pkl",
    )
    acts_area_path = os.path.join(
        args.exp_dir,
        "activations",
        f"{args.name}",
        "test",
        "test_area_activations.pkl",
    )
    utils.save_df_to_pkl(test_adv_acts, acts_adv_path)

    utils.save_df_to_pkl(test_area_acts, acts_area_path)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
