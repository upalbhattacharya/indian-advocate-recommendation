#!/usr/bin/env python

# Birth: 2022-10-28 10:03:56.902540957 +0530
# Modify: 2022-11-03 21:53:49.028981352 +0530

import argparse
import logging
import os
from collections import Counter
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import utils
from data_generator import MultiTaskDataset
from evaluate import evaluate
from metrics import metrics
from model.multi_task_loss import MultiTaskLoss
from model.net import SimpleMultiTaskMultiLabelPrediction
from torch.utils.data import DataLoader

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__email__ = "upal.bhattacharya@gmail.com"
__version__ = "1.0"


def train_one_epoch(
    model,
    optimizer,
    criterion,
    multiTaskLoss,
    data_loader,
    params,
    metrics,
    target_names,
    args,
):
    model.train()
    m = nn.Sigmoid()

    loss_batch = []
    loss_batch_adv = []
    loss_batch_area = []
    accumulate_adv = utils.Accumulate()
    accumulate_area = utils.Accumulate()

    # Training Loop
    num_batches = len(data_loader)
    data_loader = iter(data_loader)

    for i in tqdm.tqdm(range(num_batches), mininterval=10, desc="Training"):
        data, target_adv, target_area, _ = next(data_loader)
        data = list(data)

        target_adv = target_adv.to(args.device)
        target_area = target_area.to(args.device)

        y_pred_adv, y_pred_area = model(data)

        y_pred_area_mod = []
        target_area_mod = []

        # Setting area prediction to zero for cases with no determined areas
        for t in range(len(target_area)):
            if not torch.any(target_area[t].bool()):
                continue
                # y_pred_area[t] = y_pred_area[t] * target_area[t]
            y_pred_area_mod.append(y_pred_area[t])
            target_area_mod.append(target_area[t])

        if y_pred_area_mod != []:
            y_pred_area_mod = torch.stack(y_pred_area_mod, dim=0)
            target_area_mod = torch.stack(target_area_mod, dim=0)
        else:
            y_pred_area_mod = torch.Tensor([])
            y_pred_area_mod = y_pred_area_mod.to(args.device)
            target_area_mod = torch.Tensor([])
            target_area_mod = target_area_mod.to(args.device)

        loss_adv = criterion["adv"](y_pred_adv.float(), target_adv.float())
        if torch.numel(y_pred_area_mod) != 0:
            loss_area = criterion["area"](
                y_pred_area_mod.float(), target_area_mod.float()
            )
        else:
            loss_area = criterion["area"](
                torch.zeros(len(target_names["area"])).to(args.device),
                torch.zeros(len(target_names["area"])).to(args.device),
            )
        losses = torch.stack((loss_adv, loss_area))
        multi_task_loss = multiTaskLoss(losses)
        multi_task_loss.backward()

        # Sub-batching
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(multi_task_loss.item())
            loss_batch_adv.append(loss_adv.item())
            loss_batch_area.append(loss_area.item())

        outputs_batch_adv = (
            m(y_pred_adv).detach().data.cpu().numpy() > params.threshold
        ).astype(np.int32)

        targets_batch_adv = (target_adv.data.cpu().detach().numpy()).astype(
            np.int32
        )

        accumulate_adv.update(outputs_batch_adv, targets_batch_adv)

        outputs_batch_area = (
            m(y_pred_area).detach().data.cpu().numpy() > params.threshold
        ).astype(np.int32)

        targets_batch_area = (target_area.data.cpu().detach().numpy()).astype(
            np.int32
        )

        accumulate_area.update(outputs_batch_area, targets_batch_area)

        # Delete statements might be unnecessary
        del data
        del target_adv
        del target_area
        del target_area_mod
        del y_pred_adv
        del y_pred_area
        del y_pred_area_mod
        del outputs_batch_adv
        del targets_batch_adv
        del outputs_batch_area
        del targets_batch_area
        torch.cuda.empty_cache()

    else:
        # last batch
        if (i + 1) % params.update_grad_every != 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(multi_task_loss.item())
            loss_batch_adv.append(loss_adv.item())
            loss_batch_area.append(loss_area.item())

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
    summary_batch["loss_avg"] = sum(loss_batch) * 1.0 / len(loss_batch)
    summary_batch["loss_adv"] = sum(loss_batch_adv) * 1.0 / len(loss_batch_adv)
    summary_batch["loss_area"] = (
        sum(loss_batch_area) * 1.0 / len(loss_batch_area)
    )
    return summary_batch


def train_and_evaluate(
    model,
    optimizer,
    criterion,
    multiTaskLoss,
    train_loader,
    train_check_loader,
    val_loader,
    params,
    metrics,
    target_names,
    args,
):
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

        _ = train_one_epoch(
            model,
            optimizer,
            criterion,
            multiTaskLoss,
            train_loader,
            params,
            metrics,
            target_names,
            args,
        )

        val_stats = evaluate(
            model,
            criterion,
            multiTaskLoss,
            val_loader,
            params,
            metrics,
            target_names,
            args,
        )

        val_adv_acts = val_stats["adv"]["activations"]
        del val_stats["adv"]["activations"]

        val_area_acts = val_stats["area"]["activations"]
        del val_stats["area"]["activations"]

        train_stats = evaluate(
            model,
            criterion,
            multiTaskLoss,
            train_check_loader,
            params,
            metrics,
            target_names,
            args,
        )

        train_adv_acts = train_stats["adv"]["activations"]
        del train_stats["adv"]["activations"]

        train_area_acts = train_stats["area"]["activations"]
        del train_stats["area"]["activations"]

        train_macro_f1_adv = train_stats["adv"]["prec_rec_f1_sup"]["macro_f1"]
        val_macro_f1_adv = val_stats["adv"]["prec_rec_f1_sup"]["macro_f1"]

        train_macro_f1_area = train_stats["area"]["prec_rec_f1_sup"][
            "macro_f1"
        ]
        val_macro_f1_area = val_stats["area"]["prec_rec_f1_sup"]["macro_f1"]

        is_train_best = train_macro_f1_adv >= best_train_macro_f1
        is_val_best = val_macro_f1_adv >= best_val_macro_f1

        logging.info(
            (
                f"Performance for epoch {epoch}:\n"
                "=====Adv====\n"
                "============\n"
                f"val macro F1: {val_macro_f1_adv:0.5f}\n"
                f"train macro F1: {train_macro_f1_adv:0.5f}\n"
                f"Avg val loss: {val_stats['loss_adv']:0.5f}\n"
                f"Avg train loss: {train_stats['loss_adv']:0.5f}\n"
                "====Area====\n"
                "============\n"
                f"val macro F1: {val_macro_f1_area:0.5f}\n"
                f"train macro F1: {train_macro_f1_area:0.5f}\n"
                f"Avg val loss: {val_stats['loss_area']:0.5f}\n"
                f"Avg train loss: {train_stats['loss_area']:0.5f}\n"
                "===Overall==\n"
                "============\n"
                f"Avg val loss: {val_stats['loss_avg']:0.5f}\n"
                f"Avg train loss: {train_stats['loss_avg']:0.5f}\n"
            )
        )

        train_json_path = os.path.join(
            args.exp_dir,
            "metrics",
            f"{args.name}",
            "train",
            f"epoch_{epoch + 1}_train_f1.json",
        )
        utils.save_dict_to_json(train_stats, train_json_path)

        train_adv_acts_path = os.path.join(
            args.exp_dir,
            "activations",
            f"{args.name}",
            "train",
            f"epoch_{epoch + 1}_train_adv_activations.pkl",
        )
        utils.save_df_to_pkl(train_adv_acts, train_adv_acts_path)

        train_area_acts_path = os.path.join(
            args.exp_dir,
            "activations",
            f"{args.name}",
            "train",
            f"epoch_{epoch + 1}_train_area_activations.pkl",
        )
        utils.save_df_to_pkl(train_area_acts, train_area_acts_path)

        val_json_path = os.path.join(
            args.exp_dir,
            "metrics",
            f"{args.name}",
            "val",
            f"epoch_{epoch + 1}_val_f1.json",
        )
        utils.save_dict_to_json(val_stats, val_json_path)

        val_adv_acts_path = os.path.join(
            args.exp_dir,
            "activations",
            f"{args.name}",
            "val",
            f"epoch_{epoch + 1}_val_adv_activations.pkl",
        )
        utils.save_df_to_pkl(val_adv_acts, val_adv_acts_path)

        val_area_acts_path = os.path.join(
            args.exp_dir,
            "activations",
            f"{args.name}",
            "val",
            f"epoch_{epoch + 1}_val_area_activations.pkl",
        )
        utils.save_df_to_pkl(val_area_acts, val_area_acts_path)

        if is_train_best:
            best_train_macro_f1 = train_macro_f1_adv
            train_stats["epoch"] = epoch + 1

            train_json_path = os.path.join(
                args.exp_dir,
                "metrics",
                f"{args.name}",
                "train",
                "best_train_f1.json",
            )
            utils.save_dict_to_json(train_stats, train_json_path)

            best_adv_acts_path = os.path.join(
                args.exp_dir,
                "activations",
                f"{args.name}",
                "train",
                "best_train_adv_activations.pkl",
            )
            utils.save_df_to_pkl(train_adv_acts, best_adv_acts_path)

            best_area_acts_path = os.path.join(
                args.exp_dir,
                "activations",
                f"{args.name}",
                "train",
                "best_train_area_activations.pkl",
            )
            utils.save_df_to_pkl(train_area_acts, best_area_acts_path)

        if is_val_best:
            best_val_macro_f1 = val_macro_f1_adv
            val_stats["epoch"] = epoch + 1

            val_json_path = os.path.join(
                args.exp_dir,
                "metrics",
                f"{args.name}",
                "val",
                "best_val_f1.json",
            )
            utils.save_dict_to_json(val_stats, val_json_path)

            best_adv_acts_path = os.path.join(
                args.exp_dir,
                "activations",
                f"{args.name}",
                "val",
                "best_val_adv_activations.pkl",
            )
            utils.save_df_to_pkl(val_adv_acts, best_adv_acts_path)

            best_area_acts_path = os.path.join(
                args.exp_dir,
                "activations",
                f"{args.name}",
                "val",
                "best_val_area_activations.pkl",
            )
            utils.save_df_to_pkl(val_area_acts, best_area_acts_path)

            logging.info(
                (
                    "New best adv val macro F1 found\n"
                    "=====Adv====\n"
                    "============\n"
                    f"val macro F1: {val_macro_f1_adv:0.5f}\n"
                    f"train macro F1: {train_macro_f1_adv:0.5f}\n"
                    f"Avg val loss: {val_stats['loss_adv']:0.5f}\n"
                    f"Avg train loss: {train_stats['loss_adv']:0.5f}\n"
                    "====Area====\n"
                    "============\n"
                    f"val macro F1: {val_macro_f1_area:0.5f}\n"
                    f"train macro F1: {train_macro_f1_area:0.5f}\n"
                    f"Avg val loss: {val_stats['loss_area']:0.5f}\n"
                    f"Avg train loss: {train_stats['loss_area']:0.5f}\n"
                    "===Overall==\n"
                    "============\n"
                    f"Avg val loss: {val_stats['loss_avg']:0.5f}\n"
                    f"Avg train loss: {train_stats['loss_avg']:0.5f}\n"
                )
            )

            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
            }

            utils.save_checkpoint(
                state,
                is_val_best,
                os.path.join(args.exp_dir, "model_states", f"{args.name}"),
                (epoch + 1) % params.save_every == 0,
            )

    # Save last epoch stats
    utils.save_checkpoint(
        state,
        is_val_best,
        os.path.join(args.exp_dir, "model_states", f"{args.name}"),
        True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dirs",
        nargs="+",
        type=str,
        default=["data/"],
        help=("Directory containing training " "and validation cases."),
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

    # Setting data paths
    train_paths = []
    val_paths = []
    for path in args.data_dirs:
        train_paths.append(os.path.join(path, "train"))
        val_paths.append(os.path.join(path, "validation"))

    targets_paths = {
        "adv": args.targets_paths_advs,
        "area": args.targets_paths_areas,
    }

    unique_labels = {
        "adv": args.unique_labels_advs,
        "area": args.unique_labels_areas,
    }

    # Datasets
    train_dataset = MultiTaskDataset(
        data_paths=train_paths,
        targets_paths=targets_paths,
        unique_labels=unique_labels,
    )

    val_dataset = MultiTaskDataset(
        data_paths=val_paths,
        targets_paths=targets_paths,
        unique_labels=unique_labels,
    )

    logging.info(
        (
            "Training with "
            f"{len(train_dataset.unique_labels['adv'])} adv targets"
        )
    )
    logging.info(
        (
            "Training with "
            f"{len(train_dataset.unique_labels['area'])} area targets"
        )
    )
    logging.info(f"Training on {len(train_dataset)} datapoints")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )

    train_check_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = SimpleMultiTaskMultiLabelPrediction(
        labels=train_dataset.unique_labels,
        max_length=params.max_length,
        truncation_side=params.truncation_side,
        model_name=args.embed_model_name,
        mode="train",
        device=args.device,
    )

    model.to(args.device)

    is_regression = torch.Tensor([False, False])
    multiTaskLoss = MultiTaskLoss(is_regression=is_regression, reduction="sum")
    multiTaskLoss.to(args.device)

    # Defining optimizer and loss function
    parameters = list(model.parameters()) + list(multiTaskLoss.parameters())

    optimizer = optim.Adam(parameters, lr=params.lr)

    logging.info("Calculating positive weights for loss for adv")

    target_counts_adv = Counter(
        chain.from_iterable(
            train_dataset.targets_dict[v]["adv"]
            for v in train_dataset.idx.values()
        )
    )
    logging.info(f"Number of positives for adv classes: {target_counts_adv}")

    pos_weight_adv = [
        (1.0 - target_counts_adv[k] * 1 / len(train_dataset))
        * (len(train_dataset) * 1.0 / target_counts_adv.get(k, 1))
        for k in train_dataset.unique_labels["adv"]
    ]

    pos_weight_adv = torch.FloatTensor(pos_weight_adv)
    pos_weight_adv.to(args.device)
    logging.info(
        f"Calculated positive weights for advocates are: {pos_weight_adv}"
    )
    loss_adv = nn.BCEWithLogitsLoss(
        reduction="sum", pos_weight=pos_weight_adv
    ).to(args.device)

    logging.info("Calculating positive weights for loss for area")

    target_counts_area = Counter(
        chain.from_iterable(
            train_dataset.targets_dict[v]["area"]
            for v in train_dataset.idx.values()
        )
    )
    logging.info(f"Number of positives for area classes: {target_counts_area}")

    pos_weight_area = [
        (1.0 - target_counts_area[k] * 1 / len(train_dataset))
        * (len(train_dataset) * 1.0 / target_counts_area.get(k, 1))
        for k in train_dataset.unique_labels["area"]
    ]

    pos_weight_area = torch.FloatTensor(pos_weight_area)
    pos_weight_area.to(args.device)
    logging.info(
        f"Calculated positive weights for areas are: {pos_weight_area}"
    )
    loss_area = nn.BCEWithLogitsLoss(
        reduction="sum", pos_weight=pos_weight_area
    ).to(args.device)

    loss_fn = {"area": loss_area, "adv": loss_adv}

    train_and_evaluate(
        model,
        optimizer,
        loss_fn,
        multiTaskLoss,
        train_loader,
        train_check_loader,
        val_loader,
        params,
        metrics,
        train_dataset.unique_labels,
        args,
    )

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
