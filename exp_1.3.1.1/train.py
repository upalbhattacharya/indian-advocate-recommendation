#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-21 12:06:17.571342513 +0530
# Modify: 2022-03-25 13:31:26.087212424 +0530

"""Training and evaluation methods for model."""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from data_generator import DataGenerator
from evaluate import evaluate
from metrics import metrics
from model.net import HANPrediction

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def train_one_epoch(model, optimizer, loss_fn, data_loader, params,
                    metrics, args):
    """Train model for one epoch. Return summary information of metrics
    and loss.
    """

    # Setting model to train
    model.train()

    # For accumulation of data for metrics
    accumulate = utils.Accumulate()
    loss_batch = []
    # Getting names of targets for visual understanding of saved metrics
    target_names = data_loader.unique_labels

    criterion = loss_fn
    # Training loop for one epoch
    for i, (data, target, gate) in enumerate(iter(data_loader.yield_batch())):

        logging.info(f"Shape of input {data.shape}")
        data = data.to(args.device)
        target = target.to(args.device)
        gate = gate.to(args.device)
        y_pred = model(data)
        y_pred = torch.mul(y_pred, gate)
        logging.info(f"Shape of results {y_pred.shape}")

        # Clear previous gradients and update weights
        loss = criterion(y_pred.float(), target.float())
        loss.backward()

        # Sub-batch behaviour to ensure better batch size for training
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

        outputs_batch = (y_pred.data.cpu().detach().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)

        # Deleting from memory to prevent memory overload
        del target
        del data
        del outputs_batch
        del targets_batch
        del y_pred
        torch.cuda.empty_cache()

    else:
        # for last batch if it does not update the parameter weights
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
                       test_loader, params, args, exp_dir,
                       name, metrics, restore_file=None):

    # Set default start epoch
    start_epoch = 0

    # Setting best macro F1 for train and test
    best_test_macro_f1 = 0.0
    best_train_macro_f1 = 0.0

    # Checking if a restore file is provided
    if restore_file is not None:
        restore_path = os.path.join(
            exp_dir, f"{restore_file}.pth.tar")

        logging.info("Found checkpoint at {restore_path}. Loading.")

        start_epoch = utils.load_checkpoint(restore_path, model, optimizer) + 1
    # Train over the required number of epochs
    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Logging for epoch {epoch + 1}.")

        _ = train_one_epoch(model, optimizer, loss_fn,
                            train_loader, params, metrics, args)

        test_stats = evaluate(model, loss_fn,
                              test_loader, params, metrics, args)

        # Getting the stats for the training data
        train_stats = evaluate(model, loss_fn,
                               train_loader, params, metrics, args)

        # Getting the macro stats for train and test
        train_macro_f1 = train_stats['f1']['macro_f1']
        is_train_best = train_macro_f1 >= best_train_macro_f1

        test_macro_f1 = test_stats['f1']['macro_f1']
        is_test_best = test_macro_f1 >= best_test_macro_f1

        logging.info(
                (f"Test macro F1: {test_macro_f1:0.5f} "
                 f"Train macro F1: {train_macro_f1:0.5f} "
                 f"Avg test loss: {test_stats['loss_avg']} "
                 f"Avg train loss: {train_stats['loss_avg']}."))

        # Save stats
        if (epoch % params.save_every == 0):
            train_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "train",
                    f"epoch_{epoch + 1}_train_f1.json")
            utils.save_dict_to_json(train_stats, train_json_path)

            test_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "test",
                    f"epoch_{epoch + 1}_test_f1.json")
            utils.save_dict_to_json(test_stats, test_json_path)

        # Save training stats if it is the best
        if is_train_best:
            best_train_macro_f1 = train_macro_f1
            train_stats["epoch"] = (epoch + 1)

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "train",
                    "best_train_f1.json")
            utils.save_dict_to_json(train_stats, best_json_path)

        # Save test stats if it is the best
        if is_test_best:
            best_test_macro_f1 = test_macro_f1
            test_stats["epoch"] = (epoch + 1)

            logging.info(
                    (f"New best macro F1: {best_test_macro_f1:0.5f} "
                     f"Train macro F1: {train_macro_f1:0.5f} "
                     f"Avg test loss: {test_stats['loss_avg']} "
                     f"Avg train loss: {train_stats['loss_avg']}."))

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "test",
                    "best_test_f1.json")
            utils.save_dict_to_json(test_stats, best_json_path)

        state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                }

        # Checking and saving checkpoint
        utils.save_checkpoint(state, is_test_best,
                              os.path.join(exp_dir, "model_states",
                                           f"{name}"),
                              (epoch + 1) % params.save_every == 0)

    # Checking and saving checkpoint for last epoch
    utils.save_checkpoint(state, is_test_best,
                          os.path.join(exp_dir, "model_states",
                                       f"{name}"),
                          True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dirs", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and testing cases."))
    parser.add_argument("-t", "--targets_paths", nargs="+", type=str,
                        default=["targets/targets.json"],
                        help="Path to targets json file.")
    parser.add_argument("-e", "--embed_path",
                        default="embeddings/embeddings.pkl",
                        help="Path to embeddings json file.")
    parser.add_argument("-x", "--exp_dir", default="experiments/",
                        help=("Directory to load parameters and to save "
                              "model states and metrics."))
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of model.")
    parser.add_argument("-p", "--params", default="params.json",
                        help="Name of params file to load from exp_dir.")
    parser.add_argument("-de", "--device", type=str, default="cuda",
                        help="Device to train on.")
    parser.add_argument("-id", "--device_id", type=int, default=0,
                        help="Device ID to run on if using GPU.")
    parser.add_argument("-r", "--restore_file", default=None,
                        help=("Optional file to reload a saved model "
                              "and optimizer."))
    parser.add_argument("-ul", "--unique_labels", nargs="+", type=str,
                        required=True, help="Labels to use as targets.")

    args = parser.parse_args()

    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}.log"))

    # Selecting the right device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    # Setting the device id
    if args.device == "cuda":
        args.device = f"cuda:{args.device_id}"

    logging.info(f"Device is {args.device}.")

    # Loading model parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(
            params_path), f"No configuration file found at {params_path}."
    params = utils.Params(params_path)

    # Setting a seed for reproducability
    torch.manual_seed(47)
    if(args.device == "cuda"):
        torch.cuda.manual_seed(47)

    # Setting the data path
    train_paths = [os.path.join(path, "train") for path in args.data_dirs]
    test_paths = [os.path.join(path, "test") for path in args.data_dirs]

    # Creating the training and testing data generators
    train_loader = DataGenerator(
                            data_paths=train_paths,
                            targets_paths=args.targets_paths,
                            embed_path=args.embed_path,
                            batch_size=params.batch_size,
                            max_sent_len=params.max_sent_len,
                            max_sent_num=params.max_sent_num,
                            unique_labels=args.unique_labels,
                            neg_ratio=params.neg_ratio)

    test_loader = DataGenerator(
                            data_paths=test_paths,
                            targets_paths=args.targets_paths,
                            embed_path=args.embed_path,
                            batch_size=params.batch_size,
                            max_sent_len=params.max_sent_len,
                            max_sent_num=params.max_sent_num,
                            unique_labels=args.unique_labels,
                            neg_ratio=params.neg_ratio)

    model = HANPrediction(input_size=params.input_size,
                          hidden_dim=params.hidden_dim,
                          labels=train_loader.unique_labels,
                          device=args.device)

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCELoss(reduction='sum')

    train_and_evaluate(model, optimizer, loss_fn, train_loader,
                       test_loader, params, args, args.exp_dir,
                       args.name, metrics, restore_file=args.restore_file)

    logging.info("="*80)


if __name__ == "__main__":
    main()
