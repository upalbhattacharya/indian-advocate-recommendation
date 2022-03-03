#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-02-22 17:36:32.981007137 +0530
# Modify: 2022-02-24 17:32:43.780586753 +0530

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

    criterion = loss_fn
    # Training loop for one epoch
    for data, target in iter(data_loader.yield_batch()):

        logging.info(f"Shape of input {data.shape}")
        data = data.to(args.device)
        target = target.to(args.device)
        y_pred = model(data)
        
        logging.info(f"Shape of results {y_pred.shape}")

        # Clear previous gradients and update weights
        optimizer.zero_grad()
        loss = criterion(y_pred.float(), target.float())
        loss.backward(retain_graph=False)
        optimizer.step()

        outputs_batch = (y_pred.data.cpu().detach().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)
        loss_batch.append(loss.item())

        # Deleting from memory to prevent memory overload
        del target
        del data
        del outputs_batch
        del targets_batch
        del y_pred

    outputs, targets = accumulate()
    summary_batch = {metric: metrics[metric](outputs, targets)
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
        logging.info(f"Logging for epoch {epoch}.")

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
                    f"epoch_{epoch}_train_f1.json")
            utils.save_dict_to_json(train_stats, train_json_path)

            test_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "test",
                    f"epoch_{epoch}_test_f1.json")
            utils.save_dict_to_json(test_stats, test_json_path)

        # Save training stats if it is the best
        if is_train_best:
            best_train_macro_f1 = train_macro_f1

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "train",
                    "best_train_f1.json")
            utils.save_dict_to_json(train_stats, best_json_path)

        # Save test stats if it is the best
        if is_test_best:
            best_test_macro_f1 = test_macro_f1

            logging.info(
                    (f"New best macro F1: {best_test_macro_f1:0.5f} "
                     f"Train macro F1: {train_macro_f1:0.5f} "
                     f"Avg test loss: {test_stats['loss_avg']} "
                     f"Avg train loss: {train_stats['loss_avg']}."))

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "test",
                    "best_test_f1.json")
            utils.save_dict_to_json(train_stats, best_json_path)

        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                }

        # Checking and saving checkpoint
        utils.save_checkpoint(state, is_test_best,
                              os.path.join(exp_dir, "model_states",
                                           f"{name}"),
                              epoch % params.save_every == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="data/",
                        help=("Directory containing training "
                              "and testing cases."))
    parser.add_argument("-t", "--targets_path",
                        default="targets/targets.json",
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
    parser.add_argument("-r", "--restore_file", default=None,
                        help=("Optional file to reload a saved model "
                              "and optimizer."))

    args = parser.parse_args()

    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}.log"))

    # Selecting the right device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

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

    # Creating the training and testing data generators
    train_loader = DataGenerator(
                            data_path=os.path.join(args.data_dir, "train"),
                            targets_path=args.targets_path,
                            embed_path=args.embed_path,
                            batch_size=params.batch_size,
                            max_sent_len=params.max_sent_len,
                            max_sent_num=params.max_sent_num)

    test_loader = DataGenerator(
                            data_path=os.path.join(args.data_dir, "test"),
                            targets_path=args.targets_path,
                            embed_path=args.embed_path,
                            batch_size=params.batch_size,
                            max_sent_len=params.max_sent_len,
                            max_sent_num=params.max_sent_num)

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
